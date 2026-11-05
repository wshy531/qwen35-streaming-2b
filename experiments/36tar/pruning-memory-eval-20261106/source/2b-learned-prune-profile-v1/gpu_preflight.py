"""Three train-only clips, three scorer updates; no capability estimate."""
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT/'runs/learned_prune_profile_prepare_20261101_01'
for name in ['2b-sember-v1', '2b-resolution-cost-v1', '2b-learned-prune-mechanism-v1']:
    sys.path.insert(0, str(ROOT/'experiments'/name))
from sember_common import runtime
from resolution_runtime import make_builder
from audit_media import canvas
from bias_prune import GroupScorer, encode
from training import task_loss, question_text


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def decode(env, path, audit, builder):
    desired = {s['frame_index']: s for s in audit['samples']}; frames = []
    with env['av'].open(str(path)) as c:
        stream = c.streams.video[0]; stream.thread_count = 1
        for i, frame in enumerate(c.decode(stream)):
            if i not in desired: continue
            image = canvas(frame); spec = desired[i]
            assert hashlib.sha256(image.tobytes()).hexdigest() == spec['rgb_sha256']
            frames.append(env['FramePacket'](frame_id=i, timestamp=spec['pts_s'], rgb=env['np'].asarray(image), decoded_count=i+1))
    assert len(frames) == len(desired)
    pairs = []
    for i in range(0, len(frames), 2):
        a, b = frames[i], frames[min(i+1, len(frames)-1)]
        inputs, _ = builder.pair(a, b, 0, 0, synthetic_padding=i+1 == len(frames))
        pairs.append((a.timestamp, b.timestamp, inputs))
    return pairs


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK') == '4'
    for name, digest in read(PREP/'execution_lock.json').items(): assert sha(ROOT/name) == digest, name
    assert not args.out.exists() and args.out.resolve().is_relative_to((ROOT/'runs').resolve())
    env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261101)
    assert t.cuda.is_available() and t.cuda.device_count()==1
    assert env['psutil'].virtual_memory().available >= 12*2**30
    args.out.mkdir();started=time.monotonic()
    model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
        dtype=t.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model.eval().requires_grad_(False)
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    scorer=GroupScorer(model.config.vision_config.hidden_size).to('cuda')
    optimizer=t.optim.AdamW(scorer.parameters(),lr=1e-4,weight_decay=0.)
    versions={n:p._version for n,p in model.named_parameters()};checks=[];calls=0
    audit={r['video_id']:r for r in read(ROOT/'runs/learned_prune_data_prepare_20261101_01/CPU_media_audit.json')['rows']}
    for row in read(PREP/'selection.json'):
        if time.monotonic()-started > 12*60: raise TimeoutError('profile fixed deadline')
        assert row['source']['role']=='train' and row['question']['role']=='train'
        path=ROOT/'datasets/learned-prune-nextqa-v1'/row['source']['mirror_path']
        assert sha(path)==row['source']['mirror_lfs_sha256']
        pairs=decode(env,path,audit[row['source']['video_id']],builder)
        def forward():
            nonlocal calls
            records=[];state=None
            with env['fp32_reference']():
                for index,(begin,end,batch) in enumerate(pairs):
                    batch=batch.to('cuda')
                    features,ids,_,state=encode(model.model.visual,batch.pixel_values_videos,batch.video_grid_thw,scorer,previous=state)
                    records.append((index,begin,end,features,ids));calls+=1
                return task_loss(model,builder,records,question_text(row['question']),chr(65+row['question']['answer_index']))
        optimizer.zero_grad(set_to_none=True);t.cuda.empty_cache();t.cuda.reset_peak_memory_stats();t.cuda.synchronize();tick=time.monotonic()
        before={n:p.detach().clone() for n,p in scorer.named_parameters()}
        loss,tokens=forward();assert t.isfinite(loss);loss.backward()
        grad=float(t.sqrt(sum((p.grad**2).sum() for p in scorer.parameters() if p.grad is not None)))
        assert grad>1e-12 and all(p.grad is None or t.isfinite(p.grad).all() for p in scorer.parameters())
        assert all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
        t.nn.utils.clip_grad_norm_(scorer.parameters(),1.);optimizer.step();t.cuda.synchronize()
        assert any(not t.equal(p,before[n]) for n,p in scorer.named_parameters())
        checks.append(dict(video_id=row['source']['video_id'],pairs=len(pairs),input_tokens=tokens,loss=float(loss.detach()),
            scorer_gradient_norm=grad,seconds=time.monotonic()-tick,peak_allocated_bytes=t.cuda.max_memory_allocated()))
        del loss,before,pairs;gc.collect();t.cuda.empty_cache()
        print(json.dumps(checks[-1]),flush=True)
    t.save(scorer.state_dict(),args.out/'scorer_profile_only.pt')
    result=dict(status='passed_not_capability_evidence',steps=checks,updates=3,vision_calls=calls,
        trainable_parameters=sum(p.numel() for p in scorer.parameters()),base_unchanged=True,generated_tokens=0,
        pretrained_loads=1,seconds=time.monotonic()-started)
    (args.out/'completed.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
