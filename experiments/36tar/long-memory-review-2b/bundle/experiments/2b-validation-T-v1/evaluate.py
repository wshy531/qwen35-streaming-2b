"""Fixed T comparison: CPU preflight and explicitly authorized Slurm workers.

Reuses frozen model/forest code. The native adapter changes only the explicit
complete-input guard from3072 to8192; it never truncates video or questions.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT/'experiments/2b-natural-eval-v1'
SHORT = ROOT/'experiments/2b-short-event-v1'
sys.path.insert(0,str(BASE))
sys.path.append(str(SHORT))
from runtime_bridge import runtime


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.write('\n')


def verify(run):
    assert run.resolve().is_relative_to((ROOT/'runs').resolve())
    protocol = read(run/'runtime_protocol.json')
    assert sha(run/'requests.json') == protocol['requests_sha256']
    assert sha(run/'runtime_lock.json') == protocol['runtime_lock_sha256']
    for rel,digest in read(run/'runtime_lock.json').items():
        assert sha(ROOT/rel) == digest, rel
    rows = read(run/'requests.json')
    assert len(rows) == len({r['id'] for r in rows}) == protocol['questions'] == 98
    groups = defaultdict(list)
    for row in rows:
        assert set(row) == {'id','cohort','media_path','media_sha256','model_visible'}
        assert set(row['model_visible']) == {'question','options','query_time_s','frames'}
        assert Path(row['media_path']).parts[:2] == ('data','T-new40-20261020')
        assert not Path(row['media_path']).is_absolute() and '..' not in Path(row['media_path']).parts
        groups[row['media_path']].append(row)
    assert len(groups) == 39
    for items in groups.values():
        first = items[0]['model_visible']
        assert all(r['model_visible']['frames']==first['frames'] and
                   r['model_visible']['query_time_s']==first['query_time_s'] for r in items)
        assert all(f['pts_s']<=first['query_time_s']+1e-9 for f in first['frames'])
    identity = read(ROOT/'runs/2b-natural-eval-inputs-20261020-01/cpu_memory/preflight.json')['identity']
    active = ROOT/'repo/experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    for path,key in [(ROOT/'repo/qwen35_local/modeling_qwen3_5.py','original_model_sha256'),
                     (active/'qwen35_local/modeling_qwen3_5.py','stream_model_sha256'),
                     (active/'streaming/online.py','online_sha256'),
                     (active/'streaming/hybrid.py','hybrid_sha256'),
                     (ROOT/'models/Qwen3.5-2B/DOWNLOAD_MANIFEST.json','manifest_sha256')]:
        assert sha(path) == identity[key], key
    return protocol,groups


def builder(env):
    return env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(
        ROOT/'models/Qwen3.5-2B',local_files_only=True),canvas=(192,256),resize_mode='letterbox')


def native_inputs(env,model,build,packets,specs,metadata,question,device):
    # Same processor, template, pixels, metadata and official RoPE oracle-grid
    # adaptation as frozen fixed_stage2.native_inputs. Only guard is larger.
    torch = env['torch']
    chosen = [packets[f['frame_id']] for f in specs]
    if len(chosen)%2:
        chosen.append(chosen[-1])
    meta = dict(metadata,frames_indices=[f.frame_id for f in chosen])
    inputs = build.processor(text=[build.template(question)],
        videos=[env['np'].stack([f.rgb for f in chosen])],video_metadata=[meta],
        do_resize=False,do_sample_frames=False,return_tensors='pt').to(device)
    assert inputs.input_ids.shape[1]+8 <= 8192, 'complete native input exceeds8192; no truncation'
    oracle = torch.repeat_interleave(inputs.video_grid_thw,inputs.video_grid_thw[:,0],0).clone()
    oracle[:,0] = 1
    coords,delta = model.model.get_rope_index(**dict(inputs,video_grid_thw=oracle))
    inputs['position_ids'] = torch.cat((torch.arange(inputs.input_ids.shape[1],device=device)[None,None,:],coords),0)
    return inputs,delta,dict(input_tokens=inputs.input_ids.shape[1],
        visual_tokens=int((inputs.input_ids==build.processor.video_token_id).sum()),
        tail_duplicate=bool(len(specs)%2),full_context_guard=8192,
        position_adapter='inherited official per-temporal-group oracle grid')


def banks(env,nodes):
    import fixed_stage2 as fixed
    from policies import CappedForest,ReservoirRecent,block
    exports,status,events = fixed.policies(env,nodes)
    capped,reservoir = CappedForest(),ReservoirRecent()
    for node,event in zip(nodes,events):
        capped.append(block(node),starts_event=event['starts_event'])
        reservoir.append(node)
    choices = dict(full_pairwise_reference=nodes,**exports,
                   B02_cap4=capped.export(),reservoir4=reservoir.export(),text_only=[])
    status.update(B02_cap4=capped.status(),reservoir4=reservoir.status())
    assert all(sum(len(n.features) for n in ns)<=384
               for name,ns in choices.items() if name!='full_pairwise_reference')
    return choices,status,events


def cpu(run,stage):
    protocol,groups = verify(run)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    env = runtime(stage)
    torch = env['torch']
    torch.set_num_threads(1)
    torch.manual_seed(123)
    assert not torch.cuda.is_available()
    import fixed_stage2 as fixed
    from capability_pilot import answer
    build = builder(env)
    model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](
        **env['tiny_config'](build.processor),attn_implementation='sdpa')).eval()
    count,maximum,pixel_sources,tiny = 0,0,0,0
    tested = set()
    started = time.monotonic()
    with torch.inference_mode(),env['fp32_reference']():
        for media,rows in groups.items():
            specs = rows[0]['model_visible']['frames']
            if stage == 'native':
                packets,metadata = fixed.decode(env,rows)
                for j,row in enumerate(rows):
                    inputs,delta,meta = native_inputs(env,model,build,packets,specs,metadata,
                                                     env['prompt'](row['model_visible']),'cpu')
                    maximum = max(maximum,meta['input_tokens'])
                    if j == 0:
                        pixels = []
                        for i in range(0,len(specs),2):
                            a,b = specs[i],specs[min(i+1,len(specs)-1)]
                            pair,_ = build.pair(packets[a['frame_id']],packets[b['frame_id']],0,0)
                            pixels.append(pair.pixel_values_videos)
                        assert torch.equal(inputs.pixel_values_videos,torch.cat(pixels))
                        pixel_sources += 1
                        del pixels,pair
                    if not tiny:
                        model.model.rope_deltas = delta
                        _,first = env['complete'](model,build.processor,dict(inputs),1)
                        assert env['np'].isfinite(first).all()
                        tiny += 1
                    count += 1
                    del inputs,delta
                del packets
            else:
                legacy = env['load']()
                nodes = []
                for i in range(0,len(specs),2):
                    a,b = specs[i],specs[min(i+1,len(specs)-1)]
                    nodes.append(legacy.forest.raw(legacy.features.raw_block(
                        torch.randn(48,32),i//2,a['pts_s'],b['pts_s'])))
                choices,_,_ = banks(env,nodes)
                for row in rows:
                    question = env['prompt'](row['model_visible'])
                    for arm,ns in choices.items():
                        before = fixed.node_digest(ns)
                        if ns:
                            _,_,_,meta = env['context'](model,build,ns,question,
                                limits=env['Limits'](8192,6144,128),max_new_tokens=8)
                            maximum = max(maximum,meta['input_tokens'])
                        if arm not in tested:
                            answer(env,model,build,ns,question)
                            tested.add(arm)
                            tiny += 1
                        assert before == fixed.node_digest(ns)
                        count += 1
                del nodes,choices
            assert time.monotonic()-started < 180, 'bounded CPU preflight'
            gc.collect()
    assert count == (98 if stage=='native' else 686)
    save(run/f'cpu_{stage}_preflight.json',dict(status='passed',stage=stage,contexts=count,
        tiny_answers=tiny,actual_pixel_equal_sources=pixel_sources,max_input_tokens=maximum,
        seconds=time.monotonic()-started,pretrained_model_loads=0,
        protocol_sha256=sha(run/'runtime_protocol.json'),runtime_lock_sha256=sha(run/'runtime_lock.json')))
    print(dict(status='passed',stage=stage,contexts=count,tiny_answers=tiny,max_input_tokens=maximum),flush=True)


def worker(run,stage,out):
    protocol,groups = verify(run)
    auth = read(run/'GPU_authorization.json')
    assert auth['approved'] is True and auth['protocol_sha256']==sha(run/'runtime_protocol.json')
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
    for name in ['native','memory']:
        proof = read(run/f'cpu_{name}_preflight.json')
        assert proof['status']=='passed' and proof['protocol_sha256']==sha(run/'runtime_protocol.json')
    env = runtime(stage)
    torch = env['torch']
    torch.set_num_threads(1)
    torch.manual_seed(123)
    assert torch.cuda.mem_get_info()[0]>=16*2**30 and env['psutil'].virtual_memory().available>=12*2**30
    import fixed_stage2 as fixed
    from capability_pilot import answer
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    model = env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    assert model.config.text_config.hidden_size==2048 and next(model.parameters()).dtype==torch.float32
    save(out/'loaded.json',dict(loads=1,seconds=time.monotonic()-started))
    build = builder(env)
    count,generated,timelines = 0,0,[]
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as stream,torch.inference_mode(),env['fp32_reference']():
        for media,rows in groups.items():
            packets,metadata = fixed.decode(env,rows)
            specs = rows[0]['model_visible']['frames']
            if stage == 'memory':
                nodes = fixed.encode(env,model,build,packets,specs)
                choices,status,events = banks(env,nodes)
                hashes = {a:fixed.node_digest(ns) for a,ns in choices.items()}
                timelines.append(dict(media=media,states=status,events=events,
                    full_reference_tokens=sum(len(n.features) for n in nodes),bank_digests=hashes))
            for row in rows:
                for arm in protocol['arms'][stage]:
                    assert time.monotonic()-started < 1470, 'worker1500s bound with cleanup reserve'
                    tick = time.monotonic()
                    try:
                        question = env['prompt'](row['model_visible'])
                        if stage=='native':
                            saved = model.model.rope_deltas
                            try:
                                inputs,delta,meta = native_inputs(env,model,build,packets,specs,metadata,question,'cuda')
                                model.model.rope_deltas = delta
                                result,first = env['complete'](model,build.processor,dict(inputs),8)
                                assert env['np'].isfinite(first).all()
                                value = dict(text=result['answer'],generated_ids=result['generated_ids'],**meta,
                                             first_logits_sha256=hashlib.sha256(first.tobytes()).hexdigest())
                                del inputs,delta,first
                            finally:
                                model.model.rope_deltas = saved
                        else:
                            value = answer(env,model,build,choices[arm],question)
                        item = dict(id=row['id'],cohort=row['cohort'],arm=arm,status='ok',**value)
                    except (ValueError,FloatingPointError) as error:
                        item = dict(id=row['id'],cohort=row['cohort'],arm=arm,status='readout_error',
                                    text='',generated_ids=[],error=str(error))
                    item.update(seconds=time.monotonic()-tick,
                        input_frames_sha256=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest())
                    stream.write(json.dumps(item,ensure_ascii=False)+'\n')
                    stream.flush()
                    count += 1
                    generated += len(item['generated_ids'])
                    if stage=='memory':
                        assert all(fixed.node_digest(choices[a])==h for a,h in hashes.items()), 'query mutation'
                    assert torch.cuda.memory_allocated()<24*2**30 and env['psutil'].Process().memory_info().rss<20*2**30
            if stage=='memory':
                del nodes,choices
            del packets
            gc.collect()
            torch.cuda.empty_cache()
            print(dict(stage=stage,completed=count),flush=True)
    assert count == (98 if stage=='native' else 686)
    save(out/'timelines.json',timelines)
    save(out/'completed.json',dict(status='complete',stage=stage,records=count,generated_tokens=generated,
        seconds=time.monotonic()-started,model_loads=1,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        protocol_sha256=sha(run/'runtime_protocol.json'),no_truncation=True,query_bank_isolation=True))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['cpu','worker'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--stage',choices=['native','memory'],required=True)
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    if args.mode=='cpu':
        cpu(args.run,args.stage)
    else:
        worker(args.run,args.stage,args.out)
