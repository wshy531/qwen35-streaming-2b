"""Single authorized diagnostic run, no architecture or parameter search."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from prepare import ROOT,read,sha
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
sys.path.insert(0,str(ROOT/'experiments/2b-long-stream-v1'))


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def verify(run):
    assert run.resolve().is_relative_to((ROOT/'runs').resolve())
    protocol=read(run/'protocol.json')
    for name,h in protocol['inputs_sha256'].items():assert sha(run/name)==h,name
    for name,h in read(run/'runtime_lock.json').items():assert sha(ROOT/name)==h,name
    rows=read(run/'requests.json');assert len(rows)==len({r['id'] for r in rows})==6
    for r in rows:
        assert not Path(r['media_path']).is_absolute() and '..' not in Path(r['media_path']).parts
        assert hashlib.sha256(json.dumps(r['committed_frames'],sort_keys=True).encode()).hexdigest()==r['committed_frames_sha256']
    return protocol,rows


def worker(run,out):
    protocol,rows=verify(run);auth=read(run/'GPU_authorization.json')
    assert auth['approved'] is True and auth['protocol_sha256']==sha(run/'protocol.json')
    proof=read(run/'cpu_cluster.json');assert proof['status']=='passed' and proof['protocol_sha256']==sha(run/'protocol.json')
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
    from runtime_bridge import runtime
    env=runtime('memory');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    assert torch.cuda.mem_get_info()[0]>=24*2**30 and env['psutil'].virtual_memory().available>=12*2**30
    from online_bank import OnlineBank,nodes_digest
    from replay import VideoSource
    from diagnostic import choices,answer,ARMS
    from score_memory_eval import prompt
    out.mkdir(exist_ok=False);start=time.monotonic();count=generated=encodes=0;peak_RSS=0;snapshots=[]
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    assert model.config.text_config.hidden_size==2048
    save(out/'loaded.json',dict(pretrained_loads=1,seconds=time.monotonic()-start))
    build=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    def guard():
        nonlocal peak_RSS
        peak_RSS=max(peak_RSS,env['psutil'].Process().memory_info().rss)
        assert peak_RSS<20*2**30 and torch.cuda.memory_allocated()<32*2**30
        assert time.monotonic()-start<1500,'one bounded diagnostic worker'
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as predictions,(out/'tokens.jsonl').open('x',encoding='utf-8') as tokens,torch.inference_mode(),env['fp32_reference']():
        for number,row in enumerate(rows):
            fullspec=dict(source_path=row['source_path'],media_sha256=row['media_sha256'],frames=row['committed_frames'])
            source=VideoSource(ROOT/row['media_path'],fullspec,env['FramePacket'])
            it=iter(source.frames());raw=[];pending=[];bank=OnlineBank('M2');prefix=hashlib.sha256(b'[');seen=0
            try:
                for _ in range(len(row['committed_frames'])):
                    frame=next(it);packet=source.buffer.pop(frame.frame_id);pending.append(packet)
                    m=row['committed_frames'][seen]
                    if seen:prefix.update(b', ')
                    prefix.update(json.dumps(m,sort_keys=True).encode());seen+=1
                    if len(pending)<2:continue
                    a,b=pending;pending=[]
                    inp,_=build.pair(a,b,0,0);inp=inp.to('cuda')
                    vision=model.model.get_video_features(inp.pixel_values_videos,inp.video_grid_thw,return_dict=True)
                    features=torch.cat(vision.pooler_output,0)
                    block=env['load']().features.raw_block(features,len(raw),a.timestamp,b.timestamp)
                    raw.append(env['load']().forest.raw(block))
                    bank.append(features,len(raw)-1,a.timestamp,b.timestamp);encodes+=1
                    del inp,vision,features,block,a,b,packet;guard()
            finally:it.close();source.buffer.clear()
            assert not pending and len(raw)==row['total_prefix_pairs']
            prefix.update(b']');assert prefix.hexdigest()==row['committed_frames_sha256']
            assert sha(ROOT/row['media_path'])==row['media_sha256']
            packed=bank.export();actual=[n.metadata() for n in packed]
            assert actual==row['expected_bank_nodes'],'bank replay differs; stop, no forced boundaries'
            snapshot=out/f'bank-{number:02d}.npz';arrays={}
            for i,n in enumerate(packed):
                for name in ('features','mass','coords'):arrays[f'n{i}_{name}']=getattr(n,name).numpy()
            env['np'].savez_compressed(snapshot,**arrays)
            snapshots.append(dict(id=row['id'],path=snapshot.name,sha256=sha(snapshot),nodes=actual,
                temporary_raw_feature_bytes=sum(t.numel()*t.element_size() for n in raw for t in (n.features,n.mass,n.coords)),
                raw_history_is_diagnostic_not_bounded_candidate=True))
            arms=choices(raw,packed,row);hashes={a:nodes_digest(ns) for a,ns in arms.items()}
            for arm in ARMS:
                emitted=[]
                def callback(t):
                    emitted.append(t['token_id']);tokens.write(json.dumps(dict(id=row['id'],arm=arm,**t))+'\n');tokens.flush()
                result=answer(env,model,build,arms[arm],prompt(row),callback)
                assert result['generated_ids']==emitted
                predictions.write(json.dumps(dict(id=row['id'],arm=arm,status='ok',**result))+'\n');predictions.flush()
                count+=1;generated+=len(result['generated_ids']);guard()
                assert all(nodes_digest(arms[a])==h for a,h in hashes.items())
            assert sum(p.stat().st_size for p in out.glob('bank-*.npz'))<24*2**20
            del raw,packed,bank,arms,arrays,source;gc.collect();torch.cuda.empty_cache()
            print(dict(completed_cases=number+1,answers=count),flush=True)
    assert count==36 and generated<=288 and encodes==sum(r['total_prefix_pairs'] for r in rows)
    save(out/'snapshots.json',snapshots)
    save(out/'completed.json',dict(status='complete',answers=count,generated_tokens=generated,visual_encodings=encodes,
        pretrained_loads=1,seconds=time.monotonic()-start,peak_RSS_observed_bytes=peak_RSS,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),protocol_sha256=sha(run/'protocol.json'),
        candidate_default_unchanged=True,wallclock_test=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();worker(a.run,a.out)
