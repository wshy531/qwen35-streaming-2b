"""One authorized fixed streamed comparison; no history feature archive."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from policy import ROOT
sys.path.insert(0,str(ROOT/'experiments/2b-long-stream-v1'))
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def verify(run,*,local=False):
    assert run.resolve().is_relative_to((ROOT/'runs').resolve())
    protocol=read(run/'protocol.json')
    for name,h in protocol['input_sha256'].items():assert sha(run/name)==h,name
    for name,h in read(run/'runtime_lock.json').items():
        p=ROOT/name if not local or not name.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(p)==h,name
    rows=read(run/'requests.json');sources=read(run/'sources.json');snapshots=read(run/'snapshots.json')
    assert len(rows)==len({q['id'] for q in rows})==34 and len(sources)==6
    assert len(snapshots)==34 and sum(len(s['frames'])//2 for s in sources)==1426
    return protocol,rows,sources,snapshots


def worker(run,out):
    protocol,rows,sources,snapshots=verify(run)
    auth=read(run/'GPU_authorization.json');assert auth['approved'] and auth['protocol_sha256']==sha(run/'protocol.json')
    cpu=read(run/'cpu_cluster.json');assert cpu['status']=='passed' and cpu['protocol_sha256']==sha(run/'protocol.json')
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
    from runtime_bridge import runtime
    env=runtime('memory');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    assert torch.cuda.mem_get_info()[0]>=24*2**30 and env['psutil'].virtual_memory().available>=12*2**30
    from observer_compare import CompareObserver,ARMS
    from replay import VideoSource,replay
    out.mkdir(exist_ok=False);start=time.monotonic();count=generated=encodes=resets=0;peak=0
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    assert model.config.text_config.hidden_size==2048
    save(out/'loaded.json',dict(pretrained_loads=1,seconds=time.monotonic()-start))
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),
        canvas=(192,256),resize_mode='letterbox')
    with (out/'events.jsonl').open('x',encoding='utf-8') as events,(out/'tokens.jsonl').open('x',encoding='utf-8') as tokens,torch.inference_mode(),env['fp32_reference']():
        def guard():
            nonlocal peak
            peak=max(peak,env['psutil'].Process().memory_info().rss)
            assert peak<20*2**30 and torch.cuda.memory_allocated()<32*2**30
            assert time.monotonic()-start<1500
            assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<64*2**20
        def event(item):
            nonlocal count,generated
            if item['event']['kind']=='query':
                assert set(item['answer']['arms'])==set(ARMS)
                count+=3;generated+=sum(len(a['generated_ids']) for a in item['answer']['arms'].values())
            events.write(json.dumps(item)+'\n');events.flush();guard()
        def token(item):tokens.write(json.dumps(item)+'\n');tokens.flush()
        for number,s in enumerate(sources):
            source=VideoSource(ROOT/s['media_path'],s,env['FramePacket'])
            obs=CompareObserver(env,model,builder)
            try:
                replay(source,[q for q in rows if q['source_path']==s['source_path']],obs,snapshots,event,token,read_from='three_banks')
                encodes+=obs.core.vision_encodings;resets+=obs.core.resets
            finally:obs.close()
            del obs,source;gc.collect();torch.cuda.empty_cache();guard()
            print(dict(completed_sources=number+1,answers=count),flush=True)
    assert count==102 and generated<=816 and encodes==1426
    save(out/'completed.json',dict(status='complete',answers=count,generated_tokens=generated,visual_encodings=encodes,
        pretrained_loads=1,shared_main_resets=resets,seconds=time.monotonic()-start,
        peak_RSS_observed_bytes=peak,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        protocol_sha256=sha(run/'protocol.json'),default_unchanged=True,wallclock=False,
        no_full_history_feature_archive=True,shared_process_timing_NOT_per_arm_speed_claim=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();worker(a.run,a.out)
