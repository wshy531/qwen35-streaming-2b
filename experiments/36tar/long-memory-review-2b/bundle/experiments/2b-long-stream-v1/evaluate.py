"""Timed causal replay worker. No submission, download or implicit authorization."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def write(path,item):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(item,f,ensure_ascii=False,indent=2)


def verify(run):
    assert run.resolve().is_relative_to((ROOT/'runs').resolve())
    p=read(run/'protocol.json')
    for name,digest in p['input_sha256'].items():assert sha(run/name)==digest,name
    for rel,digest in read(run/'runtime_lock.json').items():assert sha(ROOT/rel)==digest,rel
    rows=read(run/'requests.json');sources=read(run/'sources.json');snapshots=read(run/'snapshots.json')
    assert len(rows)==len({q['id'] for q in rows})==34 and len(sources)==6 and len(snapshots)==34
    assert all(set(q)=={'id','source_path','query_time_s','question','options','cohort'} for q in rows)
    for source in sources:
        path=Path(source['media_path']);assert not path.is_absolute() and '..' not in path.parts
    assert p['arms']==['native','main_only','M0','M2','text_only']
    return p,rows,sources,snapshots


def runtime(stage,repo=None):
    import runtime_bridge
    if repo is not None:
        runtime_bridge.REPO=repo
        runtime_bridge.ACTIVE=repo/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    return runtime_bridge.runtime(stage)


def worker(run,stage,out):
    protocol,rows,sources,snapshots=verify(run)
    authorization=read(run/'GPU_authorization.json')
    assert authorization['approved'] is True and authorization['protocol_sha256']==sha(run/'protocol.json')
    for name in ('native','memory'):
        proof=read(run/f'cpu_{name}.json')
        assert proof['status']=='passed' and proof['protocol_sha256']==sha(run/'protocol.json')
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')
    env=runtime(stage);torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    assert env['psutil'].virtual_memory().available>=12*2**30 and torch.cuda.mem_get_info()[0]>=24*2**30
    from observer import Observer
    from native import NativeObserver
    from replay import VideoSource,replay
    from score_memory_eval import prompt
    out.mkdir(exist_ok=False);start=time.monotonic();predictions=generated=0;timeline=[]
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    assert model.config.text_config.hidden_size==2048 and model.config.text_config.max_position_embeddings==262144
    write(out/'loaded.json',dict(pretrained_loads=1,seconds=time.monotonic()-start,stage=stage))
    build=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    process=env['psutil'].Process();peak_RSS=0
    def guard():
        nonlocal peak_RSS
        peak_RSS=max(peak_RSS,process.memory_info().rss)
        assert peak_RSS<20*2**30 and torch.cuda.memory_allocated()<60*2**30
        assert time.monotonic()-start<(900 if stage=='native' else 2400),'bounded worker time'
    with (out/'events.jsonl').open('x',encoding='utf-8') as events,(out/'tokens.jsonl').open('x',encoding='utf-8') as tokens:
        def record(stream,item):
            guard();stream.write(json.dumps(item,ensure_ascii=False)+'\n');stream.flush()
            assert events.tell()+tokens.tell()<64*2**20,'bounded logs'
        for arm in (['native'] if stage=='native' else ['main_only','M0','M2']):
            for spec in sources:
                source=VideoSource(ROOT/spec['media_path'],spec,env['FramePacket'])
                observer=(NativeObserver(env,model,build,source) if arm=='native' else
                    Observer(env,model,build,policy=None if arm=='main_only' else arm,
                        main_tokens=2048,question_tokens=128,answer_tokens=8))
                def emit(item):record(tokens,dict(arm=arm,**item))
                def event(item):
                    nonlocal predictions,generated
                    record(events,dict(arm=arm,**item))
                    if item['event']['kind']!='query':return
                    predictions+=1;generated+=len(item['answer']['generated_ids'])
                    if arm=='M0':
                        row=next(q for q in rows if q['id']==item['event']['query_id'])
                        result=observer.ask(prompt(row),read_from='text',on_token=lambda t:
                            record(tokens,dict(arm='text_only',source_path=spec['source_path'],query_id=row['id'],**t)))
                        record(events,dict(arm='text_only',source_path=spec['source_path'],event=item['event'],
                            committed_frames_sha256=item['committed_frames_sha256'],answer=result))
                        predictions+=1;generated+=len(result['generated_ids'])
                try:
                    with torch.inference_mode(),env['fp32_reference']():
                        result=replay(source,[q for q in rows if q['source_path']==spec['source_path']],observer,
                            [s for s in snapshots if s['source_path']==spec['source_path']],event,emit,
                            read_from='native' if arm=='native' else 'main' if arm=='main_only' else 'bank')
                    timeline.append(dict(arm=arm,**result,final=observer.status()))
                finally:observer.close()
                del observer,source;gc.collect();torch.cuda.empty_cache()
                guard();print(dict(stage=stage,arm=arm,completed=predictions),flush=True)
    assert predictions==(34 if stage=='native' else 136)
    assert generated<=predictions*8
    write(out/'completed.json',dict(status='complete',stage=stage,records=predictions,generated_tokens=generated,
        pretrained_loads=1,seconds=time.monotonic()-start,timelines=timeline,
        peak_RSS_observed_bytes=peak_RSS,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        protocol_sha256=sha(run/'protocol.json'),realtime_playback=False,
        note='causal replay, incremental text callbacks; no wall-clock playback claim'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--stage',choices=['native','memory'],required=True)
    parser.add_argument('--out',required=True,type=Path)
    a=parser.parse_args();worker(a.run,a.stage,a.out)
