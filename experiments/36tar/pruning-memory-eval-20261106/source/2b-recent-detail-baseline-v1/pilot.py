"""One approved real-weight development comparison. No training or new media."""
import argparse
import importlib.util
import gc
import json
import os
import sys
import time
from guard import *
from qa import ask
sys.path.insert(0,str(ROOT/'experiments/2b-midprune-memory-v1'))
_legacy_spec=importlib.util.spec_from_file_location('legacy_midprune_pilot',ROOT/'experiments/2b-midprune-memory-v1/pilot.py')
_legacy=importlib.util.module_from_spec(_legacy_spec)
sys.modules[_legacy_spec.name]=_legacy
_legacy_spec.loader.exec_module(_legacy)
exact_pairs=_legacy.exact_pairs
from protocol import runtime,make_builder,encode_pair,CoverageScorer,sha
from response import question
from response_v4 import parse

P=ROOT/'runs/recent_detail_prepare_20261103_01'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    lock=read(P/'execution_lock.json')
    for n,h in lock.items():
        path=ROOT/n
        if not path.exists() and n.startswith('repo/'):path=ROOT/'publish/qwen35-streaming-2b'/n[5:]
        assert sha(path)==h,n
    auth=read(P/'GPU_authorization.json')
    assert auth['approved'] and auth['execution_lock_sha256']==sha(P/'execution_lock.json')
    for stem in ('CPU','CPU_QA','CPU_layout'):
        assert read(P/f'{stem}_local.json')==read(P/f'{stem}_cluster.json')
        assert read(P/f'{stem}_local.json')['passed']
    assert read(P/'CPU_media_cluster.json')['pixel_verified_pairs']==429
    assert not a.out.exists() and a.out.resolve().is_relative_to((ROOT/'runs').resolve())
    a.out.mkdir();start=time.monotonic();answers=vision=generated=warm=0
    def guard():
        if time.monotonic()-start>18*60:raise TimeoutError('fixed stop; preserve partial output')
        assert sum(f.stat().st_size for f in a.out.iterdir() if f.is_file())<32*2**20
    try:
        spec=read(P/'protocol.json');qs=read(P/'questions.json');sources=read(P/'sources.json')
        env=runtime();torch.set_num_threads(1)
        assert torch.cuda.device_count()==1 and env['psutil'].virtual_memory().available>=12*2**30
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
            dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False)
        builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
        with torch.no_grad(),env['fp32_reference'](),(a.out/'answers.jsonl').open('x') as out:
            for source in sources:
                path=Path(spec['media_root'])/source['name']
                assert path.resolve().parent==Path(spec['media_root']).resolve() and sha(path)==source['media_sha256']
                memories={arm:RecentDetail(arm) for arm in ARMS};selector=CoverageScorer();delivered=set()
                tasks=[q for q in qs if q['source_name']==source['name']]
                for i,(first,second) in enumerate(exact_pairs(env,path,source['sampled_trace'])):
                    guard();batch,_=builder.pair(first,second,0,0);batch=batch.to('cuda')
                    features,ids,_=encode_pair(model,batch,'full',None,None,selector);vision+=1
                    assert torch.equal(ids.cpu(),torch.arange(108))
                    if not warm:
                        for arm in ARMS:
                            m=RecentDetail(arm);m.append(features,0,first.timestamp,second.timestamp)
                            r=ask(env,model,builder,m,'What is visible now? Answer briefly.',limit=1)
                            generated+=len(r['generated_ids']);warm+=1;m.close()
                    for m in memories.values():m.append(features,i,first.timestamp,second.timestamp)
                    assert digest(memories['uniform810'].export()[:-1])==digest(memories['protect864'].export()[:-1])
                    del features,ids,batch
                    for q in tasks:
                        if q['query_time_s']!=second.timestamp:continue
                        assert q['id'] not in delivered;delivered.add(q['id'])
                        for arm,m in memories.items():
                            guard();tick=time.monotonic()
                            r=ask(env,model,builder,m,question(q,0,'B'),limit=128)
                            r.update(parse(r['text'],len(q['options']),'B'))
                            r.update(id=q['id'],arm=arm,role=q['role'],stage=q['stage'],case_id=q['case_id'],
                                     query_time_s=q['query_time_s'],source_name=q['source_name'],bank_status=m.status(),QA_seconds=time.monotonic()-tick)
                            answers+=1;generated+=len(r['generated_ids'])
                            out.write(json.dumps(r)+'\n');out.flush()
                        print(json.dumps(dict(answers=answers,vision=vision,seconds=time.monotonic()-start)),flush=True)
                assert delivered=={q['id'] for q in tasks}
                for m in memories.values():m.close()
                del memories,selector;gc.collect();torch.cuda.empty_cache()
        assert answers==54 and vision==429 and warm==3 and generated<=6915
        save(a.out/'completed.json',dict(status='complete',answers=answers,visual_calls=vision,warmup_answers=warm,
             generated_tokens=generated,model_loads=1,updates=0,seconds=time.monotonic()-start,
             peak_allocated_bytes=torch.cuda.max_memory_allocated(),shared_quality_not_standalone_cost=True))
    except Exception as exc:
        save(a.out/'failed.json',dict(error=repr(exc),answers=answers,visual_calls=vision,generated_tokens=generated));raise

if __name__=='__main__':main()
