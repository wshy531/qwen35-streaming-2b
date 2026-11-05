"""First quality precheck on the same label-blind eight new sources; three arms."""
import os,sys,time,json,hashlib,gc,argparse
from pathlib import Path
from session import ROOT,Session
from sember_common import runtime
from resolution_runtime import make_builder
sys.path.insert(0,str(ROOT/'experiments/streamingbench-interface-v1'))
sys.path.insert(0,str(ROOT/'experiments/2b-perception-transition-v1'))
from source import CausalPairs
from canonical_prompt import question_prompt,rotate
PREP=ROOT/'runs/content_quality_profile_prepare_20261101_01'
MODES=['full','mid_fixed','mid_content']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def verify():
    for rel,h in read(PREP/'execution_lock.json').items():
        p=ROOT/rel
        if rel.startswith('repo/') and not p.exists():p=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(p)==h,rel
def main(out):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':raise RuntimeError('approved Slurm required')
    verify();auth=read(PREP/'GPU_authorization.json')
    assert auth['approved'] and auth['execution_lock_sha256']==sha(PREP/'execution_lock.json')
    assert not out.exists() and out.resolve().is_relative_to((ROOT/'runs').resolve())
    original=ROOT/'runs/streamingbench_router_prepare_20261101_01'
    tasks=read(original/'profile_requests.json');protocol=read(original/'protocol.json')
    media={s['name']:s for s in read(original/'media.json')};data=ROOT/'datasets/StreamingBench-eval200-v1/videos'
    for name in protocol['profile_sources']:
        p=data/name;assert p.resolve().parent==data.resolve() and p.stat().st_size==media[name]['bytes'] and sha(p)==media[name]['sha256']
    env=runtime();t=env['torch'];t.set_num_threads(1)
    assert t.cuda.device_count()==1 and t.cuda.mem_get_info()[0]>=24*2**30
    out.mkdir();start=time.monotonic();answers=generated=vision=warmups=0;profiles=[]
    try:
        model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
            dtype=t.float32,device_map='cuda',attn_implementation='sdpa').eval();model.requires_grad_(False)
        builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),(288,384))
        def guard():
            assert time.monotonic()-start<1110 and env['psutil'].Process().memory_info().rss<20*2**30
            assert t.cuda.max_memory_allocated()<32*2**30
            assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<64*2**20
        with t.inference_mode(),(out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for si,name in enumerate(protocol['profile_sources']):
                session=Session(env,model,builder,modes=MODES)
                def commit(a,b,padding):guard();session.append_pair(a,b,padding=padding)
                driver=CausalPairs(data/name,commit);began=time.monotonic()
                try:
                    for q in sorted((q for q in tasks if q['media_name']==name),key=lambda q:(q['query_time_s'],q['id'])):
                        guard();state=driver.advance(q['query_time_s']);assert state['last_committed_pts']<=q['query_time_s']
                        if not warmups:
                            for mode in MODES:
                                w=session.ask(mode,question_prompt(rotate(q,0)[0]),lambda item:None,max_new_tokens=8)
                                warmups+=1;generated+=len(w['generated_ids'])
                        for shift in range(4):
                            modes=MODES[(si+shift)%3:]+MODES[:(si+shift)%3]
                            for mode in modes:
                                guard();r=session.ask(mode,question_prompt(rotate(q,shift)[0]),lambda item:None,max_new_tokens=8)
                                assert r['visual_tokens']<=864 and r['read_state_unchanged'] and r['branch_released']
                                stream.write(json.dumps(dict(task_id=q['id'],arm=mode,shift=shift,source_state=state,**r))+'\n');stream.flush()
                                answers+=1;generated+=len(r['generated_ids'])
                    profiles.append(dict(name=name,wall_seconds=time.monotonic()-began,vision_calls=session.visual_calls,
                        timing=session.timing.copy(),status=session.status(),source=driver.status()))
                    vision+=session.visual_calls
                finally:driver.close();session.close()
                del session;gc.collect();t.cuda.empty_cache()
                print(json.dumps(dict(source=name,answers=answers,vision=vision)),flush=True)
        assert answers==480 and warmups==3 and generated<=3864 and vision<=5106
        verify();save(out/'completed.json',dict(status='complete',formal_answers=answers,warmup_answers=warmups,
            vision_calls=vision,generated_tokens=generated,profiles=profiles,pretrained_loads=1,
            wall_seconds=time.monotonic()-start,same_eight_pair_coverage=True,not_final_external_confirmation=True))
    except Exception as e:save(out/'failed.json',dict(error=str(e),answers=answers,generated_tokens=generated));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
