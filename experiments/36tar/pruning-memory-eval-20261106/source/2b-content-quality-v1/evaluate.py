"""Prepared quality runner: original R2 causal prefixes, no new sampling or labels."""
import argparse,gc,hashlib,json,os,time
from collections import defaultdict
from pathlib import Path
from session import ROOT,Session,MODES
from sember_common import runtime,DATA
from resolution_runtime import make_builder
from resolution_source import CausalSource,PairDriver
from sember_protocol import prompt
PREP=ROOT/'runs/content_quality_prepare_20261101_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def main(out):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':raise RuntimeError('approved allocation required')
    # Explicit future execution freeze/receipt; current preparation has neither.
    lock=read(PREP/'execution_lock.json');auth=read(PREP/'GPU_authorization.json')
    assert auth['approved'] and auth['execution_lock_sha256']==sha(PREP/'execution_lock.json')
    for rel,h in lock['sha256'].items():
        p=ROOT/rel
        if rel.startswith('repo/') and not p.exists():p=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(p)==h,rel
    assert not out.exists() and out.resolve().is_relative_to((ROOT/'runs').resolve())
    protocol=read(PREP/'protocol.json');assert protocol['arms']==list(MODES)
    qs=read(PREP/'requests.json');videos={v['video_id']:v for v in read(PREP/'videos.json')}
    expected=read(PREP/'expected_prefixes.json');grouped=defaultdict(list)
    for q in qs:grouped[q['video_id']].append(q)
    env=runtime();t=env['torch'];t.set_num_threads(1)
    assert t.cuda.device_count()==1 and t.cuda.mem_get_info()[0]>=24*2**30
    # Frozen original baseline is supplied only for equality checks, never routing/selection.
    refpath=ROOT/protocol['baseline_reference_path'];assert sha(refpath)==protocol['baseline_reference_SHA256']
    refs={r['question_id']:r for r in map(json.loads,refpath.read_text(encoding='utf-8').splitlines()) if r['arm']==protocol['baseline_reference_arm']}
    for vid in grouped:
        v=videos[vid];p=DATA/v['relative_path'];assert p.resolve().is_relative_to(DATA.resolve())
        assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
    out.mkdir();began=time.monotonic();answers=generated=encodings=matches=warmups=0;profiles=[]
    try:
        model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
            dtype=t.float32,device_map='cuda',attn_implementation='sdpa').eval();model.requires_grad_(False)
        assert len(model.model.visual.blocks)==24
        builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),(288,384))
        def guard():
            assert time.monotonic()-began<auth['wall_seconds']-90
            assert env['psutil'].Process().memory_info().rss<20*2**30 and t.cuda.max_memory_allocated()<32*2**30
            assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<128*2**20
        with t.inference_mode(),(out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for vid,queries in sorted(grouped.items()):
                guard();session=Session(env,model,builder);source=CausalSource(DATA/videos[vid]['relative_path'])
                def append(a,b,*,padding):guard();return session.append_pair(a,b,padding=padding)
                driver=PairDriver(source,append);start=time.monotonic()
                try:
                    for index,q in enumerate(sorted(queries,key=lambda q:(q['question_time'],q['question_id']))):
                        prefix=driver.advance(q['question_time']);target=expected[q['question_id']]
                        for key in ('pairs','last_committed_pts','real_frames','submitted_prefix_sha256','true_EOF_tail_padded'):
                            assert prefix[key]==target[key],(q['question_id'],key)
                        if not warmups:
                            for mode in MODES:
                                w=session.ask(mode,prompt(q),lambda item:None)
                                generated+=len(w['generated_ids']);warmups+=1
                        modes=list(MODES);offset=index%len(modes);modes=modes[offset:]+modes[:offset]
                        for mode in modes:
                            guard();result=session.ask(mode,prompt(q),lambda item:None)
                            if mode=='full':
                                ref=refs[q['question_id']]
                                for key in ('generated_ids','text','question_suffix_sha256','bounded_context_ids_sha256','visual_tokens'):
                                    assert result[key]==ref[key],(q['question_id'],key)
                                matches+=1
                            stream.write(json.dumps(dict(question_id=q['question_id'],video_id=vid,query_prefix=prefix,**result))+'\n');stream.flush()
                            answers+=1;generated+=len(result['generated_ids'])
                    profiles.append(dict(video_id=vid,wall_seconds=time.monotonic()-start,
                        state=session.status(),timing=session.timing.copy(),visual_calls=session.visual_calls))
                    encodings+=session.visual_calls
                finally:driver.close();session.close()
                del session;gc.collect();t.cuda.empty_cache()
                print(json.dumps(dict(sources=len(profiles),answers=answers,encodings=encodings)),flush=True)
        assert answers==2390 and matches==478 and warmups==5 and generated<=38320
        save(out/'completed.json',dict(status='complete',answers=answers,generated_tokens=generated,
            baseline_matches=matches,warmup_answers=warmups,profiles=profiles,vision_calls=encodings,pretrained_loads=1,
            shared_quality_not_independent_cost=True,old_development_not_new_test=True))
    except Exception as e:save(out/'failed.json',dict(error=str(e),answers=answers,generated_tokens=generated));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
