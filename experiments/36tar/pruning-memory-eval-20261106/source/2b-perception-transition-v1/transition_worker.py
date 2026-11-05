"""One new authorization required; reference gate before any transition answers."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from canonical_prompt import question_prompt, rotate
from transition_common import ROOT, PREP, INPUTS, ARMS, read, save, sha
from transition_scoring import analyze, parse_original

OLD = ROOT / 'runs/perception_interference_prepare_20261027_01'


def shared():
    # Reuse the frozen first-round mechanics, not its long question formatter.
    sys.path.insert(0, str(ROOT / 'experiments/2b-perception-interference-v1'))
    from interference_execution import exact_pairs, project, append, ask
    from sember_common import runtime
    from sember_runtime import make_builder
    from sember_banks import Bank
    return exact_pairs, project, append, ask, runtime, make_builder, Bank


def reference_gate(raw, expected):
    keyed = {(r['task_id'],r['arm'],r['shift']):r for r in raw}
    keys = {(r['task_id'],r['arm'],r['shift']) for r in expected}
    if len(raw) != 30 or len(keyed) != 30 or set(keyed) != keys:
        raise ValueError('exactly 30 reference responses required')
    results = []
    for r in expected:
        answer = keyed[(r['task_id'],r['arm'],r['shift'])]
        actual = parse_original(answer['text'], r['shift'])
        results.append(dict(task_id=r['task_id'], arm=r['arm'], shift=r['shift'],
                            actual=actual, expected=r['expected_content_choice_original'],
                            matched=actual==r['expected_content_choice_original']))
    return dict(passed=all(r['matched'] for r in results), matched=sum(r['matched'] for r in results),
                denominator=30, rows=results)


def checkpoints(env, model, builder, media, trace, query_times, guard):
    """Consume each source once, keeping only its bounded bank and descriptor."""
    exact_pairs, project, append, _, _, _, Bank = shared()
    times = set(query_times)
    if not times or len(times) != len(query_times):
        raise ValueError('unique explicit query times required')
    bank, descriptor = Bank('LS', 108), None
    delivered = set()
    for index, (first, second) in enumerate(exact_pairs(media, trace)):
        guard()
        value, _ = project(env, model, builder, first, second)
        descriptor = append(bank, value, index, first, second, descriptor)
        del value
        if second.timestamp in times:
            state = bank.status()
            if state['tokens'] > 864 or state['recent_tokens'] != 432:
                raise RuntimeError('bounded bank or four recent pairs violated')
            delivered.add(second.timestamp)
            yield second.timestamp, bank, index+1
    if delivered != times:
        raise ValueError('query not reached in exact complete causal prefix')


def main(out, lock_path, media_root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK') != '4':
        raise RuntimeError('explicitly approved one-GPU/four-CPU Slurm allocation required')
    if not out.resolve().is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('owned output directory required')
    from transition_freeze import verify
    verify(lock_path)
    exact_pairs, project, _, ask, runtime, make_builder, _ = shared()
    env = runtime()
    torch, psutil = env['torch'], env['psutil']
    torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info()[0] < 24*2**30:
        raise RuntimeError('one GPU and declared free-memory margin required')
    if psutil.virtual_memory().available < 12*2**30:
        raise RuntimeError('system memory margin below 12 GiB')
    sources = read(INPUTS/'sources_LOCAL_ONLY.json')
    old_media = read(OLD/'cluster_media_CPU.json')['sources']
    for row in [*sources,*old_media]:
        path = media_root/row.get('name',row.get('media_name',''))
        if path.resolve().parent != media_root.resolve() or sha(path)!=row['media_sha256']:
            raise ValueError('exact frozen media SHA mismatch')
    out.mkdir(exist_ok=False)
    started, calls, formal_tokens, warm_tokens = time.monotonic(), [0], 0, 0
    def guard():
        if time.monotonic()-started>1050 or psutil.Process().memory_info().rss>20*2**30:
            raise RuntimeError('worker time/RSS bound exceeded')
        if torch.cuda.max_memory_allocated()>48*2**30:
            raise RuntimeError('GPU allocation-memory bound exceeded')
        if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>32*2**20:
            raise RuntimeError('result bound exceeded')
    model, loading = env['Qwen3_5ForConditionalGeneration'].from_pretrained(
        ROOT/'models/Qwen3.5-2B', local_files_only=True, dtype=torch.float32,
        device_map='cuda', attn_implementation='sdpa', output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('pretrained loading mismatch')
    model.eval()
    if model.config.text_config.hidden_size!=2048:
        raise ValueError('frozen Qwen3.5-2B required')
    builder = make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    def hook(module,args):
        calls[0]+=1
    handle=model.model.visual.register_forward_pre_hook(hook)
    save(out/'startup.json',dict(loads=1,precision='float32',execution_lock_sha256=sha(lock_path),
        loading_summary={k:len(loading.get(k) or ()) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')}))
    old_tasks={r['id']:r for r in read(OLD/'inputs/tasks_LOCAL_ONLY.json')}
    old_pairs=read(OLD/'inputs/manifest_LOCAL_ONLY.json')['pairs']
    expected=read(OLD/'old_reference_predictions_LOCAL_ONLY.json')
    reference=read(PREP/'canonical_input_reference_LOCAL_ONLY.json')
    old_ids={(r['id'],r['arm'],r['shift']):r['input_ids_sha256'] for r in reference['rows']}
    reference_raw=[]
    with (out/'reference_answers.jsonl').open('x',encoding='utf-8') as stream:
        for pi,pair in enumerate(old_pairs):
            audit=next(r for r in old_media if r['pair_id']==pair['pair_id'])
            task=old_tasks[pair['pair_id']+':old']
            for _,bank,_ in checkpoints(env,model,builder,media_root/audit['media_name'],
                                        audit['sampled_trace'],[pair['committed_through_s']],guard):
                warm_ids={}
                if pi==0:
                    for arm in ('recent','full'):
                        result=ask(env,model,builder,bank,question_prompt(rotate(task,0)[0]),arm,[])
                        warm_ids[arm]=result['generated_ids']
                        warm_tokens+=len(result['generated_ids'])
                    save(out/'warmup.json',warm_ids)
                for item in [r for r in expected if r['task_id']==task['id']]:
                    guard()
                    rotated,mapping=rotate(task,item['shift'])
                    result=ask(env,model,builder,bank,question_prompt(rotated),item['arm'],[])
                    if result['input_ids_sha256']!=old_ids[(pair['old_question_id'],item['arm'],item['shift'])]:
                        raise RuntimeError('reference text input differs from original partner input')
                    if pi==0 and item['shift']==0 and result['generated_ids']!=warm_ids[item['arm']]:
                        raise RuntimeError('isolated warm/formal read changed')
                    row=dict(task_id=task['id'],shift=item['shift'],display_to_original=mapping,**result)
                    reference_raw.append(row)
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
                    formal_tokens+=len(result['generated_ids'])
            del bank
            gc.collect(); torch.cuda.empty_cache()
    gate=reference_gate(reference_raw,expected)
    save(out/'reference_gate.json',gate)
    if calls[0]!=355:
        raise RuntimeError('reference phase visual count drift')
    if not gate['passed']:
        save(out/'completed.json',dict(status='stopped_reference_not_reproduced',reference_answers=30,
            transition_answers=0,warmup_answers=2,visual_calls=calls[0],generated_tokens=formal_tokens,
            warmup_generated_tokens=warm_tokens,wall_seconds=time.monotonic()-started,trained=False))
        print(dict(status='stopped_reference_not_reproduced',matched=gate['matched']),flush=True)
        handle.remove()
        return
    tasks=read(INPUTS/'tasks_LOCAL_ONLY.json')
    byid={r['id']:r for r in tasks}
    schedule=read(INPUTS/'schedule.json')
    records,profiles=[],[]
    with (out/'transition_answers.jsonl').open('x',encoding='utf-8') as stream:
        for si,source in enumerate(sources):
            # Fixed next source, not chosen by question or observed outputs.
            donor_source=sources[(si+1)%len(sources)]
            donors=[]
            for first,second in exact_pairs(media_root/donor_source['name'],donor_source['sampled_trace'][:8]):
                guard(); donors.append(project(env,model,builder,first,second)[0])
            if len(donors)!=4:
                raise ValueError('four donor pairs required')
            selected=[t for t in tasks if t['source_name']==source['name']]
            times=sorted({t['query_time_s'] for t in selected})
            for query,bank,pair_count in checkpoints(env,model,builder,media_root/source['name'],source['sampled_trace'],times,guard):
                ids={t['id'] for t in selected if t['query_time_s']==query}
                for item in [r for r in schedule if r['task_id'] in ids]:
                    guard()
                    task=byid[item['task_id']]
                    rotated,mapping=rotate(task,item['shift'])
                    result=ask(env,model,builder,bank,question_prompt(rotated),item['arm'],donors)
                    row=dict(task_id=task['id'],shift=item['shift'],display_to_original=mapping,
                             query_time_s=query,committed_pairs=pair_count,**result)
                    records.append(row)
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
                    formal_tokens+=len(result['generated_ids'])
                profiles.append(dict(source_name=source['name'],query_time_s=query,committed_pairs=pair_count,
                    bank=bank.status(),donor_source_name=donor_source['name'],donor_pairs=4,
                    donor_bytes=sum(v.numel()*v.element_size() for v in donors),
                    no_full_feature_archive=True,no_observation_language_state=True))
            del bank,donors
            gc.collect(); torch.cuda.empty_cache()
            print(dict(completed_sources=si+1,transition_answers=len(records),visual_calls=calls[0]),flush=True)
    guard()
    if len(records)!=270 or calls[0]!=804 or formal_tokens+warm_tokens>2416:
        raise RuntimeError('work differed from fixed budget')
    save(out/'analysis.json',analyze(records,tasks,read(INPUTS/'cases_LOCAL_ONLY.json'),schedule))
    save(out/'completed.json',dict(status='complete',reference_answers=30,transition_answers=270,
        warmup_answers=2,generated_tokens=formal_tokens,warmup_generated_tokens=warm_tokens,
        visual_calls=calls[0],pretrained_loads=1,profiles=profiles,wall_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        RSS_bytes=psutil.Process().memory_info().rss,trained=False,automatic_retry=False,
        causal_file_replay_not_wallclock_1x=True,independent_review=False))
    handle.remove()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--lock',type=Path,required=True)
    parser.add_argument('--media-root',type=Path,required=True)
    args=parser.parse_args()
    main(args.out,args.lock,args.media_root)
