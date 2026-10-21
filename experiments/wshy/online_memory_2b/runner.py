"""Prepare, CPU-preflight, run once on Slurm, and score outside model workers."""
import argparse
from collections import Counter
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .isolation import configure, output_path

configure()

from .frozen import ACTIVE, ARMS, NATURAL, PACKAGE, REPO, module, runtime
from .protocol import (prepare, read, rows, safe_path, save, sha, validate_requests,
                       verify_model, verify_run, LONG_SHORT_ARMS, LONG_SHORT_PROTOCOL,
                       COMPARISON_SUITES)
from .receipts import INHERITED_COUNTS, inherited_result, successful_counts, verify_inherited


def environment(env):
    import transformers
    return dict(python=sys.version, torch=env['torch'].__version__,
                transformers=transformers.__version__, numpy=env['np'].__version__,
                 av=env['av'].__version__, accelerate=version('accelerate'),
                 psutil=env['psutil'].__version__)


def isolated_readout(env, model, builder, nodes, question):
    torch = env['torch']
    rope = model.model.rope_deltas
    rope_copy = rope.clone() if isinstance(rope, torch.Tensor) else None
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    try:
        return env['answer'](env, model, builder, nodes, question)
    finally:
        if model.model.rope_deltas is not rope or (rope_copy is not None and not torch.equal(rope, rope_copy)):
            raise RuntimeError('question changed persistent model position state')
        cpu_changed = not torch.equal(cpu_rng, torch.get_rng_state())
        cuda_changed = False
        if cuda_rng:
            current = torch.cuda.get_rng_state_all()
            cuda_changed = len(current) != len(cuda_rng) or any(
                not torch.equal(a, b) for a, b in zip(cuda_rng, current))
        if cpu_changed or cuda_changed:
            raise RuntimeError('question changed the model RNG state')


def guarded_banks(sessions, readout):
    """A question in one arm must not mutate any of the other live controls."""
    before = {arm:s.state_digest() for arm,s in sessions.items()}
    try:
        return readout()
    finally:
        if any(s.state_digest()!=before[arm] for arm,s in sessions.items()):
            for session in sessions.values():
                session.valid = False
            raise RuntimeError('question changed one or more comparison banks')


def historical_replay(expected, qid, arm, result, frame_hash):
    if arm not in ARMS:
        if arm not in LONG_SHORT_ARMS:
            raise ValueError('unknown candidate arm')
        return None  # A new candidate has no historical prediction to match.
    reference = expected[qid,arm]  # Missing baseline references are always fatal.
    return dict(generated_ids=result['generated_ids']==reference['generated_ids'],
        input_ids_sha256=result['input_ids_sha256']==reference['input_ids_sha256'],
        input_frames_sha256=frame_hash==reference['input_frames_sha256'],
        node_metadata=(arm=='text_only' or result['node_metadata']==reference['node_metadata']))


def paired_changes(details, ids, arms, reference):
    indexed={(r['id'],r['arm']):r for r in details}
    result={}
    for candidate in (a for a in arms if a not in (reference,'text_only')):
        result[candidate]={}
        for metric in ('strict_correct','content_correct'):
            gains=[qid for qid in ids if not indexed[qid,reference][metric] and indexed[qid,candidate][metric]]
            losses=[qid for qid in ids if indexed[qid,reference][metric] and not indexed[qid,candidate][metric]]
            result[candidate][metric]=dict(reference=reference,corrections=gains,new_losses=losses,
                                           net=len(gains)-len(losses))
    return result


def cpu(run):
    protocol, requests = verify_run(run)
    arms = protocol['arms']
    verify_model(Path(protocol['model']))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('preflight requires CUDA_VISIBLE_DEVICES empty')
    out = output_path(run / 'cpu')
    out.mkdir(exist_ok=False)
    inherited_log = output_path(out / 'inherited.txt')
    with inherited_log.open('x', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-B', str(REPO/'scripts/run_cpu_checks.py'),
                                 '--model-dir', protocol['model']],
                                cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('inherited CPU regressions failed; see ' + str(inherited_log))
    inherited = inherited_result(inherited_log.read_text(encoding='utf-8'))
    save(out/'inherited.json', dict(status='passed', **inherited,
         source_lock_sha256=sha(run/'source_lock.json'), log_sha256=sha(inherited_log)))
    log = out / 'contracts.txt'
    with log.open('x', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s',
                                 str(Path(__file__).parent/'tests'), '-v'],
                                cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('CPU contracts failed; see ' + str(log))
    successful_counts(log.read_text(encoding='utf-8'), [28])
    comparison_logs = {}
    if protocol['schema'] == LONG_SHORT_PROTOCOL['schema']:
        for folder, count in COMPARISON_SUITES.items():
            target = out/(folder+'.txt')
            with target.open('x',encoding='utf-8') as stream:
                result = subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',
                    str(Path(__file__).parent/folder),'-v'],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('comparison contracts failed: '+folder)
            successful_counts(target.read_text(encoding='utf-8'),[count])
            comparison_logs[folder] = dict(tests=count,sha256=sha(target))
    env = runtime()
    torch = env['torch']
    torch.set_num_threads(1)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        raise ValueError('GPU must remain hidden during preflight')
    processor = env['AutoProcessor'].from_pretrained(protocol['model'], local_files_only=True)
    builder = env['PairInputBuilder'](processor, canvas=(192, 256), resize_mode='letterbox')
    model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](
        **env['tiny_config'](processor), attn_implementation='sdpa')).eval()
    from .session import MemorySession
    from .stream import decode_pairs
    groups = validate_requests(requests)
    media = read(run/'media.json')
    checked, pairs, maximum_suffix = 0, 0, 0
    for h, source in groups.items():
        frames = source[0]['model_visible']['frames']
        file = safe_path(protocol['dataset'], media[h]['relative_path'])
        if sha(file) != h:
            raise ValueError('media changed after preparation')
        for pending, padded, _ in decode_pairs(env, file, frames, source[0]['model_visible']['query_time_s']):
            checked += 1 if padded else 2
            pairs += 1
        for request in source:
            count = builder.question_ids(env['prompt'](request['model_visible'])).shape[1]
            if count > 128:
                raise ValueError('complete question exceeds historical suffix bound')
            maximum_suffix = max(maximum_suffix, count)
    # Tiny answers test real model/readout integration without pretrained loads.
    tiny = []
    with torch.inference_mode(), env['fp32_reference']():
        for arm in arms:
            if arm == 'text_only':
                answer = isolated_readout(env, model, builder, [], 'Return only A.')
            else:
                session = MemorySession(arm, lambda ns, q: isolated_readout(env, model, builder, ns, q))
                for i in range(12):
                    block = session.legacy.features.raw_block(torch.randn(48, 32), i, 2.*i, 2.*i+1)
                    session.append_pair(block, frame_ids=(2*i, 2*i+1))
                answer = session.ask('Return only A.')
                session.close()
            tiny.append(dict(arm=arm, generated=len(answer['generated_ids'])))
    save(out/'preflight.json', dict(status='passed', protocol_sha256=sha(run/'protocol.json'),
         inherited_receipt_sha256=sha(out/'inherited.json'),
         inherited_log_sha256=sha(inherited_log), inherited_tests=inherited['tests_run'],
         inherited_tests_discovered=inherited['tests_discovered'],
         inherited_tests_skipped=inherited['tests_skipped'], inherited_platform_skips=inherited['platform_skips'],
         contract_tests=28,
         contracts_log_sha256=sha(log), source_lock_sha256=sha(run/'source_lock.json'),
         model_manifest_sha256=protocol['model_manifest_sha256'], environment=environment(env),
         source_frames_checked=checked, source_pairs_checked=pairs,
         suffix_questions_checked=len(requests), max_suffix_tokens=maximum_suffix,
         tiny_answers=tiny, pretrained_loads=0, media_files_checked=len(groups)))
    if comparison_logs:
        save(out/'comparison_contracts.json',dict(status='passed',suites=comparison_logs,
             source_lock_sha256=sha(run/'source_lock.json'),protocol_sha256=sha(run/'protocol.json')))
    print(json.dumps(dict(stage='cpu', status='passed', frames=checked, pairs=pairs)), flush=True)


def gpu(run, preflight):
    protocol, requests = verify_run(run)
    arms = protocol['arms']
    if output_path(preflight) != output_path(run/'cpu'):
        raise ValueError('CPU proof must belong to this run')
    proof_path = output_path(Path(preflight) / 'preflight.json')
    proof = read(proof_path)
    if (proof['status'] != 'passed' or proof['protocol_sha256'] != sha(run/'protocol.json')
            or proof['source_lock_sha256'] != sha(run/'source_lock.json')
            or proof['contracts_log_sha256'] != sha(Path(preflight)/'contracts.txt')
            or proof['pretrained_loads'] != 0):
        raise ValueError('fresh matching CPU preflight required')
    inherited = verify_inherited(proof_path.parent, sha(run/'source_lock.json'),
                                 proof['inherited_receipt_sha256'], proof['inherited_log_sha256'])
    if (proof['inherited_tests_discovered'] != sum(INHERITED_COUNTS)
            or proof['inherited_tests'] != inherited['tests_run']
            or proof['inherited_tests_skipped'] != inherited['tests_skipped']
            or proof['inherited_platform_skips'] != inherited['platform_skips']):
        raise ValueError('all 209 inherited cases and exact platform applicability must be accounted for')
    if proof['contract_tests'] != 28:
        raise ValueError('all 17 original and 11 safety contracts are required')
    successful_counts((proof_path.parent/'contracts.txt').read_text(encoding='utf-8'), [28])
    if protocol['schema'] == LONG_SHORT_PROTOCOL['schema']:
        extra = read(proof_path.parent/'comparison_contracts.json')
        if (extra['status']!='passed' or extra['source_lock_sha256']!=sha(run/'source_lock.json')
                or extra['protocol_sha256']!=sha(run/'protocol.json') or set(extra['suites'])!=set(COMPARISON_SUITES)
                or [r['arm'] for r in proof['tiny_answers']]!=arms):
            raise ValueError('fresh six-arm comparison CPU proof required')
        for folder,count in COMPARISON_SUITES.items():
            path=proof_path.parent/(folder+'.txt')
            if extra['suites'][folder]!=dict(tests=count,sha256=sha(path)):
                raise ValueError('comparison contract identity drift')
            successful_counts(path.read_text(encoding='utf-8'),[count])
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '-1'):
        raise ValueError('allocated Slurm GPU required')
    if int(os.environ.get('SLURM_RESTART_COUNT','0')) != 0:
        raise ValueError('automatic requeue/restart is not permitted')
    if int(os.environ.get('SLURM_CPUS_PER_TASK','0')) != protocol['cpus']:
        raise ValueError('request exactly the declared CPU allocation')
    verify_model(Path(protocol['model']))
    out = output_path(run/'gpu')
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    env = runtime()
    torch = env['torch']
    if torch.cuda.device_count() != 1:
        raise ValueError('exactly one allocated visible GPU required')
    torch.set_num_threads(1)
    torch.manual_seed(123)
    if environment(env) != proof['environment']:
        raise ValueError('CPU/GPU software environments differ')
    free, _ = torch.cuda.mem_get_info()
    save(out/'guard.json', dict(free_GPU_bytes=free,
         free_RAM_bytes=env['psutil'].virtual_memory().available,
         job=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(),
         protocol_sha256=sha(run/'protocol.json'), preflight_sha256=sha(proof_path)))
    if free < protocol['min_free_GPU_GiB']*2**30 or env['psutil'].virtual_memory().available < 12*2**30:
        raise MemoryError('preload resource gate')
    # Exclusive directory + started receipt prevents silent reloading/retries.
    save(out/'load_started.json', dict(maximum_pretrained_loads=1))
    load_start = time.monotonic()
    model = env['Qwen3_5ForConditionalGeneration'].from_pretrained(
        protocol['model'], local_files_only=True, dtype=torch.float32,
        device_map='cuda', attn_implementation='sdpa').eval()
    if next(model.parameters()).dtype != torch.float32 or model.config.text_config.hidden_size != 2048:
        raise ValueError('2B FP32 model identity required')
    save(out/'loaded.json', dict(pretrained_loads=1, seconds=time.monotonic()-load_start))
    builder = env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(
        protocol['model'], local_files_only=True), canvas=(192,256), resize_mode='letterbox')
    from .session import MemorySession
    from .stream import decode_pairs
    groups = validate_requests(requests)
    media = read(run/'media.json')
    expected = {(r['id'],r['arm']):r for r in read(run/'expected_predictions.json')}
    encoded = main_answers = probe_answers = generated = vision_calls = 0
    peak_rss = peak_bank_bytes = 0
    timings = dict(decode=0., encode=0., update=0., answer=0.)

    def check_budget():
        nonlocal peak_rss
        import resource
        rss = env['psutil'].Process().memory_info().rss
        # GPU execution is Linux-only; ru_maxrss is KiB and covers QA peaks
        # which may already have been released at this sampling point.
        peak_rss = max(peak_rss, rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if (time.monotonic()-started > protocol['gpu_seconds']
                or torch.cuda.max_memory_allocated() > protocol['max_allocated_GPU_GiB']*2**30
                or peak_rss > protocol['max_RSS_GiB']*2**30
                or main_answers+probe_answers > protocol['maximum_answers']
                or generated > protocol['maximum_generated_tokens']):
            raise RuntimeError('fixed runtime/resource/answer budget exceeded')

    def hook(*unused):
        nonlocal vision_calls
        vision_calls += 1

    vision_hook = model.model.visual.register_forward_hook(hook)
    try:
        with (out/'predictions.jsonl').open('x',encoding='utf-8') as predictions, \
             (out/'probes.jsonl').open('x',encoding='utf-8') as probes, \
             (out/'events.jsonl').open('x',encoding='utf-8') as events, \
             torch.inference_mode(), env['fp32_reference']():

            def write(stream, value):
                stream.write(json.dumps(value, ensure_ascii=False)+'\n')
                stream.flush()

            def answer(session, question):
                torch.cuda.synchronize()
                tick = time.monotonic()
                result = guarded_banks(sessions, lambda: (session.ask(question) if session else
                          isolated_readout(env, model, builder, [], question)))
                if len(result['generated_ids']) > protocol['max_new_tokens']:
                    raise RuntimeError('per-answer token budget exceeded')
                if vision_calls != encoded:
                    raise RuntimeError('question unexpectedly re-encoded visual input')
                torch.cuda.synchronize()
                timings['answer'] += time.monotonic()-tick
                return result

            for h, source in groups.items():
                check_budget()
                sessions = {arm: MemorySession(arm, lambda ns,q: isolated_readout(env,model,builder,ns,q))
                            for arm in arms if arm != 'text_only'}
                specs = source[0]['model_visible']['frames']
                query = source[0]['model_visible']['query_time_s']
                path = safe_path(protocol['dataset'], media[h]['relative_path'])
                if sha(path) != h:
                    raise ValueError('media changed after preflight')
                final_pairs = (len(specs)+1)//2
                for i, (pending, padded, decode_s) in enumerate(decode_pairs(env,path,specs,query)):
                    check_budget()
                    timings['decode'] += decode_s
                    a,b = pending[0][0],pending[1][0]
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    inputs,_ = builder.pair(a,b,0,0,synthetic_padding=padded)
                    features = torch.cat(model.model.get_video_features(
                        inputs.pixel_values_videos.to('cuda'), inputs.video_grid_thw.to('cuda'),
                        return_dict=True).pooler_output,0)
                    block = next(iter(sessions.values())).legacy.features.raw_block(features,i,a.timestamp,b.timestamp)
                    torch.cuda.synchronize()
                    timings['encode'] += time.monotonic()-tick
                    encoded += 1
                    if vision_calls != encoded:
                        raise RuntimeError('vision hook detects repeated/missing encoding')
                    del inputs, features
                    tick = time.monotonic()
                    for arm, session in sessions.items():
                        record = session.append_pair(block,frame_ids=(a.frame_id,b.frame_id),padded_tail=padded)
                        write(events,dict(media_sha256=h,arm=arm,event='append',**record))
                    timings['update'] += time.monotonic()-tick
                    del block
                    combined = sum(s.status()['resident_bytes'] for s in sessions.values())
                    peak_bank_bytes = max(peak_bank_bytes,combined)
                    if combined > len(sessions)*(8*394560+2048*4):
                        raise RuntimeError('combined bounded bank allocation exceeded')
                    write(events,dict(media_sha256=h,event='process_resources',pair=i,
                          combined_bank_bytes=combined,RSS_bytes=env['psutil'].Process().memory_info().rss,
                          GPU_allocated=torch.cuda.memory_allocated(),GPU_reserved=torch.cuda.memory_reserved()))
                    if i+1 in protocol['checkpoints'].get(h,[]):
                        for arm,session in sessions.items():
                            check_budget()
                            result = answer(session,env['prompt'](source[0]['model_visible']))
                            probe_answers += 1
                            generated += len(result['generated_ids'])
                            write(probes,dict(media_sha256=h,pairs=i+1,arm=arm,status='ok',
                                  prefix_only=True,accuracy_scored=False,**result))
                    del a,b,pending
                if any(s.append_count != final_pairs for s in sessions.values()):
                    raise RuntimeError('incomplete source prefix')
                frame_hash = hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest()
                for request in source:
                    for arm in arms:
                        check_budget()
                        result = answer(sessions.get(arm),env['prompt'](request['model_visible']))
                        main_answers += 1
                        generated += len(result['generated_ids'])
                        replay = historical_replay(expected,request['id'],arm,result,frame_hash)
                        write(predictions,dict(id=request['id'],cohort=request['cohort'],arm=arm,status='ok',
                              input_frames_sha256=frame_hash,replay=replay,
                              historical_reference=arm in ARMS,**result))
                        if replay is not None and not all(replay.values()):
                            raise RuntimeError('historical replay mismatch: '+request['id']+'/'+arm)
                for arm,session in sessions.items():
                    session.reset()
                    fresh = MemorySession(arm)
                    if session.state_digest()!=fresh.state_digest() or session.status()['resident_bytes']:
                        raise RuntimeError('reset retained prior source')
                    session.close()
                    fresh.close()
                del sessions
                gc.collect()
                print(json.dumps(dict(media=h,main_answers=main_answers,probe_answers=probe_answers,
                                      encoded_pairs=encoded)),flush=True)
            check_budget()
            if main_answers!=protocol['main_answers'] or probe_answers!=protocol['probe_answers']:
                raise RuntimeError('fixed denominator not completed')
            expected_pairs=sum((len(rs[0]['model_visible']['frames'])+1)//2 for rs in groups.values())
            if encoded!=expected_pairs or encoded!=vision_calls:
                raise RuntimeError('encoded pair count mismatch')
            save(out/'completed.json',dict(status='complete',pretrained_loads=1,main_answers=main_answers,
                 probe_answers=probe_answers,generated_tokens=generated,source_pairs_encoded=encoded,
                 visual_forward_calls=vision_calls,peak_combined_bank_bytes=peak_bank_bytes,
                 peak_RSS_bytes=peak_rss,peak_GPU_allocated=torch.cuda.max_memory_allocated(),
                 peak_GPU_reserved=torch.cuda.max_memory_reserved(),stage_seconds=timings,
                 seconds=time.monotonic()-started,environment=environment(env),
                 protocol_sha256=sha(run/'protocol.json'),preflight_sha256=sha(proof_path),
                 reset_checks=(len(arms)-1)*len(groups),all_QA_banks_unchanged=True,
                 output_sha256={name:sha(out/name) for name in ('predictions.jsonl','probes.jsonl','events.jsonl')},
                 scope='external bounded visual memory; fresh language cache for each answer'))
    finally:
        vision_hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def score(run):
    protocol, requests = verify_run(run)
    arms = protocol['arms']
    completed = read(run/'gpu/completed.json')
    if completed['status']!='complete' or completed['protocol_sha256']!=sha(run/'protocol.json'):
        raise ValueError('complete matching GPU output required')
    if protocol['schema']==LONG_SHORT_PROTOCOL['schema']:
        outputs={name:sha(run/'gpu'/name) for name in ('predictions.jsonl','probes.jsonl','events.jsonl')}
        if completed.get('output_sha256')!=outputs:
            raise ValueError('comparison outputs changed after worker completion')
    predictions = rows(run/'gpu/predictions.jsonl')
    expected = {(r['id'],arm) for r in requests for arm in arms}
    if len(predictions)!=len(expected) or {(r['id'],r['arm']) for r in predictions}!=expected:
        raise ValueError('complete unique prediction grid required')
    catalog = {r['id']:r for r in read(run/'scoring_ONLY.json')}
    scoring = module('scoring', PACKAGE/'bundle/experiments/2b-short-event-v1')
    details, tables, review = [], {}, []
    byid = {r['id']:r for r in requests}
    for p in predictions:
        k = catalog[p['id']]
        if (p['status']!='ok' or (p['arm'] in ARMS and not all(p['replay'].values()))
                or (p['arm'] not in ARMS and p['replay'] is not None)):
            raise ValueError('invalid answer or replay mismatch')
        frames = byid[p['id']]['model_visible']['frames']
        intervals = k['evidence_intervals'] or []
        hits = {i//2 for i,f in enumerate(frames) if any(a<=f['pts_s']<=b for a,b in intervals)}
        boundary = max(0,(len(frames)+1)//2-8)
        age = ('annotation_missing' if not intervals else 'no_sampled_hit' if not hits else
               'old_only' if max(hits)<boundary else 'recent_only' if min(hits)>=boundary else 'cross_window')
        details.append(dict(id=p['id'],cohort=k['cohort'],stratum=k['stratum'],
                            source_group_id=k['source_group_id'],evidence_age_proxy=age,arm=p['arm'],
                            **scoring.measure(p['text'],len(k['options']),k['answer_index'])))
    for dimension in ('cohort','stratum','evidence_age_proxy'):
        for value in sorted({r[dimension] for r in details}):
            table={}
            for arm in arms:
                selected=[r for r in details if r[dimension]==value and r['arm']==arm]
                table[arm]=dict(questions=len(selected),**{metric:sum(r[metric] for r in selected)
                    for metric in ('strict_correct','content_correct','format_compliant')})
            tables[dimension+'/'+value]=table
    for qid,k in sorted(catalog.items()):
        review.append(dict(id=qid,question=k['question'],options=k['options'],
            relative_media_path=k['relative_media_path'],author_evidence_intervals=k['evidence_intervals'],
            sampled_frames=byid[qid]['model_visible']['frames'],fixed_sample_visibility=None,
            later_repetition_or_context_cues=None,temporal_or_count_subtype=None,
            causal_chain_support=None,reviewer=None,independent_review_status='pending',
            semantic_answer_judgment=None))
    comparisons={ref:paired_changes(details,catalog,arms,ref)
                 for ref in ('M0','M2','M2_A06','reservoir4') if ref in arms}
    probes=rows(run/'gpu/probes.jsonl')
    expected_probes={(h,n,arm) for h,points in protocol['checkpoints'].items()
                     for n in points for arm in arms if arm!='text_only'}
    if (len(probes)!=len(expected_probes)
            or {(r['media_sha256'],r['pairs'],r['arm']) for r in probes}!=expected_probes
            or any(not r['memory_unchanged'] or not r['prefix_only'] or r['accuracy_scored'] for r in probes)):
        raise ValueError('incomplete or invalid online probe grid')
    out=output_path(run/'score')
    out.mkdir(exist_ok=False)
    save(out/'summary.json',dict(tables=tables,paired_against_M0=comparisons['M0'],
         paired_comparisons=comparisons,questions=171,
         main_answers=protocol['main_answers'],
         all_historical_sequences_replayed=True,independent_holdout=False,
         semantic_accuracy_reviewed=False,human_review_status='pending',
         evidence_age_is_proxy_not_memory_necessity=True,completed=completed,
         raw_output_hashes={name:sha(run/'gpu'/name) for name in (
             'predictions.jsonl','probes.jsonl','events.jsonl','completed.json')}))
    save(out/'details.json',details)
    save(out/'review_without_predictions.json',review)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','preflight','gpu','score'],required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path)
    parser.add_argument('--dataset-root',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--preflight',type=Path)
    parser.add_argument('--comparison',action='store_true',help='prepare the explicit six-arm long-short protocol')
    args=parser.parse_args()
    run=output_path(args.out)
    if args.comparison and args.stage!='prepare':
        parser.error('--comparison is only selected at prepare; later stages use the locked protocol')
    try:
        if args.stage=='prepare':
            if not args.model_dir or not args.dataset_root:
                parser.error('prepare requires --model-dir and --dataset-root')
            prepare(run,args.model_dir,args.dataset_root,args.manifest,comparison=args.comparison)
        elif args.stage=='preflight':
            cpu(run)
        elif args.stage=='gpu':
            if not args.preflight:
                parser.error('gpu requires --preflight')
            gpu(run,args.preflight)
        else:
            score(run)
    except Exception:
        if run.is_dir():
            failure=run/(args.stage+'_failure.json')
            if not failure.exists():
                save(failure,dict(stage=args.stage,error=traceback.format_exc(),
                                  successful_completion=False))
        raise


if __name__=='__main__':
    main()
