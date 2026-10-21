"""Independently recompute six-arm results from collected, hash-bound evidence."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT/'repos/online-memory'
ARMS = ('M0', 'M2', 'M2_A06', 'LS_shared384_v1', 'reservoir4', 'text_only')
BASELINES = ('M0', 'M2', 'reservoir4', 'text_only')
METRICS = ('strict_correct', 'content_correct', 'format_compliant')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    job = sys.argv[1]
    require(job.isdigit(), 'numeric job ID required')
    run = ROOT/('runs/ls-gpu-'+job)
    states = []
    for path in sorted((ROOT/'logs').glob('cpu-release-20261021-ls-gpu-watch-*.json')):
        outer = read(path)
        if outer['returncode'] != 0:
            continue
        for line in outer['stdout'].splitlines():
            row = json.loads(line)
            if row.get('kind') == 'command' and row['args'][0] == 'sacct':
                states.extend(entry.split('|') for entry in row['stdout'].splitlines()
                              if entry.split('|')[0] == job)
    require(states and states[-1][2:4] == ['COMPLETED', '0:0'], 'Slurm completion not established')
    protocol, complete, cpu, summary = (read(run/name) for name in (
        'protocol.json', 'gpu/completed.json', 'cpu/preflight.json', 'score/summary.json'))
    require(protocol['schema'] == 'online-memory-2b-long-short-v1'
            and protocol['arms'] == list(ARMS), 'wrong comparison protocol')
    for name, value in protocol['input_hashes'].items():
        require(sha(run/name) == value, 'input drift: '+name)
    for name, value in read(run/'source_lock.json').items():
        require(sha(REPO/name) == value, 'local/executed source mismatch: '+name)
    cpu_audit = read(ROOT/'reports/job-217843-LS-CPU-result-20261021.json')
    require(cpu_audit['status'] == 'CPU_preflight_independently_verified', 'CPU proof not verified')
    for name, value in cpu_audit['artifacts_sha256'].items():
        if not name.startswith('job_logs/'):
            require(sha(run/name) == value, 'CPU proof/input changed: '+name)
    require(complete['status'] == 'complete' and complete['pretrained_loads'] == 1, 'load/completion count')
    require(complete['protocol_sha256'] == sha(run/'protocol.json')
            and complete['preflight_sha256'] == sha(run/'cpu/preflight.json'), 'completion identity')
    require(complete['environment'] == cpu['environment'], 'CPU/GPU environment drift')
    require(not (run/'gpu_failure.json').exists() and not (run/'score_failure.json').exists(), 'failure receipt')
    for name, value in complete['output_sha256'].items():
        require(sha(run/'gpu'/name) == value, 'worker/raw output hash mismatch: '+name)
    require(set(complete['output_sha256']) == {'predictions.jsonl', 'probes.jsonl', 'events.jsonl'},
            'incomplete worker output hashes')
    for name, value in summary['raw_output_hashes'].items():
        require(sha(run/'gpu'/name) == value, 'scoring/raw output hash mismatch: '+name)
    requests = read(run/'requests.json')
    byid = {r['id']: r for r in requests}
    groups = {}
    for request in requests:
        h, frames = request['media_sha256'], request['model_visible']['frames']
        require(h not in groups or groups[h] == frames, 'inconsistent same-source samples')
        groups[h] = frames
    require(len(byid) == len(requests) == 171 and len(groups) == 67, 'request denominator')
    pred, probes = (rows(run/'gpu'/name) for name in ('predictions.jsonl', 'probes.jsonl'))
    expected = {(r['id'], r['arm']): r for r in read(run/'expected_predictions.json')}
    keys = {(r['id'], r['arm']) for r in pred}
    require(len(pred) == 1026 and keys == {(qid, arm) for qid in byid for arm in ARMS}, 'answer grid')
    historical = 0
    for p in pred:
        require(p['status'] == 'ok', 'answer failed')
        frames = groups[byid[p['id']]['media_sha256']]
        require(p['input_frames_sha256'] == hashlib.sha256(json.dumps(frames, sort_keys=True).encode()).hexdigest(),
                'prediction sample identity')
        if p['arm'] in BASELINES:
            ref = expected[p['id'], p['arm']]
            require(p['historical_reference'] and all(p['replay'].values()), 'historical replay flags')
            for field in ('generated_ids', 'input_ids_sha256', 'input_frames_sha256'):
                require(p[field] == ref[field], 'independent replay mismatch: '+field)
            if p['arm'] != 'text_only':
                require(p['node_metadata'] == ref['node_metadata'], 'historical node mismatch')
            historical += 1
        else:
            require(p['replay'] is None and p['historical_reference'] is False, 'candidate false replay claim')
        if p['arm'] != 'text_only':
            require(p['memory_unchanged'], 'QA changed bank')
    expected_probes = {(h, n, arm) for h, points in protocol['checkpoints'].items()
                       for n in points for arm in ARMS[:-1]}
    require(len(probes) == len(expected_probes) == 120
            and {(r['media_sha256'], r['pairs'], r['arm']) for r in probes} == expected_probes, 'probe grid')
    require(all(r['status'] == 'ok' and r['memory_unchanged'] and r['prefix_only']
                and r['accuracy_scored'] is False for r in probes), 'probe scope/isolation')
    generated = sum(len(r['generated_ids']) for r in pred+probes)
    require(all(0 < len(r['generated_ids']) <= 8 for r in pred+probes)
            and generated == complete['generated_tokens'] and generated <= 9168, 'generation budget')
    streams, process = defaultdict(list), []
    maximum_tokens, maximum_bytes, maximum_metadata = defaultdict(int), defaultdict(int), defaultdict(int)
    for event in rows(run/'gpu/events.jsonl'):
        if event['event'] == 'process_resources':
            process.append(event)
            continue
        require(event['event'] == 'append' and event['arm'] in ARMS[:-1], 'unknown event')
        streams[event['media_sha256'], event['arm']].append(event)
        state, arm = event['state'], event['arm']
        require(state['valid'] and not state['closed'] and not state['full_history_retained']
                and not state['language_cache_retained'], 'invalid retained state')
        require(state['visual_tokens'] <= 384 and len(state['nodes']) <= 8, 'visual memory budget')
        require(state['resident_bytes'] == state['tensor_bytes']+state['detector_bytes']
                and state['resident_bytes'] <= 8*394560+8192 and state['metadata_bytes'] <= 32768,
                'resident/metadata budget')
        maximum_tokens[arm] = max(maximum_tokens[arm], state['visual_tokens'])
        maximum_bytes[arm] = max(maximum_bytes[arm], state['resident_bytes'])
        maximum_metadata[arm] = max(maximum_metadata[arm], state['metadata_bytes'])
    require(set(streams) == {(h, arm) for h in groups for arm in ARMS[:-1]}, 'append coverage')
    for (h, arm), stream in streams.items():
        frames = groups[h]
        require(len(stream) == (len(frames)+1)//2, 'append count')
        for i, event in enumerate(stream):
            pair = frames[2*i:2*i+2]
            padded = len(pair) == 1
            if padded:
                pair = pair+pair
            require(event['pair'] == i and event['frame_ids'] == [f['frame_id'] for f in pair]
                    and event['begin'] == pair[0]['pts_s'] and event['end'] == pair[1]['pts_s']
                    and event['padded_tail'] == padded and event['state']['pairs_received'] == i+1,
                    'sample chronology/padding mismatch')
    pairs = sum((len(fs)+1)//2 for fs in groups.values())
    require(pairs == 1402 and complete['source_pairs_encoded'] == pairs == complete['visual_forward_calls']
            and len(process) == pairs, 'visual encode count')
    require(Counter((e['media_sha256'], e['pair']) for e in process)
            == Counter({(h, i): 1 for h, fs in groups.items() for i in range((len(fs)+1)//2)}),
            'process sample grid')
    require(complete['main_answers'] == 1026 and complete['probe_answers'] == 120
            and complete['reset_checks'] == 335 and complete['all_QA_banks_unchanged'], 'completion coverage')
    require(complete['peak_GPU_allocated'] <= 24*2**30 and complete['peak_RSS_bytes'] <= 20*2**30
            and complete['seconds'] <= 2550, 'process resource budget')
    require(complete['peak_combined_bank_bytes'] == max(e['combined_bank_bytes'] for e in process), 'bank peak')
    scorer_path = REPO/'experiments/36tar/memory-eval-2b/bundle/experiments/2b-short-event-v1/scoring.py'
    spec = importlib.util.spec_from_file_location('_ls_audit_frozen_scorer', scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    catalog = {r['id']: r for r in read(run/'scoring_ONLY.json')}
    details = read(run/'score/details.json')
    indexed = {(r['id'], r['arm']): r for r in details}
    require(len(details) == 1026 and set(indexed) == keys and set(catalog) == set(byid), 'scoring denominator')
    totals = {arm: Counter() for arm in ARMS}
    per_source = defaultdict(lambda: {arm: dict(questions=0, strict_correct=0) for arm in ARMS})
    for p in pred:
        k, row = catalog[p['id']], indexed[p['id'], p['arm']]
        measure = scorer.measure(p['text'], len(k['options']), k['answer_index'])
        require(all(row[key] == value for key, value in measure.items()), 'independent score mismatch')
        require(row['stratum'] == k['stratum'] and row['cohort'] == k['cohort']
                and row['source_group_id'] == k['source_group_id'], 'group label drift')
        frames, intervals = byid[p['id']]['model_visible']['frames'], k['evidence_intervals'] or []
        hits = {i//2 for i, f in enumerate(frames) if any(a <= f['pts_s'] <= b for a, b in intervals)}
        boundary = max(0, (len(frames)+1)//2-8)
        age = ('annotation_missing' if not intervals else 'no_sampled_hit' if not hits else
               'old_only' if max(hits) < boundary else 'recent_only' if min(hits) >= boundary else 'cross_window')
        require(row['evidence_age_proxy'] == age, 'evidence grouping mismatch')
        for metric in METRICS:
            totals[p['arm']][metric] += measure[metric]
        group = per_source[k['source_group_id']][p['arm']]
        group['questions'] += 1
        group['strict_correct'] += measure['strict_correct']
    require(Counter(r['stratum'] for r in catalog.values()) == Counter(F=58, T=60, C=53), 'F/T/C labels')
    require(sorted(Counter(r['cohort'] for r in catalog.values()).values()) == [47, 50, 74], 'data cohort counts')
    for dimension in ('cohort', 'stratum', 'evidence_age_proxy'):
        for value in {r[dimension] for r in details}:
            for arm in ARMS:
                selected = [r for r in details if r[dimension] == value and r['arm'] == arm]
                table = summary['tables'][dimension+'/'+value][arm]
                require(table['questions'] == len(selected)
                        and all(table[m] == sum(r[m] for r in selected) for m in METRICS), 'group scores')
    comparisons = {}
    for reference in ('M0', 'M2', 'M2_A06', 'reservoir4'):
        comparisons[reference] = {}
        for candidate in (a for a in ARMS if a not in (reference, 'text_only')):
            comparisons[reference][candidate] = {}
            for metric in METRICS[:2]:
                gains = [qid for qid in catalog if not indexed[qid, reference][metric] and indexed[qid, candidate][metric]]
                losses = [qid for qid in catalog if indexed[qid, reference][metric] and not indexed[qid, candidate][metric]]
                comparisons[reference][candidate][metric] = dict(reference=reference, corrections=gains,
                                                                 new_losses=losses, net=len(gains)-len(losses))
    require(comparisons == summary['paired_comparisons']
            and comparisons['M0'] == summary['paired_against_M0'], 'pairwise scoring mismatch')
    review = read(run/'score/review_without_predictions.json')
    require(len(review) == 171 and {r['id'] for r in review} == set(byid), 'human review denominator')
    require(all(r['independent_review_status'] == 'pending' and r['semantic_answer_judgment'] is None
                and 'answer_index' not in r and 'text' not in r for r in review), 'human review contaminated')
    require(summary['human_review_status'] == 'pending' and not summary['independent_holdout']
            and not summary['semantic_accuracy_reviewed'], 'development/review scope')
    report = dict(job_id=int(job), status='independently_verified_six_arm_development_comparison',
        CPU_job_id=217843, slurm_state=states[-1][2], exit_code=states[-1][3], slurm_elapsed=states[-1][4],
        arms=list(ARMS), questions=171, main_answers=1026, probe_answers=120, generated_tokens=generated,
        real_FP32_loads=1, original_sources=67, sampled_frames=sum(map(len, groups.values())), encoded_pairs=pairs,
        historical_answers_exactly_replayed=historical, reset_checks=335,
        maximum_visual_tokens=dict(maximum_tokens), maximum_resident_bytes=dict(maximum_bytes),
        maximum_metadata_bytes=dict(maximum_metadata), combined_bank_MiB=complete['peak_combined_bank_bytes']/2**20,
        peak_GPU_allocated_GiB=complete['peak_GPU_allocated']/2**30,
        peak_GPU_reserved_GiB=complete['peak_GPU_reserved']/2**30, peak_RSS_GiB=complete['peak_RSS_bytes']/2**30,
        worker_seconds=complete['seconds'], stage_seconds=complete['stage_seconds'],
        scores={arm: dict(v) for arm, v in totals.items()}, group_scores=summary['tables'],
        paired_comparisons=comparisons, per_source=dict(per_source), human_review='pending',
        opened_development_not_holdout=True, stable_ability_improvement_established=False,
        long_video_34_evaluated=False, per_arm_speed_claim=False,
        artifacts_sha256={p.relative_to(run).as_posix(): sha(p) for p in sorted(run.rglob('*')) if p.is_file()})
    output = ROOT/('reports/job-'+job+'-LS-GPU-result-20261021.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ('job_id', 'status', 'scores', 'historical_answers_exactly_replayed',
                                          'peak_GPU_allocated_GiB', 'peak_RSS_GiB')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
