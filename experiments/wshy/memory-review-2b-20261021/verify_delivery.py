"""Read-only stdlib verification of archived outputs; no model, media or network."""
from collections import Counter, defaultdict
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARMS = ('M0', 'M2', 'M2_A06', 'LS_shared384_v1', 'reservoir4', 'text_only')
METRICS = ('strict_correct', 'content_correct', 'format_compliant')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Archive:
    def __init__(self):
        self.files = json.loads((HERE / 'EVIDENCE_MANIFEST.json').read_text(encoding='utf-8'))['files']

    def data(self, name):
        entry = self.files[name]
        path = (HERE / entry['stored_path']).resolve()
        require(path.is_relative_to(HERE), 'archive path escapes package')
        data = path.read_bytes()
        require(len(data) == entry['bytes'] and sha(data) == entry['sha256'], 'stored identity: ' + name)
        data = gzip.decompress(data) if entry['gzip'] else data
        require(len(data) == entry['original_bytes'] and sha(data) == entry['original_sha256'], 'original identity: ' + name)
        return data

    def read(self, name):
        return json.loads(self.data(name))

    def rows(self, name):
        return [json.loads(line) for line in self.data(name).decode('utf-8').splitlines() if line.strip()]

    def integrity(self):
        for name in self.files:
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe logical path')
            data = self.data(name)
            data.decode('utf-8')
            if name.endswith('.py'):
                ast.parse(data, filename=name)


def measure(text, n, truth):
    # Same frozen grammar, recomputed here without importing the experiment scorer.
    a = re.fullmatch(r'\s*([A-Ea-e])\s*[.!]?\s*', text)
    b = re.match(r'^\s*([A-Ea-e])(?:\b|[.!)\s])', text)
    def index(match):
        value = ord(match[1].upper()) - 65 if match else -1
        return value if 0 <= value < n else None
    a, b = index(a), index(b)
    return dict(strict_choice=a, content_choice=b, strict_correct=a == truth,
                content_correct=b == truth, format_compliant=a is not None,
                content_parseable=b is not None)


def tables(details, ids, arms):
    return {arm: dict(questions=len(ids), **{
        metric: sum(details[uid, arm][metric] for uid in ids) for metric in METRICS
    }) for arm in arms}


def paired(details, ids, candidate, reference, metric):
    gains = [uid for uid in ids if details[uid, candidate][metric] and not details[uid, reference][metric]]
    losses = [uid for uid in ids if details[uid, reference][metric] and not details[uid, candidate][metric]]
    return dict(corrections=gains, new_losses=losses, net=len(gains) - len(losses))


def check_run(archive, run, arms, questions, pair_count, resets, probe_count, long=False):
    prefix = 'evidence/' + run + '/'
    read = lambda name: archive.read(prefix + name)
    rows = lambda name: archive.rows(prefix + name)
    done, proof = read('gpu/completed.json'), read('cpu/preflight.json')
    require(done['status'] == 'complete' and done['pretrained_loads'] == 1, 'completion/load count')
    require(done['source_pairs_encoded'] == done['visual_forward_calls'] == pair_count, 'visual encode count')
    require(done['reset_checks'] == resets and done['all_QA_banks_unchanged'], 'reset/QA isolation receipt')
    require(done['environment'] == proof['environment'] and proof['pretrained_loads'] == 0, 'CPU/GPU environment')
    protocol_key = 'GPU_protocol_sha256' if long else 'protocol_sha256'
    proof_key = 'CPU_preflight_sha256' if long else 'preflight_sha256'
    protocol_file = 'gpu_protocol.json' if long else 'protocol.json'
    require(done[protocol_key] == sha(archive.data(prefix + protocol_file)), 'protocol binding')
    require(done[proof_key] == sha(archive.data(prefix + 'cpu/preflight.json')), 'CPU proof binding')
    for name, expected in done.get('output_sha256', {}).items():
        require(sha(archive.data(prefix + 'gpu/' + name)) == expected, 'worker output binding')
    requests = {q['id']: q for q in read('requests.json')}
    catalog = {q['id']: q for q in read('scoring_ONLY.json')}
    require(len(requests) == questions and requests.keys() == catalog.keys(), 'request/label denominator')
    predictions = rows('gpu/predictions.jsonl')
    indexed = {(p['id'], p['arm']): p for p in predictions}
    require(len(predictions) == len(indexed) == questions * len(arms), 'duplicate/missing answers')
    require(set(indexed) == {(uid, arm) for uid in requests for arm in arms}, 'answer grid')
    computed = {}
    for key, p in indexed.items():
        uid, arm = key
        require(p['status'] == 'ok', 'answer failed')
        if arm != 'text_only':
            require(p['memory_unchanged'], 'answer modified memory')
            nodes = p['node_metadata']
        else:
            nodes = p.get('node_metadata', [])
            require(p['visual_tokens'] == 0 and nodes == [], 'text-only bank must be empty')
        require(1 <= len(p['generated_ids']) <= 8, 'per-answer token budget')
        require(p['visual_tokens'] <= 384 and len(nodes) <= 8, 'visual bank limit')
        truth = catalog[uid]['label_index' if long else 'answer_index']
        options = requests[uid]['options'] if long else catalog[uid]['options']
        computed[key] = measure(p['text'], len(options), truth)
    reported_details = {(q['id'], q['arm']): q for q in read('score/details.json')}
    require(reported_details.keys() == computed.keys(), 'score detail grid')
    for key, value in computed.items():
        require(all(reported_details[key][metric] == count for metric, count in value.items()), 'per-answer score mismatch')
    probes = [] if long else rows('gpu/probes.jsonl')
    require(len(probes) == probe_count, 'probe denominator')
    require(all(p['prefix_only'] and p['memory_unchanged'] and not p['accuracy_scored'] for p in probes), 'probe semantics')
    require(all(1 <= len(p['generated_ids']) <= 8 for p in probes), 'probe token budget')
    require(sum(len(p['generated_ids']) for p in predictions + probes) == done['generated_tokens'], 'generated token count')
    if not long:
        require(done['main_answers'] == len(predictions) and done['probe_answers'] == probe_count, 'answer receipt counts')
        for p in predictions:
            if p['arm'] in ('M0', 'M2', 'reservoir4', 'text_only'):
                require(p['replay'] and all(p['replay'].values()), 'historical replay')
                if run == 'ls-gpu-217845':
                    require(p['historical_reference'], 'comparison baseline reference flag')
    else:
        require(done['main_answers'] == 204 and done['extension_contract_tests'] == 22, 'long receipt counts')
        require(sha(archive.data(prefix + 'gpu/extension_contracts.txt')) == done['extension_contract_log_sha256'], 'extension contracts log')
    # Every append is recorded for each bounded bank. Resource events count shared encodes.
    append_counts = Counter()
    queries = {}
    states = {}
    readout_digests = {}
    process_count = 0
    for event in rows('gpu/events.jsonl'):
        source = event['source_path' if long else 'media_sha256']
        if event['event'] == 'append':
            arm = event['arm']; state = event['state']; key = source, arm
            require(event['pair'] == append_counts[key], 'nonconsecutive pair input')
            append_counts[key] += 1
            require(state['visual_tokens'] <= 384 and len(state['nodes']) <= 8, 'recorded bank overflow')
            require(not state['full_history_retained'] and not state['language_cache_retained'], 'unbounded retained history')
            states[key] = state
        elif event['event'] in ('resources', 'process_resources'):
            process_count += 1
        elif event['event'] == 'query':
            require(event['id'] not in queries and event['all_banks_unchanged'] and not event['query_padding'], 'query isolation')
            queries[event['id']] = event
            for arm in arms:
                p = indexed[event['id'], arm]
                if arm != 'text_only':
                    state = states[source, arm]
                    require(p['node_metadata'] == state['nodes'] and p['visual_tokens'] == state['visual_tokens']
                            and p['pair_indices'] == [n['first'] for n in state['nodes']], 'query bank identity')
                    # node_digest covers exported tensors; state_digest additionally covers
                    # controls and the event descriptor. They are different hashes.
                    identity = source, arm, event['committed_pairs']
                    require(len(p['node_digest']) == 64 and readout_digests.get(identity, p['node_digest']) == p['node_digest'],
                            'same-prefix readout tensor digest changed')
                    readout_digests[identity] = p['node_digest']
    require(process_count == pair_count, 'shared encoder/resource event count')
    require(all(sum(count for (source, a), count in append_counts.items() if a == arm) == pair_count
                for arm in arms if arm != 'text_only'), 'per-bank append count')
    summary = read('score/summary.json')
    if long:
        groups = dict(original28_raw_labels=[], historical_supported16=[], historical_ambiguous11=[],
                      historical_cutoff1=[], manual6_source_declared=[])
        names = {'supported': 'historical_supported16', 'ambiguous': 'historical_ambiguous11',
                 'not_established_at_committed_cutoff': 'historical_cutoff1'}
        snapshots = {s['query_id']: s for s in read('snapshots.json')}
        require(set(queries) == set(requests) == set(snapshots), 'query snapshot grid')
        for uid, q in requests.items():
            label = catalog[uid]
            require(label['independent_review'] == 'pending' and not label['confirmed_accuracy_eligible'], 'human review must remain pending')
            if uid.startswith('ovo:'):
                groups['original28_raw_labels'].append(uid)
                groups[names[label['declared_admission']]].append(uid)
            else:
                groups['manual6_source_declared'].append(uid)
            e = queries[uid]; snap = snapshots[uid]
            require(e['historical_committed_frames_sha256'] == snap['committed_frames_sha256'], 'historical query prefix')
            require(e['committed_through_s'] <= q['query_time_s'], 'future frame at query')
            for arm in arms:
                p = indexed[uid, arm]['prefix']
                require(all(p[key] == e[key] for key in p), 'prediction/query prefix identity')
        require([len(ids) for ids in groups.values()] == [28, 16, 11, 1, 6], 'long score denominators')
        require(summary['confirmed_accuracy'] is None and summary['human_review'] == 'pending', 'no confirmed human score')
        result = {name: tables(computed, ids, arms) for name, ids in groups.items()}
        require(result == summary['tables'], 'long score tables')
        for name, refs in summary['paired_comparisons'].items():
            for reference, candidates in refs.items():
                for candidate, metrics in candidates.items():
                    for metric, reported in metrics.items():
                        require(paired(computed, groups[name], candidate, reference, metric) == reported, 'long paired changes')
    else:
        groups = defaultdict(list)
        for uid, q in requests.items():
            label = catalog[uid]; frames = q['model_visible']['frames']; intervals = label['evidence_intervals'] or []
            hits = {i // 2 for i, f in enumerate(frames) if any(lo <= f['pts_s'] <= hi for lo, hi in intervals)}
            boundary = max(0, (len(frames) + 1) // 2 - 8)
            age = ('annotation_missing' if not intervals else 'no_sampled_hit' if not hits else
                   'old_only' if max(hits) < boundary else 'recent_only' if min(hits) >= boundary else 'cross_window')
            for dimension, value in [('cohort', label['cohort']), ('stratum', label['stratum']), ('evidence_age_proxy', age)]:
                groups[dimension + '/' + value].append(uid)
        require({name: tables(computed, ids, arms) for name, ids in groups.items()} == summary['tables'], '171 group tables')
        require(len(groups['evidence_age_proxy/old_only']) == 73, 'old-only proxy denominator')
        result = tables(computed, list(requests), arms)
        comparisons = (summary['paired_comparisons'] if run == 'ls-gpu-217845'
                       else {'M0': summary['paired_against_M0']})
        for reference, candidates in comparisons.items():
            for candidate, metrics in candidates.items():
                for metric, reported in metrics.items():
                    expected = dict(reference=reference, **paired(computed, list(catalog), candidate, reference, metric))
                    require(expected == reported, '171 paired changes')
    return dict(scores=result, main_answers=len(predictions), probe_answers=len(probes),
                visual_forwards=pair_count, generated_tokens=done['generated_tokens']), indexed


def main():
    delivery = json.loads((HERE / 'DELIVERY_MANIFEST.json').read_text(encoding='utf-8'))
    for relative, expected in delivery['files'].items():
        path = (REPO / relative).resolve()
        require(path.is_relative_to(REPO), 'delivery path escapes repository')
        data = path.read_bytes()
        require(len(data) == expected['bytes'] and sha(data) == expected['sha256'], 'delivery identity: ' + relative)
    archive = Archive(); archive.integrity()
    bindings = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text(encoding='utf-8'))
    for run, binding in bindings.items():
        lock = archive.read('evidence/' + run + '/source_lock.json')
        mismatches = [name for name, expected in lock.items() if sha((REPO / name).read_bytes()) != expected]
        require(len(lock) == binding['source_lock_entries'] and mismatches == binding['current_source_mismatches'], 'executed source identity: ' + run)
    spec = archive.read('evidence/long-video-gpu-217852/gpu_protocol.json')
    for name, expected in spec['extension_sha256'].items():
        require(sha((REPO / name).read_bytes()) == expected, 'executed GPU extension changed')
    results = {}
    results['217796'], old = check_run(archive, 'gpu-replay-217796', ('M0', 'M2', 'reservoir4', 'text_only'), 171, 1402, 201, 72)
    results['217845'], comparison = check_run(archive, 'ls-gpu-217845', ARMS, 171, 1402, 335, 120)
    # Independently compare the two local runs, in addition to their recorded frozen-baseline replay.
    for key, row in old.items():
        require(all(row.get(field) == comparison[key].get(field) for field in (
            'generated_ids', 'input_ids_sha256', 'input_frames_sha256', 'node_metadata')), 'local baseline replay drift')
    results['217852'], _ = check_run(archive, 'long-video-gpu-217852', ARMS, 34, 1360, 30, 0, long=True)
    receipts = json.loads((HERE / 'SLURM_RECEIPTS.json').read_text(encoding='utf-8'))
    require(set(receipts) == set(results), 'scheduler receipt jobs')
    require(all(v['scheduler_row'].split('|')[2:4] == ['COMPLETED', '0:0'] for v in receipts.values()), 'historical scheduler completion')
    print(json.dumps(dict(status='passed', delivery_files=len(delivery['files']), evidence_files=len(archive.files), runs=results,
                         main_answers_rescored=1914, mechanism_probes_checked=192,
                         source_bindings=bindings, pretrained_loads=0, cluster_connections=0,
                         GPU_submissions=0, human_review='pending',
                         note='Archived outputs checked; not a new experiment or media/content review'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
