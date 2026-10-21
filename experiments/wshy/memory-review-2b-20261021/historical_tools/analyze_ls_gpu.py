"""Describe observed LS tradeoffs without treating provenance as detail readability."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from audit_ls_gpu import read, rows, require, sha, ARMS

job = sys.argv[1]
require(job.isdigit(), 'numeric job ID required')
run = ROOT/('runs/ls-gpu-'+job)
audit = read(ROOT/('reports/job-'+job+'-LS-GPU-result-20261021.json'))
require(audit['status'] == 'independently_verified_six_arm_development_comparison', 'independent audit required')
catalog = {r['id']: r for r in read(run/'scoring_ONLY.json')}
requests = {r['id']: r for r in read(run/'requests.json')}
pred = {(r['id'], r['arm']): r for r in rows(run/'gpu/predictions.jsonl')}
details = {(r['id'], r['arm']): r for r in read(run/'score/details.json')}
events = {(r['media_sha256'], r['pair'], r['arm']): r for r in rows(run/'gpu/events.jsonl')
          if r['event'] == 'append'}
groups = {r['media_sha256']: r['model_visible']['frames'] for r in requests.values()}
first_merge, final = {}, defaultdict(dict)
for h, frames in groups.items():
    count = (len(frames)+1)//2
    for i in range(count):
        ls = events[h, i, 'LS_shared384_v1']
        state, nodes = ls['state'], ls['state']['nodes']
        require(nodes[-min(i+1, 4):] == events[h, i, 'M0']['state']['nodes'][-min(i+1, 4):],
                'LS recent four raw pair identities differ')
        require(all(ls[key] == events[h, i, arm][key] for key in ('starts_event', 'adjacent_distance')
                    for arm in ('M2', 'M2_A06')), 'event detector changed across candidates')
        require(state['visual_tokens'] == min(i+1, 8)*48, 'LS shared capacity schedule')
        if i < 8:
            require(nodes == events[h, i, 'M0']['state']['nodes']
                    and all(n['exact_pair'] for n in nodes), 'LS compressed before capacity filled')
        else:
            require(len(nodes) == 8 and sum(n['tokens'] for n in nodes[:-4]) == 192,
                    'LS history partition exceeded')
            if h not in first_merge:
                require(any(n['depth'] > 0 for n in nodes[:-4]), 'LS did not compress on overflow')
                first_merge[h] = i+1
        for arm in ('M2', 'M2_A06', 'LS_shared384_v1'):
            ns = events[h, i, arm]['state']['nodes']
            require(sum(n['original_pairs'] for n in ns) == i+1
                    and ns[0]['first'] == 0 and ns[-1]['last'] == i
                    and all(a['last']+1 == b['first'] for a, b in zip(ns, ns[1:])),
                    'history provenance gap/overlap')
            require(sum(n['mass'] for n in ns) == (i+1)*48, 'represented mass changed')
    for arm in ARMS[:-1]:
        state = events[h, count-1, arm]['state']
        final[arm][h] = dict(max_depth=max(n['depth'] for n in state['nodes']),
                             raw_pairs=sum(n['exact_pair'] for n in state['nodes']),
                             merges=state['merge_counts'], tokens=state['visual_tokens'])
require(all(value == 9 for value in first_merge.values()), 'LS first compression timing changed')
selected = set()
pair_counts = {}
for ref in ('M0', 'M2', 'M2_A06', 'reservoir4'):
    change = audit['paired_comparisons'][ref]['LS_shared384_v1']['strict_correct']
    pair_counts[ref] = dict(corrections=len(change['corrections']), new_losses=len(change['new_losses']), net=change['net'])
    if ref in ('M2', 'M2_A06'):
        selected.update(change['corrections']+change['new_losses'])
changed = []
for qid in sorted(selected):
    k, request = catalog[qid], requests[qid]
    frames = request['model_visible']['frames']
    intervals = k['evidence_intervals'] or []
    hit_pairs = sorted({i//2 for i, f in enumerate(frames) if any(a <= f['pts_s'] <= b for a, b in intervals)})
    end = (len(frames)+1)//2-1
    changed.append(dict(id=qid, question=k['question'], stratum=k['stratum'], cohort=k['cohort'],
        source_group_id=k['source_group_id'], truth=chr(65+k['answer_index']), options=k['options'],
        author_evidence_intervals=intervals, sampled_hit_pair_indices=hit_pairs,
        evidence_age_proxy=details[qid, 'M2']['evidence_age_proxy'],
        answers={arm: dict(text=pred[qid, arm]['text'], strict_correct=details[qid, arm]['strict_correct'],
                           leading_choice_correct=details[qid, arm]['content_correct']) for arm in ARMS},
        bank_evidence_provenance={arm: [dict(first=n['first'], last=n['last'], depth=n['depth'],
                    exact_pair=n['exact_pair'], annotated_pair_ids_in_span=[i for i in hit_pairs if n['first'] <= i <= n['last']])
                for n in events[request['media_sha256'], end, arm]['state']['nodes']]
                for arm in ('M2', 'M2_A06', 'LS_shared384_v1')},
        later_repetition_or_context_cues_review='pending', semantic_answer_review='pending',
        provenance_proves_detail_readability=False))
source_pairing = {}
for ref in ('M2', 'M2_A06', 'reservoir4'):
    delta = {source: value['LS_shared384_v1']['strict_correct']-value[ref]['strict_correct']
             for source, value in audit['per_source'].items()}
    source_pairing[ref] = dict(improved=sum(v > 0 for v in delta.values()),
                              regressed=sum(v < 0 for v in delta.values()), unchanged=sum(v == 0 for v in delta.values()),
                              nonzero_net={k: v for k, v in delta.items() if v})
descriptors = {}
for arm, values in final.items():
    depths = [v['max_depth'] for v in values.values()]
    descriptors[arm] = dict(sources=len(values), median_final_max_depth=statistics.median(depths),
                             maximum_final_depth=max(depths),
                             minimum_final_raw_pairs=min(v['raw_pairs'] for v in values.values()),
                             total_within_merges=sum(v['merges']['within'] for v in values.values()),
                             total_cross_merges=sum(v['merges']['cross'] for v in values.values()))
report = dict(job_id=int(job), audit_sha256=sha(ROOT/('reports/job-'+job+'-LS-GPU-result-20261021.json')),
    observed_LS_capacity_schedule_verified=True, first_compression_pair_1based=9,
    sources_reaching_overflow=len(first_merge), sources_without_overflow=len(groups)-len(first_merge),
    same_event_detector_verified=True, represented_mass_and_provenance_verified=True,
    pair_counts=pair_counts, source_pairing=source_pairing, final_node_descriptors=descriptors,
    changed_cases=changed, format_noncompliance=[dict(id=qid, arm=arm, text=pred[qid, arm]['text'])
        for (qid, arm), value in details.items() if not value['format_compliant']],
    human_semantic_review='pending', long_memory_benefit_established=False,
    decision='retain_as_optional_candidate_for_validation; no_default_activation; no_automatic_combination_or_tuning')
output = ROOT/('reports/job-'+job+'-LS-tradeoffs-20261021.json')
with output.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps({k: report[k] for k in ('pair_counts', 'source_pairing', 'final_node_descriptors',
    'sources_reaching_overflow', 'format_noncompliance')}, ensure_ascii=False))
