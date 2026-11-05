"""Export deterministic M0/M1 schedules for the existing 100 traces / 235 Qs.

No answer/author interval/witness metadata is passed to the policy scheduler.
Holdout gets technical schedules only, without semantic/quality analysis.
"""
from collections import defaultdict
import json
from pathlib import Path

from audit_sampling import query_snapshot
from memory_query_blueprint import MemoryQueryBlueprint
from score_memory_eval import digest


def main():
    root=Path(__file__).resolve().parents[2];run=root/'runs/dataset_build_20261018_01'
    release=run/'content_release_v1'
    traces=json.loads((release/'actual_sampling_traces.json').read_text(encoding='utf-8'))['videos']
    requests=[json.loads(line) for line in (run/'draft_input_export/model_requests.jsonl').read_text(encoding='utf-8').splitlines()]
    media={m['media_path']:m for m in json.loads((release/'media_inventory.json').read_text(encoding='utf-8'))['files']}
    per_source=defaultdict(list)
    for order,request in enumerate(requests):
        per_source[media[request['media_path']]['source_path']].append(dict(
            id=request['id'],order=order,query_time_s=request['model_visible']['query_time_s']))
    output=[];odd=0
    for trace in traces:
        policy=MemoryQueryBlueprint(trace['frames']);queries=[]
        for query in sorted(per_source[trace['source_path']],key=lambda q:(q['query_time_s'],q['order'])):
            snapshot=policy.query(query['query_time_s'])
            oracle=query_snapshot(trace['frames'],query['query_time_s'],None)
            for name in ('M0','M1'):
                if [p['index'] for p in snapshot[name]]!=oracle[name]['pair_indices']:
                    raise AssertionError('stream schedule differs from fresh causal prefix recomputation')
            odd+=snapshot['pending_frame_id'] is not None
            queries.append(dict(id=query['id'],**snapshot))
        output.append(dict(source_path=trace['source_path'],source_sha256=trace['source_sha256'],
            sorted_query_ids=[q['id'] for q in queries],queries=queries,
            final_persistent_state=policy.state()))
    report=dict(status='actual_frame_CPU_memory_schedule_not_true_2b_or_M2_evaluation',
        sources=len(output),queries=sum(len(s['queries']) for s in output),
        odd_snapshot_forks=odd,original_actual_trace_sha256=digest(traces),
        source_schedules=output,main_state_pollution_checks_passed=True,
        equal_to_fresh_prefix_recomputation=True,future_frame_count=0,
        input_fields_used=['source_media_identity','frame_id','pts_s','rgb_sha256','query_time_s'],
        model_integration=False,actual_visual_feature_tensors_created=0,
        limits=['Pair identities are not visual Token tensors or model state.',
            'Node spans record provenance, not retained full visual evidence.',
            'No prediction accuracy, latency or GPU memory claim.',
            'M2 is deliberately absent until real features and a verified adapter exist.'])
    (run/'actual_memory_query_blueprints.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_schedules','limits')}))


if __name__=='__main__':main()
