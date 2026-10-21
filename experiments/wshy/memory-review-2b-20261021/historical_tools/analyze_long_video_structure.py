"""Describe structural tradeoffs in already collected outputs; no inference."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
job=sys.argv[1]
if not job.isdigit():raise ValueError('numeric job ID required')
run=ROOT/('runs/long-video-gpu-'+job)
audit=ROOT/('reports/job-'+job+'-long-video-GPU-result-20261021.json')
if json.loads(audit.read_text(encoding='utf-8'))['status']!='independently_verified_real_long_video_development_comparison':
    raise ValueError('independent real execution audit required')
predictions=[json.loads(s) for s in (run/'gpu/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
streams=defaultdict(list)
with (run/'gpu/events.jsonl').open(encoding='utf-8') as stream:
    for line in stream:
        row=json.loads(line)
        if row['event']=='append':streams[row['source_path'],row['arm']].append(row)
first_merges={}
for (source,arm),stream in streams.items():
    if arm!='LS_shared384_v1':continue
    for i,row in enumerate(stream):
        nodes=row['state']['nodes']
        if row['state']['visual_tokens']!=min(i+1,8)*48:
            raise ValueError('LS shared capacity schedule drift')
        recent=min(i+1,4)
        if nodes[-recent:]!=streams[source,'M0'][i]['state']['nodes'][-recent:]:
            raise ValueError('LS recent raw identity mismatch')
        if i<8 and (any(not n['exact_pair'] for n in nodes)
                    or nodes!=streams[source,'M0'][i]['state']['nodes']):
            raise ValueError('LS compressed before capacity filled')
        if i>=8 and source not in first_merges and any(n['depth']>0 for n in nodes):
            first_merges[source]=i+1
if len(first_merges)!=6 or set(first_merges.values())!={9}:
    raise ValueError('real LS overflow timing mismatch')
structures={}
for arm in ('M2','M2_A06','LS_shared384_v1'):
    selected=[p for p in predictions if p['arm']==arm]
    depths=[max(n['depth'] for n in p['node_metadata']) for p in selected]
    spans=[max(n['original_pairs'] for n in p['node_metadata']) for p in selected]
    finals=[s[-1]['state'] for (source,a),s in streams.items() if a==arm]
    structures[arm]=dict(queries=34,median_query_max_depth=statistics.median(depths),
        maximum_query_depth=max(depths),median_query_max_node_span_pairs=statistics.median(spans),
        total_within_merges=sum(s['merge_counts']['within'] for s in finals),
        total_cross_merges=sum(s['merge_counts']['cross'] for s in finals),
        per_query=[dict(id=p['id'],source_path=p['source_path'],query_time_s=p['query_time_s'],
            max_depth=d,max_node_span_pairs=s) for p,d,s in zip(selected,depths,spans)])
result=dict(job_id=int(job),audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
    LS_capacity_schedule_verified_at_all_actual_prefixes=True,LS_first_merge_pair_1based=first_merges,
    structures=structures,interpretation='34 correlated query snapshots across6 reused sources; descriptive only',
    depth_or_span_proves_readability=False,algorithm_change_this_analysis=False)
output=ROOT/('reports/job-'+job+'-long-video-structure-20261021.json')
with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
print(json.dumps({a:{k:v for k,v in s.items() if k!='per_query'} for a,s in structures.items()}))
