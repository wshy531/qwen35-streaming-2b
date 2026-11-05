"""All existing 150 StreamingBench questions; metadata-only grouping, no predictions."""
from pathlib import Path
import json,hashlib,collections
R=Path(__file__).resolve().parents[2];P=R/'runs/protection_streamingbench_prepare_20261103_01'
P.mkdir(exist_ok=False)
raw=R/'runs/evidence_scale_prepare_20261102_01/panel.json'
qs=[q for q in json.loads(raw.read_text(encoding='utf-8')) if q['dataset']=='StreamingBench']
assert len(qs)==150 and len({q['id'] for q in qs})==150 and all('answer_index' not in q for q in qs)
groups=collections.defaultdict(list)
for q in qs:groups[q['source_id']].append(q)
local_types={'Action Recognition','Attribute Recognition','Object Recognition','Spatial Understanding','Text-Rich Understanding'}
inventory=dict(questions=150,video_files=len(groups),task_types=dict(collections.Counter(q['type'] for q in qs)),
    perception_type_candidates=sum(q['type'] in local_types for q in qs),
    perception_label_caveat='Task category only; not a guarantee of recent-only answerability or clean current labels',
    query_prefix_seconds=dict(min=min(q['query_time_s'] for q in qs),max=max(q['query_time_s'] for q in qs),
        sum=sum(q['query_time_s'] for q in qs),unique_source_replay=sum(max(q['query_time_s'] for q in g) for g in groups.values())),
    prefix_bins=dict(collections.Counter(q['length_bin'] for q in qs)),
    role='Previously opened StreamingBench development/regression panel, not a new holdout',
    source_parent_status='File-level grouping; original parent-source identity remains unresolved',
    proposed_controls=['recent432','joint864','isolated864'],
    proposed_answers=450,proposed_max_tokens=16,proposed_visual_frontend='full108,288x384',
    proposed_sampling='1FPS causal, retain original V0 source behaviour, stop at each source last question',
    no_candidate_result_filtering=True,no_new_media=True,
    execute_condition='Only after 478 plus paired experiment analysis supports continuation; exact pixels/budget/runtime lock still required',
    stop_condition='One unchanged-method cross-task comparison; no tuning on this panel or claiming new unseen generalization',
    input_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
for name,value in [('questions.json',qs),('inventory.json',inventory)]:
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(inventory,ensure_ascii=False,indent=2))
