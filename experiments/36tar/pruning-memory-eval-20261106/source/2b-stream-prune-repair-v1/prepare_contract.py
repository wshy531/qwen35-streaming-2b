"""Freeze the approved comparison scope; no GPU, no new answers, no submission."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/stream_prune_repair_prepare_20261104_01'
panel_path=ROOT/'runs/memory_prune_panel150_prepare_20261103_01/regression150.json'
reference_path=ROOT/'runs/protection_streamingbench_prepare_20261103_01/causal_reference.json'
panel=json.loads(panel_path.read_text(encoding='utf-8'))
reference=json.loads(reference_path.read_text(encoding='utf-8'))['prefixes']
questions=[q for q in panel if q['dataset']=='StreamingBench']
assert len(questions)==75 and len({q['id'] for q in questions})==75
assert len({q['source_id'] for q in questions})==72
counts=[]
for q in questions:
    state=reference[q['id']]['state']
    assert not state['future_RGB_emitted'] and state['complete_pairs']>0
    assert state['last_committed_pts'] <= q['query_time_s']+1e-6
    counts.append(dict(id=q['id'],source_id=q['source_id'],query_time_s=q['query_time_s'],
        complete_pairs=state['complete_pairs'],full_visual_tokens=108*state['complete_pairs'],
        recent_full_tokens=108*min(8,state['complete_pairs'])))
maxima={}
for c in counts:maxima[c['source_id']]=max(maxima.get(c['source_id'],0),c['complete_pairs'])
contract=dict(stage='frozen_scope_and_token_lower_bounds_only',gpu_submitted=False,
    full_interface_preflight_completed=False,questions=75,sources=72,
    new_arms=['unpruned_complete_history','guarded_LS','guarded_recent8pairs','initial_LS'],
    reuse_arms=['unpruned_recent8pairs','unpruned_LS','fixed_LS','old_CE_LS','text_only'],
    maximum_new_answers=300,regression_answers='to_be_frozen_before_submission',
    protocol='Inherited job220382 B explanation prompt, shift0, max128 generation; option correctness primary',
    resolution=[288,384],fps=1,LS_visual_budget=864,LS_recent_budget=432,
    event_rule='Inherited adaptive feature-change rule, no fixed-three-pair substitution',
    baseline_recent_pairs=8,checkpoints={'guarded':'runs/repair-guard-220499/final.pt',
        'initial':'runs/training-order-main-220483/initial.pt'},
    max_full_visual_tokens=max(c['full_visual_tokens'] for c in counts),
    input_visual_pairs_once=sum(maxima.values()),
    capacity_note='Visual counts exclude timestamps, prefix, suffix and output reserve; not final context/memory budget',
    overflow_policy='No silent truncation; retain failure reason and report full/common denominators',
    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in [panel_path,reference_path]})
OUT.mkdir(parents=True,exist_ok=True)
for name,obj in [('questions.json',questions),('input_counts.json',counts),('contract.json',contract)]:
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:contract[k] for k in ['questions','sources','max_full_visual_tokens','input_visual_pairs_once','maximum_new_answers']}))
