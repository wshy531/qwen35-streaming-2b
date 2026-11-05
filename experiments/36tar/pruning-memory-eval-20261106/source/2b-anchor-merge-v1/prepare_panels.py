"""Reuse entire existing panels; no candidate outputs read and no cherry-picking."""
import json,hashlib
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
evaluation=read(R/'runs/learned_prune_main_prepare_20261101_01/eval.json')
train=read(R/'runs/learned_prune_main_prepare_20261101_01/train.json')
panel=read(R/'runs/evidence_scale_prepare_20261102_01/panel.json')
dev=[q for q in evaluation if q['role']=='development'];reg=[q for q in evaluation if q['role']=='confirmation']
assert (len(dev),len(reg),len(panel))==(345,307,300)
sets={'development345':dev,'opened_regression307':reg,'comprehensive300':panel}
train_ids={q['id'] for q in train};train_vids={q['video_id'] for q in train}
out=R/'runs/anchor_merge_panels_20261103_01';out.mkdir(exist_ok=False)
summary={};idsets={}
for name,qs in sets.items():
    assert not train_ids&{q['id'] for q in qs}
    assert not train_vids&{q.get('video_id') for q in qs}
    clean=[{k:v for k,v in q.items() if k not in ('answer_index','answer_text')} for q in qs]
    raw=json.dumps(clean,ensure_ascii=False,indent=2).encode('utf-8');(out/(name+'.json')).write_bytes(raw)
    idsets[name]={q['id'] for q in qs}
    summary[name]=dict(questions=len(qs),file_sources=len({q.get('source_id',q.get('video_id')) for q in qs}),
        types=dict(Counter(q.get('type',q.get('task_type')) for q in qs)),sha256=hashlib.sha256(raw).hexdigest())
summary['overlap_questions']={a+'__'+b:len(idsets[a]&idsets[b]) for i,a in enumerate(sets) for b in list(sets)[i+1:]}
summary['unique_questions_all_three']=len(set.union(*idsets.values()))
summary['primary_unique_questions']=len(idsets['development345']|idsets['comprehensive300'])
summary['execution_ready']=False
summary['roles']='345 version development; 300 old multi-task regression; 307 reserved old regression. None is new blind confirmation.'
summary['protocols']='Keep original short-answer two rotations for345, explanation one original order for300; do not merge their scores.'
summary['planned_arms']=['full','fixed12','heuristic11','local_anchor_merge']
summary['formal_answers_primary']=345*4*2+300*4
summary['GPU_submitted']=False
(out/'inventory.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
