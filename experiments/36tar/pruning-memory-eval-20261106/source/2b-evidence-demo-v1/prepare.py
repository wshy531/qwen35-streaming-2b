"""Freeze a metadata-selected demonstration; never select by method outcomes."""
import collections, hashlib, json, sys
from pathlib import Path
R=Path(__file__).resolve().parents[2]
P=R/'runs/evidence_eval_prepare_20261102_01'
O=R/'runs/evidence_demo_prepare_20261102_01'
sys.path.insert(0,str(R/'experiments/2b-evidence-eval-v1'))
from response_v4 import parse
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def key(q):return hashlib.sha256(('task-evidence-demo-v1:'+q['id']).encode()).hexdigest()
def main():
    O.mkdir(exist_ok=True)
    selected=[]
    # This block reads metadata only; predictions and labels are opened later.
    for dataset in ('NExT-QA','StreamingBench'):
        pool=[q for q in read(P/'panel.json') if q['dataset']==dataset]
        by=collections.defaultdict(list)
        for q in sorted(pool,key=key):by[q['type']].append(q)
        chosen=[];used=set()
        while len(chosen)<12:
            before=len(chosen)
            for typ in sorted(by):
                candidates=[q for q in by[typ] if q['source_id'] not in used and q not in chosen]
                if candidates:
                    q=candidates[0];chosen.append(q);used.add(q['source_id'])
                    if len(chosen)==12:break
            assert len(chosen)>before,'not enough source-separated cases'
        selected+=chosen
    assert len(selected)==24
    save('selected_metadata.json',selected)
    materials_next={m['source_id']:m for m in read(P/'nextqa_materials_REMOTE.json')}
    materials_stream={m['material_id']:m for m in read(P/'stream_materials_REMOTE.json')}
    labels=read(P/'labels_all_LOCAL_ONLY.json')
    index={}
    for folder in ('evidence-nextqa-220067','evidence-stream-merged-20261102-01'):
        for line in (R/'runs'/folder/'answers.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row['protocol']=='B' and row['shift']==0:index[row['id'],row['arm']]=row
    tasks=[];cases=[]
    for qi,q in enumerate(selected):
        m=materials_next[q['source_id']] if q['dataset']=='NExT-QA' else materials_stream[q['id']]
        assert all(f['pts_s']<=q['query_time_s']+1e-6 for f in m['frames'])
        task=dict(task_id=f'task-{qi:02d}',question=q['question'],query_time_s=q['query_time_s'],frames=m['frames'])
        tasks.append(task)
        for arm in ('full','fixed12','heuristic11'):
            row=index[q['id'],arm];parsed=parse(row['text'],len(q['options']),'B',0)
            i=parsed['original_selected_index']
            cases.append(dict(opaque_id='case-'+hashlib.sha256((q['id']+arm).encode()).hexdigest()[:16],
                task_id=task['task_id'],proposed_answer=None if i is None else q['options'][i],
                response=row['text'],LOCAL_ONLY=dict(arm=arm,question_id=q['id'],source_id=q['source_id'],
                    dataset=q['dataset'],type=q['type'],official_correct=None if i is None else i==labels[q['id']],
                    input_tokens=row['input_tokens'],output_tokens=len(row['generated_ids']))))
    # Opaque deterministic order prevents a fixed method-position order at the judge.
    cases.sort(key=lambda c:c['opaque_id'])
    repeats=[]
    for dataset in ('NExT-QA','StreamingBench'):
        for arm in ('full','fixed12','heuristic11'):
            c=next(c for c in cases if c['LOCAL_ONLY']['dataset']==dataset and c['LOCAL_ONLY']['arm']==arm)
            repeats.append(dict(c,opaque_id='repeat-'+hashlib.sha256(c['opaque_id'].encode()).hexdigest()[:16],repeat_of=c['opaque_id']))
    profile={m['material_id']:m for m in read(R/'runs/evidence-profile-220065/materials.json')}
    calibrations=[]
    for i,c in enumerate(read(P/'calibration_candidates.json')):
        m=profile[c['material_id']]
        frames=[dict(f,path='runs/evidence-profile-220065/'+f['path']) for f in m['frames']]
        calibrations.append(dict(opaque_id=f'cal-{i:02d}',task_id=f'cal-task-{i:02d}',question=c['question'],frames=frames,
            proposed_answer=c['proposed_answer'],response=c['evidence'],LOCAL_ONLY=dict(expected_fact=c['expected_fact_verdict'],
            expected_support=c['expected_sufficiency'])))
    save('tasks.json',tasks);save('cases_LOCAL_ONLY.json',cases);save('repeats_LOCAL_ONLY.json',repeats)
    save('calibrations_LOCAL_ONLY.json',calibrations)
    summary=dict(questions=24,sources=len({(q['dataset'],q['source_id']) for q in selected}),
        real_explanations=72,repeats=6,calibration=20,shared_evidence_contracts=24,maximum_requests=122,
        new_GPU=0,new_generation=0,new_training=0,new_downloads=0,selection_without_predictions=True,
        datasets={d:dict(questions=12,types=dict(collections.Counter(q['type'] for q in selected if q['dataset']==d)),
            duration_range=[min(q['duration_s'] for q in selected if q['dataset']==d),max(q['duration_s'] for q in selected if q['dataset']==d)],
            query_range=[min(q['query_time_s'] for q in selected if q['dataset']==d),max(q['query_time_s'] for q in selected if q['dataset']==d)]) for d in ('NExT-QA','StreamingBench')})
    save('selection.json',summary);print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
