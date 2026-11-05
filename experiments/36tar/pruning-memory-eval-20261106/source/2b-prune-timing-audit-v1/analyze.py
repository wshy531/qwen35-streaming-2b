import argparse
from collections import Counter
from pathlib import Path
from audit_common import *
def main(run):
    done=read(run/'completed.json');assert done['status']=='complete'
    rows=[json.loads(x) for x in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    qs=read(PREP/'requests.json');labels=read(PREP/'labels_LOCAL_ONLY.json')
    expected={(q['id'],m,s) for q in qs for m in MODES for s in range(4)}
    assert len(rows)==len(expected)==120 and {(r['task_id'],r['arm'],r['shift']) for r in rows}==expected
    keyed={}
    for r in rows:
        assert r['choice']==choice(r['text'],r['shift'])
        r['correct']=r['choice']==labels[r['task_id']];keyed[r['task_id'],r['arm'],r['shift']]=r
    contrasts=[]
    for q in qs:
        record=dict(id=q['id'],type=q['task_type'],evidence_caveat=q['task_type'] in ['Prospective Reasoning','Counting'])
        record['methods']={m:dict(original_choice=keyed[q['id'],m,0]['choice'],
            original_correct=keyed[q['id'],m,0]['correct'],correct_rotations=sum(keyed[q['id'],m,s]['correct'] for s in range(4))) for m in MODES}
        record['paired']={k:dict(post_only_correct=sum(keyed[q['id'],'post_'+k,s]['correct'] and not keyed[q['id'],'mid_'+k,s]['correct'] for s in range(4)),
            mid_only_correct=sum(keyed[q['id'],'mid_'+k,s]['correct'] and not keyed[q['id'],'post_'+k,s]['correct'] for s in range(4))) for k in ['fixed','content']}
        contrasts.append(record)
    audits=[json.loads(x) for x in (run/'pair_audit.jsonl').read_text().splitlines()]
    assert len(audits)==513 and len({(x['task_id'],x['pair_index']) for x in audits})==513
    assert all(x['previous_middle_features_equal'] and all(c['same_IDs'] for c in x['checks'].values()) for x in audits)
    save(run/'analysis.json',dict(cases=contrasts,diagnosis_not_population_accuracy=True,rotations_not_independent_questions=True,
        all_pair_ID_checks_passed=True,unparsed=sum(x['choice'] is None for x in rows),no_default_change=True))
    print(json.dumps(contrasts,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);main(p.parse_args().run)
