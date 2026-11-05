"""Offline content-choice accounting; unresolved free text remains separately visible."""
import argparse,json
from pathlib import Path
from guard import ROOT,ARMS
P=ROOT/'runs/recent_detail_prepare_20261103_01'

def analyze(rows,labels):
    grouped={a:{} for a in ARMS}
    for r in rows:
        assert r['arm'] in grouped and r['id'] in labels and r['id'] not in grouped[r['arm']]
        assert r['read_state_unchanged'] and r['branch_released']
        assert r['bank_status']['stored_feature_tokens']<=r['bank_status']['total_visual_budget']
        grouped[r['arm']][r['id']]=r
    assert all(set(d)==set(labels) for d in grouped.values())
    result={}
    for arm,data in grouped.items():
        correct={i:r['original_selected_index']==labels[i]['correct_index'] for i,r in data.items()}
        out={}
        for role in ('current','old'):
            ids=[i for i in labels if labels[i]['role']==role]
            out[role]=dict(questions=len(ids),selected_content_correct=sum(correct[i] for i in ids),
                          unresolved_ids=[i for i in ids if data[i]['original_selected_index'] is None])
        cases={v['case_id'] for v in labels.values()}
        out['late_both_correct']=sum(all(correct[f'{case}:late:{role}'] for role in ('current','old')) for case in cases)
        out['versus']={}
        for base in ARMS:
            bc={i:r['original_selected_index']==labels[i]['correct_index'] for i,r in grouped[base].items()}
            out['versus'][base]=dict(gained=[i for i in correct if correct[i] and not bc[i]],lost=[i for i in correct if bc[i] and not correct[i]])
        result[arm]=out
    return dict(methods=result,interpretation='Familiar development questions; option content only, unresolved text needs content review; no explanation score or standalone speed evidence')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    done=json.loads((a.run/'completed.json').read_text());assert done['answers']==54 and done['status']=='complete'
    rows=[json.loads(s) for s in (a.run/'answers.jsonl').read_text().splitlines()]
    labels=json.loads((P/'labels_LOCAL_ONLY.json').read_text(encoding='utf-8'))
    result=analyze(rows,labels)
    with (a.run/'analysis.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))
