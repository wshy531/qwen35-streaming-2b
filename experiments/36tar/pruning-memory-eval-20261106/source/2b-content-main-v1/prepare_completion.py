"""Manual time-budget completion plan, only after a source-atomic partial receipt.

This never submits. It cannot change source selection based on answers. Retains
the original interrupted records and restarts only incomplete whole sources.
"""
import argparse,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];PREP=ROOT/'runs/content_main_prepare_20261101_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(collection,dest):
    assert dest.resolve().is_relative_to((ROOT/'runs').resolve()) and not dest.exists()
    receipt=read(collection/'collection.json')
    assert receipt['status']=='partial' and receipt['source_atomic']
    missing=receipt['remaining_sources'];assert 0<len(missing)<=32
    qs=read(PREP/'requests.json');e=read(PREP/'expected_prefixes.json')
    costs={v:max(e[q['question_id']]['pairs'] for q in qs if q['video_id']==v) for v in missing}
    groups=[[],[],[],[]];loads=[0]*4
    for v in sorted(missing,key=lambda v:(-costs[v],v)):
        i=min(range(4),key=lambda i:(loads[i],i));groups[i].append(v);loads[i]+=costs[v]
    # Fixed ten-minute allocations only; reject a completion plan that is too large.
    assert max(loads)<=850,'completion exceeds bounded remaining-window input budget'
    dest.mkdir()
    for name in ('requests.json','videos.json','expected_prefixes.json','references.json','protocol.json'):
        (dest/name).write_bytes((PREP/name).read_bytes())
    tasks=[dict(name=f'quality_{i}',kind='quality',modes=['full','post_fixed','mid_fixed','post_content','mid_content'],
                videos=sorted(g),minutes=10) for i,g in enumerate(groups) if g]
    assert set(v for task in tasks for v in task['videos'])==set(missing)
    (dest/'tasks.json').write_text(json.dumps(tasks,indent=2),encoding='utf-8')
    record=dict(status='prepared_only_not_submitted',source_atomic_receipt_SHA=sha(collection/'collection.json'),
        sources=missing,balanced_pair_loads=loads,answers=sum(q['video_id'] in missing for q in qs)*5,
        warmup_answers=len(tasks)*5,max_generated_tokens=(sum(q['video_id'] in missing for q in qs)*5+len(tasks)*5)*16,
        max_vision=sum(costs.values())*5+len(tasks)*5,new_questions=0,algorithm_changes=0,
        no_score_based_selection=True,original_failures_preserved=True,manual_completion_round=1)
    (dest/'completion_plan.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args();main(a.collection,a.dest)
