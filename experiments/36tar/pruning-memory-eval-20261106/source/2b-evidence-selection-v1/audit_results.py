"""Independent raw-output bookkeeping; does not change frozen scoring/gates."""
import argparse,json,re
from pathlib import Path
from prepare import PREP,read,save

def main(run,output):
    rows=[json.loads(l) for l in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    done=read(run/'completed.json');tasks=read(PREP/'inputs/requests.json')
    plan={r['task_id']:r for r in read(PREP/'runtime_plan.json')['tasks']}
    cpu={(r['task_id'],r['arm'],r['shift']):r for r in read(PREP/'local_real_layout_preflight.json')['rows']}
    labels=read(PREP/'audit/labels_LOCAL_ONLY.json')
    keyed={(r['task_id'],r['arm'],r['shift']):r for r in rows}
    assert len(rows)==len(keyed)==len(cpu)==1600 and set(keyed)==set(cpu)
    refs=available=missing=actual_calls=formal_tokens=0
    scores={}
    for key,r in keyed.items():
        tid,arm,shift=key;c=cpu[key]
        assert r['available']==c['available']
        if not r['available']:
            assert arm=='nonmatching_nodes' and plan[tid]['selection']['control_history_indices'] is None
            missing+=1;continue
        available+=1
        assert r['input_ids_sha256']==c['input_ids_sha256']
        assert r['input_tokens']==c['input_tokens'] and r['visual_tokens']==c['visual_tokens']
        assert r['selected_history_indices']==c['selected_history_indices']
        assert r['branch_released'] and r['bank_unchanged'] and r['mrope_anchored_to_full']
        assert r['visual_tokens']<=864 and len(r['generated_ids'])<=8
        if not r.get('reused_from'):
            actual_calls+=1;formal_tokens+=len(r['generated_ids'])
        else:
            b=keyed[tid,r['reused_from'],shift]
            assert r['generated_ids']==b['generated_ids'] and r['input_ids_sha256']==b['input_ids_sha256']
        m=re.search(r'\b([A-E])\b',r['text'].strip().upper())
        choice=None if m is None else (ord(m.group(1))-65+shift)%5
        if arm in ('recent','full'):
            ref=next(x for x in plan[tid]['references'] if x['arm']==arm and x['shift']==shift)
            assert choice==ref['expected_choice']
            for k in ('generated_ids','position_ids_sha256'):
                if k in ref:assert r[k]==ref[k]
            refs+=1
        task=next(t for t in tasks if t['id']==tid)
        g=scores.setdefault(task['role']+'|'+arm,dict(correct=0,available=0))
        g['correct']+=choice==labels[tid];g['available']+=1
    assert refs==640 and available==1545 and missing==55 and actual_calls==1105
    assert done['reference_matches']==refs and done['model_answers']==actual_calls and done['vision']==2796
    assert done['warmup_answers']==2 and 2<=done['generated_tokens']-formal_tokens<=16
    assert done['records']==1600 and done['trained'] is False and done['pretrained_loads']==1
    save(output,dict(status='passed_independent_raw_result_audit',reference_choices_and_inputs=refs,
        valid_layout_hashes=available,unavailable_control_slots=missing,actual_formal_model_calls=actual_calls,
        formal_generated_tokens_recounted=formal_tokens,warmup_generated_tokens_from_difference=done['generated_tokens']-formal_tokens,
        content_scores_recomputed=scores,format_not_penalized=True,independent_human_labels=False,
        shared_vision_not_end_to_end_cost_benchmark=True))
    print(dict(status='passed',references=refs,valid_layouts=available,formal_calls=actual_calls))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run,a.output)
