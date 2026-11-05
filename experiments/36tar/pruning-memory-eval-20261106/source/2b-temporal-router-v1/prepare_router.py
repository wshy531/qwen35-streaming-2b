"""Keep every existing task; assemble paired cached outputs without new QA."""
from pathlib import Path
from temporal_policy import normalize,lexical_route,VERSION,STAGES,HISTORY_PATTERN,CURRENT_PATTERN
from route_common import ROOT,PREP,INPUTS,read,save,sha,question_key


def main():
    old=ROOT/'runs/memory_collab_prepare_20261026_01'
    transition=ROOT/'runs/perception_transition_prepare_20261027_01'
    run=ROOT/'runs/2b-perception-transition-219310'
    requests=read(old/'partner_readonly/requests-no-labels.json')
    details=read(old/'partner_readonly/details.json')
    keyed={(r['id'],r['arm'],r['shift']):r for r in details}
    if len(keyed)!=len(details):raise ValueError('duplicate partner details')
    model_requests={};tasks=[];branches=[]
    def add(taskid,question,**meta):
        question=normalize(question);key=question_key(question)
        model_requests.setdefault(key,dict(id=key,question=question))
        tasks.append(dict(id=taskid,router_id=key,**meta))
    for i,q in enumerate(requests):
        taskid=f'o{i:03d}'
        add(taskid,q['question'],cohort='official102_reference',source_name=Path(q['source_path']).name,
            query_time_s=q['query_time_s'],role='unpartitioned_official_reference',case_id=None,
            original_task_id=q['id'],independent_content_review=False)
        for shift in range(5):
            pair={}
            for branch in ('recent','full'):
                row=keyed[(q['id'],'LS4_4/'+branch,shift)]
                if type(row['correct']) is not bool:raise ValueError('explicit cached correctness required')
                pair[branch]=dict(choice=row['content_choice_original'],correct=row['correct'])
            branches.append(dict(task_id=taskid,shift=shift,**pair))
    import json
    raw=[json.loads(s) for s in (run/'transition_answers.jsonl').read_text(encoding='utf-8').splitlines()]
    newkey={(r['task_id'],r['arm'],r['shift']):r for r in raw}
    import sys
    sys.path.insert(0,str(ROOT/'experiments/2b-perception-transition-v1'))
    from transition_scoring import parse_original
    for i,q in enumerate(read(transition/'inputs/tasks_LOCAL_ONLY.json')):
        taskid=f'd{i:03d}'
        add(taskid,q['question'],cohort='transition6_development',source_name=q['source_name'],
            query_time_s=q['query_time_s'],role=q['role'],stage=q['stage'],case_id=q['case_id'],
            original_task_id=q['id'],independent_content_review=False)
        for shift in range(5):
            pair={}
            for branch in ('recent','full'):
                row=newkey[(q['id'],branch,shift)]
                choice=parse_original(row['text'],shift)
                pair[branch]=dict(choice=choice,correct=choice==q['correct_index'])
            branches.append(dict(task_id=taskid,shift=shift,**pair))
    assert len(tasks)==120 and len(branches)==600
    expected={(t['id'],s) for t in tasks for s in range(5)}
    assert {(r['task_id'],r['shift']) for r in branches}==expected
    save(INPUTS/'router_requests.json',list(model_requests.values()))
    save(INPUTS/'evaluation_tasks_LOCAL_ONLY.json',tasks)
    save(INPUTS/'frozen_branch_outputs_LOCAL_ONLY.json',branches)
    save(INPUTS/'lexical_decisions.json',{key:lexical_route(v['question']) for key,v in model_requests.items()})
    provenance=[old/'partner_readonly/requests-no-labels.json',old/'partner_readonly/details.json',
        old/'partner_readonly/protocol.json',transition/'inputs/tasks_LOCAL_ONLY.json',run/'transition_answers.jsonl']
    save(INPUTS/'provenance.json',dict(source_files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in provenance},
        source_jobs=['partner/perception-memory-eval-20261025-01','219310'],
        cached_answers_per_task=10,options_not_router_input=True,labels_not_router_input=True,
        source_names_and_times_not_router_input=True,held_out=False,old_review_pending=True))
    count=len(model_requests)
    save(INPUTS/'protocol.json',dict(schema='temporal-router-paired-policy-v1',policy_version=VERSION,
        source_question_times=120,source_video_files=len({t['source_name'] for t in tasks}),
        distinct_question_texts=count,paired_rotation_rows=600,cached_branch_answers=1200,
        stage_states=STAGES,balanced_view_counts=[5,4],maximum_router_forwards=count*9,
        model_generated_tokens=0,new_visual_forwards=0,new_QA_answers=0,
        lexical_rule=dict(history_pattern=HISTORY_PATTERN,current_pattern=CURRENT_PATTERN,
                          priority='history first, present second, otherwise full'),
        frozen_lexical_before_outcome_analysis=True,
        decision_scope='one per normalized question; shared across answer-option rotations and repeated question text',
        outcomes_used_only_by_offline_evaluator=True,
        official_scope='original-label reference, not reviewed current-perception evidence',
        development_scope='all six opened assistant-reviewed cases, no independent review',
        old_GPU_authorization_reused=False,
        stop='No prompt/threshold sweep. Must beat lexical control, preserve old-question performance and reduce current harm across sources before independent validation.'))
    print(dict(tasks=len(tasks),question_texts=count,sources=len({t['source_name'] for t in tasks}),
               cached_pair_rows=len(branches),maximum_router_forwards=count*9,GPU_jobs=0))


if __name__=='__main__':main()
