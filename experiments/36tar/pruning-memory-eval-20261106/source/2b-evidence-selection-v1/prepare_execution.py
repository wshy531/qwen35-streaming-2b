"""Bind existing complete input receipts before any new real model answers."""
from collections import Counter,defaultdict
from prepare import ROOT,PREP,read,save,sha
from selection import choose

def main():
    tasks=read(PREP/'inputs/requests.json');bindings=read(PREP/'audit/evidence_bindings_LOCAL_ONLY.json')
    paths=[ROOT/'runs/perception_transition_prepare_20261027_01/canonical_input_reference_LOCAL_ONLY.json',
           ROOT/'runs/recent_evidence_prepare_20261028_01/inputs/evaluation_tasks_LOCAL_ONLY.json',
           ROOT/'runs/recent_evidence_prepare_20261028_01/inputs/requests.json',
           ROOT/'runs/recent_evidence_prepare_20261028_01/inputs/sources.json',
           ROOT/'runs/recent_evidence_prepare_20261028_01/inputs/frozen_branch_outputs_LOCAL_ONLY.json',
           ROOT/'runs/2b-perception-transition-219310/transition_answers.jsonl',
           ROOT/'runs/perception_transition_prepare_20261027_01/inputs/sources_LOCAL_ONLY.json']
    canonical=read(paths[0]);maps=read(paths[1]);oldreq={r['id']:r for r in read(paths[2])}
    original_to_local={r.get('original_task_id',r['id']):r['id'] for r in maps}
    frozen={(r['task_id'],r['shift']):r for r in read(paths[4])}
    transition=[__import__('json').loads(line) for line in paths[5].read_text(encoding='utf-8').splitlines() if line]
    tr={(r['task_id'],r['arm'],r['shift']):r for r in transition}
    coverage={r['task_id']:r for r in read(PREP/'audit/recorded_node_coverage.json')}
    origrows={(r['id'],r['arm'],r['shift']):r for r in canonical['rows']}
    source_rows={r['name']:dict(name=r['name'],bytes=r['bytes'],sha256=r['sha256'],frames={}) for r in read(paths[3])}
    selected=[]
    for t in tasks:
        local_id=original_to_local.get(t['id'],t['id'])
        q=oldreq[local_id]
        assert q['source_name']==t['source_name'] and q['query_time_s']==t['query_time_s']
        if t['role']=='official_reference':nodes=canonical['nodes'][t['id']+'|full']
        else:
            row=coverage[t['id']];nodes=row['history_nodes']+row['recent_nodes']
        assert len(nodes)==8 and q['committed_pairs']==nodes[-1]['last']+1
        match=choose(nodes[:-4],nodes[-4:],bindings[t['id']]['intervals'],t['query_time_s'])
        refs=[]
        for shift in range(5):
            for arm in ('recent','full'):
                if t['role']=='official_reference':
                    inp=origrows[t['id'],arm,shift]
                    choice=frozen[local_id,shift][arm]['choice']
                    ref=dict(shift=shift,arm=arm,input_ids_sha256=inp['input_ids_sha256'],expected_choice=choice)
                else:
                    inp=tr[t['id'],arm,shift]
                    ref=dict(shift=shift,arm=arm,input_ids_sha256=inp['input_ids_sha256'],
                             expected_choice=frozen[local_id,shift][arm]['choice'],generated_ids=inp['generated_ids'],
                             position_ids_sha256=inp['position_ids_sha256'])
                refs.append(ref)
        selected.append(dict(task_id=t['id'],committed_pairs=q['committed_pairs'],
            cutoff_s=q['recent_frames'][-1]['pts_s'],nodes=nodes,selection=match,references=refs))
        src=source_rows[t['source_name']]
        for f in q['recent_frames']:
            key=str(f['frame_id']);assert key not in src['frames'] or src['frames'][key]==f
            src['frames'][key]=f
    sources=[]
    for name in sorted({t['source_name'] for t in tasks}):
        src=source_rows[name];src['check_frames']=sorted(src.pop('frames').values(),key=lambda f:f['frame_id'])
        src['max_pairs']=max(r['committed_pairs'] for r in selected if next(t for t in tasks if t['id']==r['task_id'])['source_name']==name)
        src['cutoff_s']=max(r['cutoff_s'] for r in selected if next(t for t in tasks if t['id']==r['task_id'])['source_name']==name)
        sources.append(src)
    save(PREP/'runtime_plan.json',dict(tasks=selected,sources=sources,
        media_revision='897886cc967b2d0e6e45f1200605f5d3b0ffbb23',
        complete_visual_pairs=sum(s['max_pairs'] for s in sources),max_warmup_answers=2,
        max_formal_model_calls=64*5*4,max_total_model_calls=64*5*4+2,
        maximum_generation_tokens=(64*5*4+2)*8,record_rows_maximum=1600,
        lexical_answers_reused_from_matching_recent_full=True,
        all_old_640_reference_rows_required=True,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
    save(PREP/'all_recorded_node_coverage.json',selected)
    print(dict(real_recorded_query_layouts=len(selected),sources=len(sources),
        visual_pairs=sum(s['max_pairs'] for s in sources),
        interval_matches=Counter(bool(r['selection']['selected_history_indices']) for r in selected),
        unavailable_controls=sum(r['selection']['control_history_indices'] is None for r in selected)))
if __name__=='__main__':main()
