"""Pure CPU contracts plus replay of actual, previously recorded node metadata."""
import json
from pathlib import Path
from prepare import ROOT,PREP,read,sha,save
from selection import choose,subset_columns

def main(output=None, allow_missing_upstream=False):
    missing=[]
    for p,digest in read(PREP/'source_lock.json').items():
        if not (ROOT/p).exists():
            if not allow_missing_upstream:raise FileNotFoundError(p)
            missing.append(p)
        else:assert sha(ROOT/p)==digest
    for p,digest in read(PREP/'frozen_inputs_lock.json').items():assert sha(ROOT/p)==digest
    tasks=read(PREP/'inputs/requests.json');bindings=read(PREP/'audit/evidence_bindings_LOCAL_ONLY.json')
    labels=read(PREP/'audit/labels_LOCAL_ONLY.json');assert len(tasks)==64
    assert all('correct_index' not in t and 'intervals' not in t for t in tasks)
    assert set(bindings)==set(labels)=={t['id'] for t in tasks}
    coverage=read(PREP/'audit/recorded_node_coverage.json')
    for r in coverage:
        t=next(t for t in tasks if t['id']==r['task_id'])
        result=choose(r['history_nodes'],r['recent_nodes'],bindings[t['id']]['intervals'],t['query_time_s'])
        assert all(result[k]==r[k] for k in result)
        assert result['total_visual_tokens']<=864 and len(result['selected_history_indices'])<=2
    checks=[]
    def reject(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('illegal input accepted')
    base=coverage[0];h,r=base['history_nodes'],base['recent_nodes'];q=r[-1]['end']
    reject(lambda:choose(h,r,[[q+1,q+2]],q));checks.append('future evidence rejected')
    reject(lambda:choose(h,r[:-1],[],q));checks.append('recent truncation rejected')
    reject(lambda:choose(h,r,[],q,max_nodes=3));checks.append('node cap fixed')
    empty=choose(h,r,[],q);assert empty['selected_history_indices']==[] and not empty['any_interval_hit']
    checks.append('no evidence remains unknown not recent sufficient')
    assert subset_columns([2,3,4],[0,2])==[0,1,5,6,7,8]
    reject(lambda:subset_columns([2,3],[1,0]));reject(lambda:subset_columns([2,3],[2]))
    checks.append('whole node column mapping and invalid IDs')
    # Same token count is necessary for the control, not just same number of nodes.
    result=dict(status='passed_CPU_metadata_and_selection_contracts',tasks=len(tasks),
        layouts_planned=64*5*5,actual_model_contexts_checked=0,real_weight_loads=0,
        true_recorded_queries_replayed=len(coverage),checks=checks,GPU_jobs=0,
        old_queries_with_selected_history=sum(bool(x['selected_history_indices']) for x in coverage if ':old' in x['task_id']),
        old_queries_selected_compressed=sum(any(x['selected_nodes_compressed']) for x in coverage if ':old' in x['task_id']),
        no_language_model_or_video_encoder_run=True,not_model_accuracy_evidence=True,
        upstream_sources_missing_here=missing,all_upstream_sources_verified_here=not missing,
        frozen_derived_inputs_verified=True,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')})
    save(output or PREP/'CPU_metadata_preflight.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--allow-missing-upstream',action='store_true')
    a=p.parse_args();main(a.output,a.allow_missing_upstream)
