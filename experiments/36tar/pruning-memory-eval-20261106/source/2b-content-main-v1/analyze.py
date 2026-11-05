"""Content-first paired scores plus separately measured standalone costs."""
import argparse,json,sys
from collections import defaultdict
from pathlib import Path
from common import ROOT,PREP,read,save,MODES
from sember_protocol import parse_choice,format_ok
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def source_interval(deltas,counts):
    """Resample videos as clusters; associated questions never count as independent."""
    import numpy as np
    keys=sorted(deltas);x=np.array([deltas[k] for k in keys]);n=np.array([counts[k] for k in keys])
    rng=np.random.default_rng(20261101);draw=rng.integers(0,len(keys),(10000,len(keys)))
    vals=x[draw].sum(1)/n[draw].sum(1)
    low,high=np.quantile(vals,[.025,.975])
    return dict(content_accuracy_delta=sum(deltas.values())/sum(counts.values()),
        cluster_bootstrap_95_interval=[float(low),float(high)],unit='video_file',replicates=10000,
        source_identity_may_not_equal_independent_shoot=True,opened_development_exploratory=True)
def quality(folders,out):
    qs={q['question_id']:q for q in read(PREP/'requests.json')}
    labels={q['question_id']:q for q in read(ROOT/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json')}
    rows=[];completions=[];collected=False
    if len(folders)==1 and (Path(folders[0])/'collection.json').exists():
        assert read(Path(folders[0])/'collection.json')['status']=='complete'
        rows=lines(Path(folders[0])/'answers.jsonl');collected=True
    else:
        for folder in folders:
            root=Path(folder);assert read(root/'completed.json')['status']=='complete'
            for sub in root.glob('quality_*'):
                completions.append(read(sub/'completed.json'));rows+=lines(sub/'answers.jsonl')
    assert len(rows)==2390
    index={(r['question_id'],r['arm']):r for r in rows}
    assert set(index)=={(q,m) for q in qs for m in MODES}
    for (q,m),r in index.items():
        r['choice']=parse_choice(r['text'],len(qs[q]['options']))
        r['correct']=r['choice']==labels[q]['correct_letter'];r['format_ok']=format_ok(r['text'],len(qs[q]['options']))
    def stats(ids,m):
        vals=[index[q,m] for q in ids]
        return dict(n=len(vals),correct=sum(r['correct'] for r in vals),format_ok=sum(r['format_ok'] for r in vals),
            unparsed=sum(r['choice'] is None for r in vals))
    scores={m:stats(qs,m) for m in MODES}
    bytype={cat:{m:stats([q for q in qs if qs[q]['question_category']==cat],m) for m in MODES}
        for cat in sorted({q['question_category'] for q in qs.values()})}
    paired={}
    for a,b in [('mid_fixed','full'),('post_fixed','full'),('post_content','full'),('mid_content','full'),
                ('mid_content','mid_fixed'),('post_content','post_fixed'),('mid_content','post_content')]:
        gained=[q for q in qs if index[q,a]['correct'] and not index[q,b]['correct']]
        lost=[q for q in qs if not index[q,a]['correct'] and index[q,b]['correct']]
        bysource={v:sum(int(index[q,a]['correct'])-int(index[q,b]['correct']) for q in qs if qs[q]['video_id']==v)
            for v in sorted({q['video_id'] for q in qs.values()})}
        counts={v:sum(qs[q]['video_id']==v for q in qs) for v in bysource}
        paired[a+'_vs_'+b]=dict(gained=gained,lost=lost,net=len(gained)-len(lost),by_source_delta=bysource,
            uncertainty=source_interval(bysource,counts),
            reference_correct=sum(index[q,b]['correct'] for q in qs),source_positive=sum(v>0 for v in bysource.values()),
            source_negative=sum(v<0 for v in bysource.values()))
    baseline_matches=sum(c['baseline_matches'] for c in completions)
    if collected:
        baseline_matches=0
        for key,ref in read(PREP/'references.json').items():
            q,mode=key.split('|');r=index[q,mode]
            for field in ref:assert r[field]==ref[field]
            baseline_matches+=1
    result=dict(status='complete',scores=scores,by_type=bytype,paired=paired,
        original_baseline_matches=baseline_matches,source_atomic_collection_used=collected,
        content_format_separate=True,opened_development_only=True,no_new_source_confirmation=True,
        scored=[dict(question_id=q,mode=m,choice=r['choice'],correct=r['correct'],text=r['text']) for (q,m),r in index.items()])
    assert result['original_baseline_matches']==1434
    save(out,result);print(json.dumps(scores,indent=2));print(json.dumps({k:{f:v[f] for f in ('net','source_positive','source_negative','uncertainty')} for k,v in paired.items()},indent=2))
def cost(folder,out):
    folder=Path(folder);assert read(folder/'completed.json')['status']=='complete'
    tasks=[x for x in read(PREP/'tasks.json') if x['kind']=='cost'];totals={m:defaultdict(float) for m in MODES};per={m:{} for m in MODES}
    for task in tasks:
        c=read(folder/task['name']/'completed.json');assert c['status']=='complete' and c['cost_independent_process']
        mode=task['modes'][0]
        for p in c['profiles']:
            per[mode][p['video_id']]=p['total_wall_seconds']
            for field,val in [('total_seconds',p['total_wall_seconds']),('decode_seconds',p['source']['decode_seconds']),
                ('canvas_seconds',p['source']['transform_seconds']),('preprocess_seconds',p['timing']['preprocess_seconds']),
                ('vision_seconds',p['timing']['vision_seconds']),('update_seconds',p['timing']['bank_or_main_update_seconds']),('QA_seconds',p['QA_seconds'])]:totals[mode][field]+=val
            totals[mode]['max_selector_bytes']=max(totals[mode]['max_selector_bytes'],p['session']['banks'][mode]['selector_bytes'])
            totals[mode]['peak_GPU_allocated_bytes']=max(totals[mode]['peak_GPU_allocated_bytes'],c['peak_GPU_allocated_bytes'])
    assert all(len(v)==12 for v in per.values())
    ratios={m:totals[m]['total_seconds']/totals['full']['total_seconds'] for m in MODES}
    result=dict(status='complete',totals=totals,ratios_to_full=ratios,source_wall=per,
        excludes_loading_warmups_integrity=True,same_12_sources_57_questions=True,
        five_modes_four_disjoint_cyclic_blocks=True,OS_cache_not_flushed=True,
        not_edge_or_1x=True,exact_output_replay_vs_quality_pending=True)
    save(out,result);print(json.dumps(dict(totals=totals,ratios_to_full=ratios),indent=2))
def replay(quality_folders,cost_folder,out):
    index={}
    if len(quality_folders)==1 and (Path(quality_folders[0])/'collection.json').exists():
        assert read(Path(quality_folders[0])/'collection.json')['status']=='complete'
        for r in lines(Path(quality_folders[0])/'answers.jsonl'):index[r['question_id'],r['arm']]=r
    else:
        for folder in quality_folders:
            for sub in Path(folder).glob('quality_*'):
                for r in lines(sub/'answers.jsonl'):index[r['question_id'],r['arm']]=r
    checks=0
    for task in read(PREP/'tasks.json'):
        if task['kind']!='cost':continue
        for r in lines(Path(cost_folder)/task['name']/'answers.jsonl'):
            ref=index[r['question_id'],r['arm']]
            for key in ('text','generated_ids','bounded_context_ids_sha256','question_suffix_sha256','visual_tokens'):
                if r[key]!=ref[key]:raise ValueError('standalone quality drift: '+str((r['question_id'],r['arm'],key)))
            checks+=1
    assert checks==285
    save(out,dict(status='passed',standalone_answers_matched=checks,
        same_input_ids_tokens_and_answers=True,quality_source_replay=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quality',nargs='+');p.add_argument('--cost');p.add_argument('--replay',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.replay:replay(a.quality,a.cost,a.out)
    elif a.quality:quality(a.quality,a.out)
    elif a.cost:cost(a.cost,a.out)
    else:p.error('quality or cost required')
