"""Combine verified reused outputs with new answers; replay is never recounted."""
from statistics import median
from ..sember_input_diagnostic.scoring import build as old_build
from .controls import PRESETS,expected_keys,replay_ids


def build(queries,labels,reused,fresh):
    primary=[*reused,*(r for r in fresh if r['provenance']=='new')]
    keyed={(r['id'],r['arm'],r['resolution'],r['shift']):r for r in primary}
    if len(keyed)!=len(primary) or set(keyed)!=expected_keys(queries):
        raise ValueError('unique complete 5100-answer analysis required')
    replay=[r for r in fresh if r['provenance']=='replay']
    required={(qid,a,r,s) for qid in replay_ids(queries) for a,r in
              (('native','low'),('native','mid'),('text_only','none')) for s in range(5)}
    if len(replay)!=270 or {(r['id'],r['arm'],r['resolution'],r['shift']) for r in replay}!=required:
        raise ValueError('complete replay grid required')
    for row in replay:
        ref=keyed[row['id'],row['arm'],row['resolution'],row['shift']]
        for k in ('generated_ids','text','input_ids_sha256','position_ids_sha256','frames_sha256','pixels_sha256'):
            if row[k]!=ref[k]: raise ValueError('reused answer replay mismatch')
    mapping={'B':'LS','R':'A06','F':'native','T':'text_only'}
    details=[]; totals={}
    for res in PRESETS:
        rows=[]
        for q in queries:
            for old,arm in mapping.items():
                for shift in range(5):
                    r='none' if arm=='text_only' else res
                    rows.append(dict(keyed[q['id'],arm,r,shift],arm=old))
        summary,scored=old_build(queries,labels,rows)
        for old,arm in mapping.items():
            if arm=='text_only' and res!='low': continue
            r='none' if arm=='text_only' else res
            totals[arm+'/'+r]=summary['totals'][old]
            details.extend(dict(d,arm=arm,resolution=r) for d in scored if d['arm']==old)
    values={(d['id'],d['arm'],d['resolution'],d['shift']):d['correct'] for d in details}
    pairs=[]
    for arm in ('LS','A06','native'):
        pairs.extend(((arm,r),(arm,'low')) for r in ('mid','high'))
        pairs.append(((arm,'high'),(arm,'mid')))
    for res in PRESETS:
        pairs.extend(((arm,res),('native',res)) for arm in ('LS','A06'))
        pairs.extend(((arm,res),('text_only','none')) for arm in ('LS','A06','native'))
    comparisons={}
    for a,b in pairs:
        gains=[]; losses=[]; unresolved=[]
        for q in queries:
            for s in range(5):
                av,bv=values[q['id'],*a,s],values[q['id'],*b,s]
                entry=dict(id=q['id'],shift=s,source_path=q['source_path'])
                if av is None or bv is None: unresolved.append(entry)
                elif av and not bv: gains.append(entry)
                elif bv and not av: losses.append(entry)
        comparisons['/'.join(a)+'_vs_'+'/'.join(b)]=dict(corrections=gains,losses=losses,
            net=len(gains)-len(losses),unresolved=unresolved)
    # Same 18 questions x five positions for ALL measured latency comparisons.
    selected=set(replay_ids(queries)); costs={}
    for arm,res in sorted({(r['arm'],r['resolution']) for r in fresh}):
        rows=[r for r in fresh if r['arm']==arm and r['resolution']==res and r['id'] in selected]
        if len(rows)!=90: raise ValueError('matched cost panel incomplete')
        seconds=sorted(r['answer_seconds'] for r in rows)
        if any(v<0 for v in seconds): raise ValueError('negative latency')
        entry=dict(matched_questions=18,answers=90,median_answer_seconds=median(seconds),
            p95_answer_seconds=seconds[85],mean_input_tokens=sum(r['input_tokens'] for r in rows)/90)
        if arm in ('LS','A06'):
            one=[r for r in rows if r['shift']==0]
            entry.update(mean_encode_seconds=sum(r['encode_seconds'] for r in one)/18,
                mean_update_seconds=sum(r['bank_update_seconds'] for r in one)/18,
                peak_bank_bytes=max(r['bank_peak_bytes'] for r in one),
                shared_encode_cost_charged_in_full_to_each_method=True)
        costs[arm+'/'+res]=entry
    return dict(totals=totals,paired=comparisons,matched_costs=costs,unique_answers=len(details),
        independent_questions=102,independent_videos=18,replay_excluded_from_accuracy=270,
        human_semantic_review='pending',held_out=False,
        interpretation='oracle evidence diagnostic; resolution AND token budget vary; not full-stream ranking',
        decision='report accuracy/cost frontier per strategy; no automatic default change'),details
