"""R0 single-round paired analysis; all original failures remain in scope."""
import hashlib,json,re,sys
from collections import Counter
from pathlib import Path
import numpy as np
from score_memory_eval import score,paired
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/2b_temporal_route_20261020_01';BASE=ROOT/'runs/2b_natural_eval_20261020_01'
ARMS=['all_memory','long_only','recent4','temporal_route']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]
def main():
    rows=read(BASE/'scorer_rows_local.json');protocol=read(OUT/'protocol.json')
    if sys.argv[1]=='freeze':
        save(OUT/'analysis_lock.json',dict(source_sha256=sha(__file__),scorer_sha256=sha(BASE/'scorer_rows_local.json'),cohorts=['public74','open47','new50'],FT_primary=True,C_secondary=True,strict_primary=True,leading_choice_format_diagnostic=True,pairwise_references=['all_memory','recent4','M0'],bootstrap_replicates=2000,seed=20261020,unit='conservative source group',all_memory_full_ID_replay_required=True,routing_input_and_output_branch_agreement=True,stop='one candidate, no revised regex or rescue after scoring'))
        print('R0 analysis frozen');return
    lock=read(OUT/'analysis_lock.json')
    if sha(__file__)!=lock['source_sha256'] or sha(BASE/'scorer_rows_local.json')!=lock['scorer_sha256']:raise ValueError('analysis identity drift')
    job=read(OUT/'submission_result.json')['job_id'];result=OUT/('gpu_'+job)/'result';ps=lines(result/'predictions.jsonl');completed=read(result/'completed.json');traces=read(result/'routing_traces.json')
    if len(ps)!=684 or len({(p['id'],p['arm']) for p in ps})!=684 or completed['records']!=684 or completed['protocol_sha256']!=sha(OUT/'protocol.json'):raise ValueError('684 fixed answers required')
    previous=lines(BASE/'gpu_217141/result/memory/predictions.jsonl');prior={(p['id'],p['arm']):p for p in previous};pp={(p['id'],p['arm']):p for p in ps}
    replay=sum(pp[r['id'],'all_memory']['generated_ids']==prior[r['id'],'M2']['generated_ids'] for r in rows)
    if replay!=171:raise ValueError('M2 full-ID replay failed; do not reuse baseline')
    tmap={t['id']:t for t in traces};branch_matches=0;input_matches=0
    for r in rows:
        t=tmap[r['id']];a='long_only' if t['used_long_only'] else 'all_memory';route=pp[r['id'],'temporal_route'];selected=pp[r['id'],a]
        branch_matches+=route['generated_ids']==selected['generated_ids'];input_matches+=route['input_ids_sha256']==selected['input_ids_sha256']
    if branch_matches!=171 or input_matches!=171:raise ValueError('routing branch mismatch')
    subsets={c:[r for r in rows if r['cohort']==c] for c in ['public74','open47','new50']}
    for c,rs in list(subsets.items()):
        subsets[c+'_FT']=[r for r in rs if r['stratum'] in ['F','T']];subsets[c+'_C']=[r for r in rs if r['stratum']=='C'];subsets[c+'_cue']=[r for r in rs if tmap[r['id']]['has_temporal_cue']]
    tables={};details=[]
    for name,rs in subsets.items():
        ids={r['id'] for r in rs};scores={a:score(rs,[p for p in ps if p['arm']==a and p['id'] in ids]) for a in ARMS};scores['M0']=score(rs,[p for p in previous if p['arm']=='M0' and p['id'] in ids])
        arms=[*ARMS,'M0'];groups=sorted({r['source_group_id'] for r in rs});rng=np.random.default_rng(20261020);draw=rng.integers(0,len(groups),(2000,len(groups)))
        matrix=np.array([[np.mean([d['correct'] for d in scores[a]['details'] if d['video']==g]) for a in arms] for g in groups]);boot=matrix[draw].mean(1)
        table={}
        for j,a in enumerate(arms):
            vals=dict(**scores[a]['overall'],macro_interval95=np.percentile(boot[:,j],[2.5,97.5]).tolist())
            for ref in ['all_memory','recent4','M0']:
                k=arms.index(ref);vals['vs_'+ref]=dict(**paired(scores[ref],scores[a]),macro_delta=float((matrix[:,j]-matrix[:,k]).mean()),macro_interval95=np.percentile(boot[:,j]-boot[:,k],[2.5,97.5]).tolist())
            preds=[p for p in (previous if a=='M0' else ps) if p['arm']==a and p['id'] in ids];normalized=[]
            for p in preds:
                m=re.match(r'^\s*([A-Ea-e])(?:\b|[.!)\s])',p['text']);normalized.append(dict(p,text=m[1].upper() if m else p['text']))
            vals['leading_choice_diagnostic']=score(rs,normalized)['overall'];table[a]=vals
        tables[name]=table
        if name in ['public74','open47','new50']:
            for d in scores['temporal_route']['details']:
                details.append(dict(**d,cohort=name,question=next(r['question'] for r in rs if r['id']==d['id']),has_cue=tmap[d['id']]['has_temporal_cue'],used_long_only=tmap[d['id']]['used_long_only'],answers={a:dict(text=pp[d['id'],a]['text'],correct=next(x['correct'] for x in scores[a]['details'] if x['id']==d['id'])) for a in ARMS}))
    stats=dict(triggered=sum(t['used_long_only'] for t in traces),cue_questions=sum(t['has_temporal_cue'] for t in traces),empty_old_fallbacks=sum(t['empty_long_fallback'] for t in traces),read_visual_tokens={a:dict(min=min(p['visual_tokens'] for p in ps if p['arm']==a),max=max(p['visual_tokens'] for p in ps if p['arm']==a),mean=float(np.mean([p['visual_tokens'] for p in ps if p['arm']==a]))) for a in ARMS})
    output=dict(status='complete_one_fixed_round',candidate='R0',tables=tables,routing=stats,M2_replay_IDs_equal=replay,routing_input_branch_matches=input_matches,routing_generation_branch_matches=branch_matches,execution=completed,limitations=['Already-open171 questions on67 media; no independent final holdout.','Single lexical gate does not understand which event before/after refers to.','Storage remains384 max; query arms have different actual read budgets.','Short temporal cues can require recent evidence; all losses retained.','Leading-option diagnostics distinguish format from semantic changes.','Only one candidate round, no rescue regex/prompt adjustment.'])
    save(OUT/'analysis.json',output);save(OUT/'per_question.json',details)
    print(json.dumps({c:{a:v['correct'] for a,v in t.items()} for c,t in tables.items()}));print(json.dumps(stats))
if __name__=='__main__':main()
