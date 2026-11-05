"""Labels stay local. Prespecified strict/format and paired source analysis."""
import csv,hashlib,json,re,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/2b_natural_eval_20261020_01'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from score_memory_eval import score,paired
ARMS=['native_original','full_pairwise_reference','M0','M1','M2','text_only']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows():
    groups=[('public74','runs/natural_eval_v2_20261019_01/admitted74.json','runs/natural_eval_v2_20261019_01/strata74.json'),('open47','runs/holdout_qualification_20261019_01/natural_core.json','runs/holdout_qualification_20261019_01/strata.json'),('new50','runs/08b_question_readout_new24_20261019_01/scorer_rows.json',None)]
    result=[]
    for cohort,path,strata in groups:
        ss={r['id']:r for r in read(ROOT/strata)} if strata else None
        for r in read(ROOT/path):
            r=dict(r);r['cohort']=cohort;r['stratum']=ss[r['id']]['stratum'] if ss else r['curation']['stratum']
            r['source_group_id']=r.get('analysis_group',r['source_group_id']);result.append(r)
    if len(result)!=171 or len({r['id'] for r in result})!=171:raise ValueError('171 unique scoring IDs')
    inputs={r['id']:r for r in read(RUN/'requests.json')}
    for r in result:
        if r['question']!=inputs[r['id']]['model_visible']['question'] or r['options']!=inputs[r['id']]['model_visible']['options']:raise ValueError('scoring/input mismatch')
    return result
def load_predictions(base):
    records=[]
    for stage in ['native','memory']:
        path=base/stage/'predictions.jsonl'
        records.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
    if len(records)!=1026 or len({(r['id'],r['arm']) for r in records})!=1026:raise ValueError('all1026 distinct predictions required')
    if {r['arm'] for r in records}!=set(ARMS):raise ValueError('wrong arms')
    return records
def table(rows,predictions):
    scores={a:score(rows,[p for p in predictions if p['arm']==a and p['id'] in {r['id'] for r in rows}]) for a in ARMS}
    groups=sorted({r['source_group_id'] for r in rows});rng=np.random.default_rng(20261020);draw=rng.integers(0,len(groups),(2000,len(groups)))
    matrix=np.array([[np.mean([int(d['correct']) for d in scores[a]['details'] if d['video']==g]) for a in ARMS] for g in groups]);boot=matrix[draw].mean(1)
    result={a:dict(**scores[a]['overall'],macro_interval95=np.percentile(boot[:,j],[2.5,97.5]).tolist()) for j,a in enumerate(ARMS)}
    for a in ARMS:
        j=ARMS.index(a)
        for ref in ['M0','full_pairwise_reference','text_only']:
            k=ARMS.index(ref);result[a]['vs_'+ref]=dict(**paired(scores[ref],scores[a]),macro_delta=float((matrix[:,j]-matrix[:,k]).mean()),macro_delta_interval95=np.percentile(boot[:,j]-boot[:,k],[2.5,97.5]).tolist())
    leading={}
    for a in ARMS:
        ps=[]
        for p in predictions:
            if p['arm']!=a:continue
            m=re.match(r'^\s*([A-Ea-e])(?:\b|[.!)\s])',p.get('text',''))
            ps.append(dict(p,text=m[1].upper() if m else p.get('text','')))
        leading[a]=score(rows,[p for p in ps if p['id'] in {r['id'] for r in rows}])['overall']
    return dict(strict=result,leading_choice_diagnostic=leading),scores
def prior():
    mapping=[('public74','runs/08b_frozen74_compare_20261019_01'),('open47','runs/08b_holdout47_compare_20261019_01'),('new50','runs/08b_question_readout_new24_20261019_01')]
    result=[]
    for cohort,path in mapping:
        for stage in ['native','memory']:
            p=ROOT/path/stage/'predictions.jsonl'
            for line in p.read_text(encoding='utf-8').splitlines():
                r=json.loads(line);a={'full_pairwise':'full_pairwise_reference','M1_all8':'M1'}.get(r['arm'],r['arm'])
                if a in ARMS:result.append(dict(r,arm=a,cohort=cohort))
    return result
def main():
    data=rows();save(RUN/'scorer_rows_local.json',data)
    if len(sys.argv)==2 and sys.argv[1]=='freeze':
        save(RUN/'analysis_lock.json',dict(source_sha256=sha(__file__),scorer_sha256=sha(RUN/'scorer_rows_local.json'),questions=171,FT=118,C=53,strict_primary=True,leading_choice_only_format_diagnostic=True,seed=20261020,bootstrap_replicates=2000,unit='conservative source group',analysis='cohorts separately; FT primary and C secondary; paired corrections/loss vs recent/full/text; no significance-driven extra runs; fixed 0.8B cross-model matched arms only'))
        print('analysis frozen before GPU outputs');return
    lock=read(RUN/'analysis_lock.json')
    if sha(__file__)!=lock['source_sha256'] or sha(RUN/'scorer_rows_local.json')!=lock['scorer_sha256']:raise ValueError('analysis drift')
    base=RUN/sys.argv[1];pred=load_predictions(base);tables={};cross={};old=prior()
    subsets={c:[r for r in data if r['cohort']==c] for c in ['public74','open47','new50']}
    for c,rs in list(subsets.items()):
        subsets[c+'_FT']=[r for r in rs if r['stratum'] in ['F','T']];subsets[c+'_C']=[r for r in rs if r['stratum']=='C']
    details=[]
    for name,rs in subsets.items():
        tables[name],scored=table(rs,pred)
        for a in ARMS:
            op=[p for p in old if p['arm']==a and p['id'] in {r['id'] for r in rs}]
            if len(op)==len(rs):
                oscore=score(rs,op);cross.setdefault(name,{})[a]=dict(previous_08B_correct=oscore['overall']['correct'],current_2B_correct=scored[a]['overall']['correct'],paired_08B_to_2B=paired(oscore,scored[a]))
        if name in ['public74','open47','new50']:
            for a in ARMS:
                ps={p['id']:p for p in pred if p['arm']==a}
                for d in scored[a]['details']:details.append(dict(cohort=name,arm=a,text=ps[d['id']]['text'],**d))
    perarm={a:[p for p in pred if p['arm']==a] for a in ARMS}
    native={p['id']:p for p in perarm['native_original']};full={p['id']:p for p in perarm['full_pairwise_reference']}
    equivalence=dict(cases=171,actual_input_frame_hashes_same=sum(native[k]['input_frames_sha256']==full[k]['input_frames_sha256'] for k in native),generated_IDs_same=sum(native[k].get('generated_ids')==full[k].get('generated_ids') for k in native),scope='same actual frames, distinct whole/pairwise vision protocol; equality is observed output agreement only')
    timelines=read(base/'memory/timelines.json');capacity={a:dict(max_tokens=max(t['states'][a]['tokens'] for t in timelines),max_tensor_bytes=max(t['states'][a]['candidate_tensor_bytes'] for t in timelines)) for a in ['M0','M1','M2']}
    result=dict(tables=tables,cross_model=cross,input_output_agreement=equivalence,capacity=capacity,execution={s:read(base/s/'completed.json') for s in ['native','memory']},records=len(pred),limitations=['All cohorts opened previously for0.8B; not a fresh final holdout.','Low-resolution fixed causal sample, 8 answer-token cap; strict and leading-format diagnostics separate.','Single-assistant content qualification, conservative related-household grouping; not independently adjudicated.','Author evidence interval coverage is not necessary-evidence proof.','Fixed animal-calibrated forest threshold transferred without tuning; merged positions heuristic.','External memory snapshot questions; no new online reset or wall-clock claim.','Candidate384Token is storage budget; full-reference cache and model weights are additional.','Timing descriptive, not warmed/interleaved speed benchmark.'])
    save(RUN/'analysis.json',result);save(RUN/'scored_details.json',details)
    with (RUN/'per_question.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    print(json.dumps({k:{a:v['correct'] for a,v in x['strict'].items()} for k,x in tables.items()}))
if __name__=='__main__':main()
