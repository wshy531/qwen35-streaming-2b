"""Frozen H1 comparisons. Local gold only; never imported by GPU execution."""
import argparse, json, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def read(p): return json.loads(p.read_text(encoding='utf8'))
def rows(p): return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main(out, partial=False):
    prep=ROOT/'runs/7b_head_simple_prepare_20261112_01'
    plan=read(prep/'plan.json'); tt=read(prep/'tasks.json'); tasks={t['question']['id']:t for t in tt}
    gold=read(ROOT/'runs/7b_method_transfer_prepare_20261110_01/evaluation_keys_LOCAL_ONLY.json')
    gold.update(read(ROOT/'runs/7b_selector_short_prepare_20261110_01/keys_LOCAL_ONLY.json'))
    base={}
    for rel,h in plan['source_files'].items():
        assert sha(ROOT/rel)==h
        for r in rows(ROOT/rel):
            k=r['id'],r['route'],r['shift']
            if k in base:
                assert base[k]['selected_original']==r['selected_original']
            else:base[k]=r
    incoming=rows(out/'answers.jsonl');seen=set();unique_new=set();unique_reused=set()
    for r in incoming:
        k=r['id'],r['route'],r['shift']; assert k not in seen;seen.add(k)
        t=tasks[r['id']];assert r['choice']==t['methods'][r['route']]
        assert r['shift'] in (0,2) and r['suite']==t['question']['suite']
        nc=len(t['question']['options']);v=np.asarray(r['option_logits'])
        assert v.shape==(nc,) and np.isfinite(v).all()
        assert (int(v.argmax())+r['shift'])%nc==r['selected_original']
        unique=(r['id'],tuple(r['choice']),r['shift'])
        if r['provenance']=='new': unique_new.add(unique)
        else:
            assert r['provenance']=='reused' and plan['source_files'][r['source']]==r['source_sha']
            cached=[e['row'] for e in t['cached'] if e['choice']==r['choice'] and e['row']['shift']==r['shift']]
            assert len(cached)==1
            for f in ('selected_original','option_logits','visual_tokens'): assert cached[0][f]==r[f]
            unique_reused.add(unique)
        base[k]=r
    controls=list(tt[0]['methods'])
    complete=[i for i in tasks if all((i,a,s) in seen for a in controls for s in (0,2))]
    if not partial:
        done=read(out/'completed.json')
        assert len(complete)==done['questions']==done['states']==210
        assert done['classifications']==294 and done['references']==12 and done['reused']==700
        assert len(incoming)==2520 and len(unique_new)==282 and len(unique_reused)==700
        checks=rows(out/'checks.jsonl'); assert len(checks)==12
        for r in checks:
            ref=tasks[r['id']]['reference']['row']
            assert r['selected_original']==ref['selected_original'] and r['visual_tokens']==ref['visual_tokens']
            np.testing.assert_allclose(r['option_logits'],ref['option_logits'],atol=1e-5,rtol=1e-5)
    arms=['recent8','LS','head','adapted','control','small_step']+controls
    tables=[];paired={}
    for suite in ('SB150','short60'):
        ids=[i for i in complete if tasks[i]['question']['suite']==suite]
        if not ids:continue
        for a in arms:
            rr=[base[i,a,s] for i in ids for s in (0,2)]
            tables.append(dict(suite=suite,method=a,questions=len(ids),correct=sum(r['selected_original']==gold[r['id']] for r in rr),
                               n=len(rr),mean_tokens=float(np.mean([r['visual_tokens'] for r in rr]))))
        for a,bs in [('control',['LS','recent8','first','last','control_first','control_last']),
                     ('small_step',['LS','recent8','first','last','small_first','small_last'])]:
            for b in bs:
                g=l=0;cluster=defaultdict(list)
                for i in ids:
                    d=0
                    for s in (0,2):
                        ac=base[i,a,s]['selected_original']==gold[i];bc=base[i,b,s]['selected_original']==gold[i]
                        g+=ac and not bc;l+=bc and not ac;d+=int(ac)-int(bc)
                    cluster[tasks[i]['question']['source_id']].append(d)
                z=np.array([(sum(v),len(v)*2) for v in cluster.values()]);rng=np.random.default_rng(711112)
                boot=z[rng.integers(len(z),size=(10000,len(z)))].sum(1)
                paired[f'{suite}:{a}_vs_{b}']=dict(gains=int(g),losses=int(l),net=int(g-l),
                    source_bootstrap_95=np.quantile(boot[:,0]/boot[:,1],[.025,.975]).tolist())
    result=dict(partial=partial,questions=len(complete),tables=tables,paired=paired,
        unique_new_scores=len(unique_new),unique_reused_scores=len(unique_reused),
        limits=['opened regression sets','two dependent option rotations','all methods use content70',
                'same-count controls inherit head decision count','no independent timing claim'])
    (out/('partial_analysis.json' if partial else 'analysis.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(questions=len(complete),tables=tables)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--partial',action='store_true')
    a=p.parse_args();main(a.out,a.partial)
