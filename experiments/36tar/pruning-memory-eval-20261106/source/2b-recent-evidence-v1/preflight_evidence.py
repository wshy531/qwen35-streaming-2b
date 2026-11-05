"""CPU contracts, complete layouts and shared random tiny-model forwards."""
import argparse
from collections import Counter
import copy
import os
from pathlib import Path
import sys
from gate_common import ROOT,HERE,INPUTS,read,save,sha,shared
from evidence_policy import render,aggregate
from evidence_runtime import inputs,score,tensor_sha

def main(output,tiny):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':raise RuntimeError('CPU-only preflight required')
    env=shared();torch=env['torch'];torch.set_num_threads(1)
    from sember_runtime import make_builder
    from sember_nodes import pair_node,digest
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor)
    # Embedding-only proxy: no real model weights and no language forward for full layouts.
    class Proxy:
        def __init__(self):self.layer=torch.nn.Embedding(len(processor.tokenizer),8)
        def get_input_embeddings(self):return self.layer
    proxy=Proxy();requests=read(INPUTS/'requests.json');sources=read(INPUTS/'sources.json')
    assert len(requests)==120 and len(sources)==18 and len({q['question'] for q in requests})==113
    allowed={'id','question','options','source_name','query_time_s','committed_pairs','recent_frames'}
    assert all(set(q)==allowed for q in requests)
    seen={s:Counter() for s in ('LOCAL','HISTORY')}
    for view in (0,1):
        p=render('What now?',['a','b','c','d','e'],view)
        for code,s in zip(p['codes'],p['semantic_order']):seen[s][code]+=1
    assert all(v=={'A':1,'B':1} for v in seen.values())
    for target,expected in [('LOCAL','recent'),('HISTORY','full')]:
        vs=[dict(render('What now?',['a','b','c','d','e'],v),candidate_logits=[3. if s==target else 0. for s in render('What now?',['a','b','c','d','e'],v)['semantic_order']]) for v in (0,1)]
        assert aggregate(vs)['route']==expected
    ties=[dict(render('What now?',['a','b','c','d','e'],v),candidate_logits=[0.,0.]) for v in (0,1)]
    assert aggregate(ties)['route']=='full'
    try:aggregate(ties[:1])
    except ValueError:pass
    else:raise AssertionError('incomplete views accepted')
    layouts=[]
    for q in requests:
        nodes=[pair_node(torch.zeros(108,8),q['committed_pairs']-4+i,
            q['recent_frames'][2*i]['pts_s'],q['recent_frames'][2*i+1]['pts_s'],(9,12)) for i in range(4)]
        before=digest(nodes)
        for shift in range(5):
            options=q['options'][shift:]+q['options'][:shift]
            for view in (0,1):
                p=render(q['question'],options,view)
                poisoned=dict(q,correct_index=4,prediction='DO_NOT_RENDER',role='old')
                assert render(poisoned['question'],options,view)==p
                for mode,evidence in (('recent_video',True),('no_video',False)):
                    emb,pos,ids,codes,meta=inputs(proxy,builder,nodes,p,evidence)
                    assert meta['visual_tokens']==(432 if evidence else 0)
                    assert meta['input_tokens']+8<=2048 and emb.shape[1]==ids.shape[1]==pos.shape[2]
                    layouts.append(dict(id=q['id'],shift=shift,view=view,mode=mode,codes=codes,
                        input_ids_sha256=tensor_sha(ids),position_ids_sha256=tensor_sha(pos),input_tokens=ids.shape[1]))
        assert digest(nodes)==before
    assert len(layouts)==2400
    # No-label / frozen-answer files are never imported by worker or runtime.
    import evidence_worker
    from unittest.mock import patch
    with patch.dict(os.environ,{'SLURM_JOB_ID':'','SLURM_CPUS_PER_TASK':'4'}):
        try:evidence_worker.main(ROOT/'runs/CPU_forbidden_evidence_GPU',Path('absent'))
        except RuntimeError as e:assert 'Slurm' in str(e)
        else:raise AssertionError('non-Slurm GPU entry accepted')
    from evidence_analysis import baselines,decision,evaluate
    base=baselines()
    assert not decision(base['full'],base['full'],base)['passed']
    raw=read(INPUTS/'frozen_branch_outputs_LOCAL_ONLY.json')
    oracle={(r['task_id'],r['shift']):('recent' if r['recent']['correct'] and not r['full']['correct'] else 'full') for r in raw}
    assert decision(evaluate(oracle),base['full'],base)['passed']
    actual=[]
    if tiny:
        del proxy
        torch.manual_seed(20261028)
        model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](processor),attn_implementation='sdpa')).float().eval()
        width=model.get_input_embeddings().embedding_dim
        nodes=[pair_node(torch.randn(108,width),i,2.*i,2.*i+1,(9,12)) for i in range(4)]
        q=requests[0];before=digest(nodes);calls=[0]
        hook=model.model.visual.register_forward_pre_hook(lambda *a:calls.__setitem__(0,calls[0]+1))
        for evidence in (True,False):
            for view in (0,1):
                r=score(env,model,builder,nodes,render(q['question'],q['options'],view),evidence)
                actual.append(dict(mode='recent_video' if evidence else 'no_video',**{k:v for k,v in r.items() if k!='seconds'}))
        for evidence,index in ((True,0),(False,2)):
            r=score(env,model,builder,nodes,render(q['question'],q['options'],0),evidence)
            assert r['candidate_logits']==actual[index]['candidate_logits']
        assert calls[0]==0 and digest(nodes)==before and model.model.rope_deltas is None
        # Visual perturbation changes embeddings only at visual slots; control input is identical.
        changed=[n.copy() for n in nodes];changed[0].features.add_(1)
        p=render(q['question'],q['options'],0)
        a=inputs(model,builder,nodes,p,True);b=inputs(model,builder,changed,p,True)
        assert torch.equal(a[1],b[1]) and torch.equal(a[2],b[2]) and not torch.equal(a[0],b[0])
        a=inputs(model,builder,nodes,p,False);b=inputs(model,builder,changed,p,False)
        assert torch.equal(a[0],b[0])
        hook.remove()
    result=dict(status='passed_CPU_recent_evidence',layout_count=2400,layouts=layouts,
        maximum_input_tokens=max(l['input_tokens'] for l in layouts),tiny_prefills=6 if tiny else 0,tiny_results=actual,
        checks=dict(balanced_codes=True,tie_full=True,incomplete_views_rejected=True,label_fields_not_rendered=True,
            recent_tokens_432=True,text_control_tokens_zero=True,recent_values_unchanged=True,
            GPU_entry_without_Slurm_rejected=True,gate_negative_positive_controls=True),
        GPU_jobs=0,pretrained_loads=0,real_QA_answers=0,capability_claim=False,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.sh')},
        input_sha256={p.name:sha(p) for p in INPUTS.iterdir() if p.is_file()})
    save(output,result)
    print({k:v for k,v in result.items() if k not in ('layouts','tiny_results','source_sha256','input_sha256')},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tiny',action='store_true')
    a=p.parse_args();main(a.output,a.tiny)
