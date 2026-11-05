"""Real tokenizer/context composition, synthetic features, zero real weights."""
import os,sys,hashlib,itertools
from pathlib import Path
from prepare import ROOT,PREP,read,save
from selection import choose

def main(output=None):
    output=output or PREP/'CPU_layout_preflight.json'
    if not output.resolve().is_relative_to(PREP.resolve()) or output.exists():
        raise ValueError('new output under owned preparation directory required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':raise RuntimeError('CPU only')
    sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
    sys.path.insert(0,str(ROOT/'experiments/2b-perception-transition-v1'))
    from sember_common import runtime
    from sember_runtime import make_builder,context
    from sember_nodes import Node,pair_node,merge,digest
    from canonical_prompt import question_prompt,rotate
    from evidence_selection_readout import compose
    env=runtime();torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(20261029)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor)
    class Proxy:
        def __init__(self):self.emb=torch.nn.Embedding(len(processor.tokenizer),8)
        def get_input_embeddings(self):return self.emb
    proxy=Proxy();tasks=read(PREP/'inputs/requests.json');layout=[]
    # Same synthetic bank for all questions; these are layout contracts, not their actual banks.
    h=[]
    for i in range(4):
        a=pair_node(torch.randn(108,8),2*i,4*i,4*i+1,(9,12))
        b=pair_node(torch.randn(108,8),2*i+1,4*i+2,4*i+3,(9,12))
        h.append(merge(a,b))
    r=[pair_node(torch.randn(108,8),8+i,16+2*i,17+2*i,(9,12)) for i in range(4)]
    nodes=h+r;before=digest(nodes)
    selections=[[],[0,1,2,3],[],[0],[2]]
    for t in tasks:
        for shift in range(5):
            q=question_prompt(rotate(t,shift)[0])
            full=context(proxy,builder,nodes,q,visual_limit=864,context_limit=8192,max_new_tokens=8)
            for arm,sel in zip(read(PREP/'protocol.json')['arms'],selections):
                emb,pos,ids,meta=compose(proxy,builder,nodes,sel,allow_full=arm=='full',question=q)
                cols=meta['full_context_columns']
                assert torch.equal(emb,full[0][:,cols]) and torch.equal(ids,full[2][:,cols])
                assert torch.equal(pos[1:],full[1][1:,:,cols])
                assert torch.equal(pos[0,0],torch.arange(len(cols)))
                if arm=='full':assert all(torch.equal(v,w) for v,w in zip((emb,pos,ids),full[:3]))
                assert meta['visual_tokens']<=864
                layout.append(dict(id=t['id'],shift=shift,arm=arm,tokens=len(cols),
                    input_sha=hashlib.sha256(ids.numpy().tobytes()).hexdigest()))
    assert len(layout)==1600 and digest(nodes)==before
    # One synthetic random tiny model, three selected contexts; repeat in reverse order to check isolation.
    del proxy
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](processor),attn_implementation='sdpa')).float().eval()
    width=model.get_input_embeddings().embedding_dim
    real_nodes=[pair_node(torch.randn(108,width),i,2*i,2*i+1,(9,12)) for i in range(8)]
    sums={};q=question_prompt(rotate(tasks[0],0)[0])
    with torch.inference_mode():
        for sel in ([],[0],[0,1], [0,1],[0],[]):
            emb,pos,ids,meta=compose(model,builder,real_nodes,sel,question=q)
            out=model.model.language_model(inputs_embeds=emb,position_ids=pos,use_cache=False,return_dict=True)
            v=out.last_hidden_state[:,-1].clone();key=str(sel)
            assert torch.isfinite(v).all()
            if key in sums:assert torch.equal(sums[key],v)
            sums[key]=v
    save(output,dict(status='passed',layouts=len(layout),
        actual_bank_layouts=False,synthetic_layouts_with_real_tokenizer=True,
        full_path_identity=True,retained_values_and_mrope_exact=True,whole_recent_preserved=True,
        tiny_random_prefills=6,real_pretrained_loads=0,real_QA_answers=0,GPU_jobs=0,
        maximum_input_tokens=max(r['tokens'] for r in layout),layout_rows=layout))
    print('Passed: 1600 synthetic layouts, full identity, retained values/positions, 6 isolated tiny prefills; zero real weights/GPU.')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    main(p.parse_args().output)
