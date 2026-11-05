"""CPU-only tokenizer layouts and random tiny-model representation extraction."""
from pathlib import Path
import argparse
import hashlib
import os
import sys
from prepare import ROOT, INPUT, read, save, digest
from feature_runtime import inputs, extract

def run(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':
        raise RuntimeError('CPU-only explicit CUDA disable required')
    sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
    from sember_common import runtime
    env=runtime();torch=env['torch'];torch.set_num_threads(1)
    from sember_nodes import pair_node
    from sember_runtime import make_builder
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor)
    class Proxy:
        def __init__(self):self.embedding=torch.nn.Embedding(len(processor.tokenizer),8)
        def get_input_embeddings(self):return self.embedding
    proxy=Proxy();layouts=[]
    for q in read(INPUT/'requests.json'):
        frames=q['recent_frames']
        nodes=[pair_node(torch.zeros(108,8),q['committed_pairs']-4+i,
                frames[2*i]['pts_s'],frames[2*i+1]['pts_s'],(9,12)) for i in range(4)]
        for evidence in (True,False):
            emb,pos,ids,meta=inputs(proxy,builder,nodes,q['question'],evidence)
            assert meta['visual_tokens']==(432 if evidence else 0)
            assert ids.shape[1]+8<=2048 and pos.shape==(4,1,ids.shape[1])
            layouts.append(dict(id=q['id'],evidence=evidence,**meta,
                ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
                positions_sha256=hashlib.sha256(pos.numpy().tobytes()).hexdigest()))
    del proxy
    torch.manual_seed(20261028)
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](processor),
        attn_implementation='sdpa')).float().eval()
    width=model.get_input_embeddings().embedding_dim
    nodes=[pair_node(torch.randn(108,width),i,2.*i,2.*i+1,(9,12)) for i in range(4)]
    calls=dict(vision=0,lm_head=0)
    def hook(name):
        def call(*args):calls[name]+=1
        return call
    vh=model.model.visual.register_forward_pre_hook(hook('vision'))
    lh=model.lm_head.register_forward_pre_hook(hook('lm_head'))
    old_hooks=len(model.model.language_model.norm._forward_hooks)
    actual={};maxerr=0.
    for evidence in (True,False):
        f,meta=extract(env,model,builder,nodes,'Which object is visible?',evidence)
        repeated,_=extract(env,model,builder,nodes,'Which object is visible?',evidence)
        assert torch.equal(f,repeated)
        with torch.inference_mode(),env['fp32_reference']():
            emb,pos,ids,_=inputs(model,builder,nodes,'Which object is visible?',evidence)
            native=model.model.language_model(inputs_embeds=emb,position_ids=pos,use_cache=False).last_hidden_state[0,-1].cpu()
        error=float((native-f).abs().max());maxerr=max(maxerr,error)
        assert torch.allclose(native,f,rtol=1e-5,atol=1e-6)
        actual[evidence]=f
    changed=[n.copy() for n in nodes];changed[-1].features.add_(2)
    perturbed,_=extract(env,model,builder,changed,'Which object is visible?',True)
    no_video,_=extract(env,model,builder,changed,'Which object is visible?',False)
    assert not torch.equal(perturbed,actual[True]) and torch.equal(no_video,actual[False])
    assert len(model.model.language_model.norm._forward_hooks)==old_hooks
    assert model.model.rope_deltas is None and calls==dict(vision=0,lm_head=0)
    vh.remove();lh.remove()
    receipt=dict(status='passed_CPU_layout_and_tiny_representation', layouts=layouts,layout_count=len(layouts),
        max_input_tokens=max(l['input_tokens'] for l in layouts),
        random_tiny_width=width, tiny_extractions=6, tiny_direct_reference_prefills=2,
        maximum_continued_direct_hidden_error=maxerr,vision_forwards=0,lm_head_forwards=0,
        repeat_exact=True,visual_perturbation_changes_video_feature=True,no_video_invariant=True,
        hooks_removed=True,rope_restored=True,visual_bank_unchanged=True,
        real_model_loaded=False,real_features=False,real_head_fit=False,GPU_jobs=0,
        source_sha256={p.name:digest(p) for p in sorted(Path(__file__).resolve().parent.glob('*.py'))})
    save(output,receipt)
    print({k:v for k,v in receipt.items() if k not in ('layouts','source_sha256')},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
