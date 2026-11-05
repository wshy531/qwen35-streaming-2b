"""History-value intervention and reference chunking control on a tiny model."""
import os,sys,json,hashlib
from pathlib import Path
from dataclasses import replace
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/2b-prune-timing-audit-v1'))
from audit_common import runtime,make_builder
import protected_readout as p
from quality_banks import selected_node
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261103)
processor=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
cfg=env['tiny_config'](processor)
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
builder=make_builder(env,processor);width=model.get_input_embeddings().embedding_dim
nodes=[selected_node(t.randn(108,width),t.arange(108),i,float(2*i),float(2*i+1)) for i in range(4)]
question='What is the person holding now? A. A cup. B. A bag.'
fingerprints=[];old_compose=p.compose
def traced(hist,recent,prefix):
    h=hashlib.sha256();counts={}
    for field in ('key_cache','value_cache','conv_states','recurrent_states'):
        values=[x for x in getattr(recent.cache,field) if x is not None];counts[field]=len(values)
        assert values,field
        for x in values:h.update(x.cpu().numpy().tobytes())
    fingerprints.append((h.hexdigest(),counts))
    return old_compose(hist,recent,prefix)
p.compose=traced
try:
    a=p.answer(env,model,builder,nodes[:2],nodes[2:],question,limit=2)
    swapped=[replace(n,features=-n.features) for n in nodes[:2]]
    b=p.answer(env,model,builder,swapped,nodes[2:],question,limit=2)
finally:p.compose=old_compose
assert fingerprints[0]==fingerprints[1]
for key in ('input_ids','position_ids'):assert t.equal(a[key],b[key])
joint=p.answer(env,model,builder,nodes[:2],nodes[2:],question,isolated=False,limit=2)
saved=model.model.rope_deltas;branch=env['HybridSequence'](model,max_tokens=4096)
try:
    with t.inference_mode(),env['fp32_reference']():
        emb,pos,ids,meta=p.context(model,builder,nodes,question,max_new_tokens=2)
        for lo in range(0,emb.shape[1],512):
            hi=min(lo+512,emb.shape[1]);logits=branch.append(inputs_embeds=emb[:,lo:hi],position_ids=pos[:,:,lo:hi],logits_mode='last' if hi==emb.shape[1] else 'none')
        delta=float((logits.cpu()-joint['first_logits']).abs().max())
        assert t.allclose(logits.cpu(),joint['first_logits'],atol=1e-6,rtol=1e-5)
        assert logits.argmax(-1).item()==joint['first_logits'].argmax(-1).item()
finally:branch.cache.reset();model.model.rope_deltas=saved
out=R/'runs/isolated_recent_checks_20261103_01';out.mkdir(exist_ok=False)
(out/'CPU_local.json').write_text(json.dumps(dict(status='passed',history_replacement_recent_state_exact=True,
    populated_mixed_layers=fingerprints[0][1],joint_reference_max_abs_error=delta,joint_first_choice_same=True,
    tiny_QA=3,tiny_reference_prefill=1,GPU=0,pretrained_loads=0,does_not_prove_answer_protection=True),indent=2))
print('History swap preserves recent KV/conv/recurrent; reference chunking passes',delta)
