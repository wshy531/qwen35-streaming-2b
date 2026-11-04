import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_scale_prepare_20261102_01'
sys.path.insert(0,str(R/'experiments/2b-evidence-eval-v1'))
import gpu_profile as s
from response import question
from response_v4 import parse
sys.path.insert(0,str(R/'experiments/2b-hard-mask-prune-v1'))
from hard_prune import GroupScorer,encode as hard_encode
sys.path.insert(0,str(R/'experiments/streamingbench-interface-v1'))
from source import CausalPairs
ARMS=('full','fixed12','heuristic11','mask_ce','text_only')
CHECKPOINT=R/'runs/hard-mask-main-220062/mask_ce_step1237.pt'
CHECKPOINT_SHA='de8ed891233a466ec087b90412b2e92eea07bb0b79e1e999cf7e9167e620f9ff'
read=s.read;save=s.save;sha=s.sha
def encode(model,batch,arm,head,state,selector):
    if arm=='mask_ce':
        features,ids,_,state=hard_encode(model.model.visual,batch.pixel_values_videos,batch.video_grid_thw,head,previous=state,mode='gather')
        return features,ids,state
    return s.encode_pair(model,batch,arm,None,None,selector)
def layout(env,model,builder,records,q,arm):
    prompt=question(q,0,'B')
    if arm!='text_only':return s.context(model,builder,records,prompt)
    t=env['torch'];ids=t.cat((builder.prefix_ids,builder.question_ids(prompt)),1)
    assert not (ids==builder.processor.video_token_id).any()
    assert not (ids==builder.processor.tokenizer.convert_tokens_to_ids(builder.processor.vision_start_token)).any()
    device=model.get_input_embeddings().weight.device
    pos=t.arange(ids.shape[1]).expand(4,-1)[:,None].to(device)
    return model.get_input_embeddings()(ids.to(device)),pos,ids
def generate(env,model,builder,records,q,arm,limit=128):
    emb,pos,ids=layout(env,model,builder,records,q,arm)
    assert ids.shape[1]+limit<=4096
    branch=env['HybridSequence'](model,max_tokens=4096,session_id='evidence-scale')
    saved=model.model.rope_deltas
    try:
        for start in range(0,emb.shape[1],512):
            end=min(start+512,emb.shape[1])
            logits=branch.append(inputs_embeds=emb[:,start:end],position_ids=pos[:,:,start:end],logits_mode='last' if end==emb.shape[1] else 'none')
        eos=model.generation_config.eos_token_id
        stops=set(eos if isinstance(eos,list) else [] if eos is None else [eos]);stops.add(builder.processor.tokenizer.eos_token_id)
        generated,reason=env['greedy_decode'](branch,logits,limit,stops)
    finally:branch.cache.reset();model.model.rope_deltas=saved
    text=builder.processor.tokenizer.decode(generated,skip_special_tokens=True)
    return dict(text=text,generated_ids=generated,reason=reason,input_tokens=ids.shape[1],
        input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
        position_ids_sha256=hashlib.sha256(pos.cpu().numpy().tobytes()).hexdigest(),
        visual_tokens=sum(len(r[3]) for r in records) if arm!='text_only' else 0,**parse(text,len(q['options']),'B'))
