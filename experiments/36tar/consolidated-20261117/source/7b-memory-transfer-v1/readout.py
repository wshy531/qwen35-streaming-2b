"""Qwen2.5 native language model + explicit visual-memory layout. No model patch."""
from pathlib import Path
import sys
import torch
import numpy as np
from q25_nodes import pair_node,digest
from q25_banks import Bank,distance

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-temporal-router-v1'))
from temporal_policy import lexical_route
sys.path.insert(0,str(ROOT/'experiments/2b-evidence-eval-v1'))
from response import question

GRID=(10,14)
TOKENS=140
ARMS=('recent8frames','recent16frames','complete_history','LS','lexical')
THRESHOLD=0.047536224126815796

class Memory:
    def __init__(self):
        self.ls=Bank('LS',TOKENS,budget=1120,recent=560,closed=420)
        self.recent=Bank('M0',TOKENS,budget=1120,recent=560,closed=420)
        self.previous=None
        # Only the complete-history reference retains this archive.
        # Shared benchmark allocation is never reported as bounded-memory cost.
        self.reference=[]

    def append(self,features,index,begin,end):
        node=pair_node(features,index,begin,end,GRID)
        desc=node.features.mean(0)
        event=self.previous is None or distance(self.previous,desc)>THRESHOLD
        self.ls.append(node,starts_event=event)
        self.recent.append(node,starts_event=event)
        self.reference.append(node)
        self.previous=desc.clone()

    def select(self,arm,q):
        if arm=='recent8frames':return [e.node for e in self.ls.recent]
        if arm=='recent16frames':return self.recent.nodes()
        if arm=='complete_history':return self.reference
        if arm=='LS':return self.ls.nodes()
        if arm=='lexical':return self.select('LS' if lexical_route(q['question'])['route']=='full' else 'recent8frames',q)
        raise ValueError(arm)

def template(processor,prompt):
    return processor.apply_chat_template([{'role':'user','content':[
        {'type':'video'},{'type':'text','text':prompt}]}],tokenize=False,add_generation_prompt=True)

def native_inputs(processor,frames,prompt):
    metadata=dict(total_num_frames=len(frames),fps=1.0,frames_indices=list(range(len(frames))))
    batch=processor(text=[template(processor,prompt)],videos=[np.stack(frames)],
        video_metadata=[metadata],do_resize=False,do_sample_frames=False,return_tensors='pt')
    assert batch['second_per_grid_ts'].tolist()==[2.0]
    return batch

def pair_inputs(processor,frames,device='cpu'):
    assert len(frames)==2
    batch=processor.video_processor(videos=[torch.from_numpy(np.stack(frames)).permute(0,3,1,2)],
        do_resize=False,do_sample_frames=False,return_tensors='pt')
    assert batch['video_grid_thw'].tolist()==[[1,20,28]]
    return {key:batch[key].to(device) for key in ('pixel_values_videos','video_grid_thw')}

def layout(processor,nodes,prompt):
    assert nodes and all(n.grid==GRID for n in nodes)
    coords=torch.cat([n.coords for n in nodes]); count=len(coords)
    text=template(processor,prompt)
    assert text.count(processor.video_token)==1
    ids=processor.tokenizer(text.replace(processor.video_token,processor.video_token*count),
        return_tensors='pt')['input_ids']
    matches=torch.where(ids[0]==processor.video_token_id)[0]
    assert len(matches)==count and torch.equal(matches,torch.arange(int(matches[0]),int(matches[0])+count))
    start=int(matches[0]);stop=start+count
    # Retain surviving source time and spatial coordinates. The earliest original
    # source pair represented by these nodes anchors the selected clip's clock.
    origin=min(n.first for n in nodes)
    visual=coords.T.clone()
    visual[0]=(visual[0]-origin)*4  # 2 seconds per frame pair * 2 position units/sec
    visual+=start
    pos=torch.cat((torch.arange(start).expand(3,-1),visual,
        (torch.arange(ids.shape[1]-stop)+int(visual.max())+1).expand(3,-1)),dim=1).unsqueeze(1)
    assert ids.shape[1]+128<=128000 and int(pos.max())+128<128000
    return ids,pos,matches

def context(model,processor,nodes,prompt):
    ids,pos,matches=layout(processor,nodes,prompt)
    device=model.get_input_embeddings().weight.device
    ids=ids.to(device);pos=pos.to(device)
    embeddings=model.get_input_embeddings()(ids)
    visual=torch.cat([n.features for n in nodes]).to(device=device,dtype=embeddings.dtype)
    assert visual.shape==(len(matches),model.config.text_config.hidden_size)
    embeddings[:,matches.to(device)]=visual
    return embeddings,pos,ids

@torch.no_grad()
def generate_context(model,embeddings,positions,*,limit=128,guard=lambda:None):
    device=embeddings.device;cache=None;logits=None
    try:
        for start in range(0,embeddings.shape[1],512):
            guard();end=min(start+512,embeddings.shape[1])
            out=model.model.language_model(inputs_embeds=embeddings[:,start:end],
                position_ids=positions[:,:,start:end],attention_mask=torch.ones((1,end),device=device,dtype=torch.long),
                past_key_values=cache,use_cache=True)
            cache=out.past_key_values
            if end==embeddings.shape[1]:logits=model.lm_head(out.last_hidden_state[:,-1:])
        first_logits=logits[0,-1].float().cpu().clone()
        generated=[];eos=model.generation_config.eos_token_id
        stops=set(eos if isinstance(eos,list) else [eos])
        next_position=int(positions.max())+1
        for step in range(limit):
            guard();token=int(logits[0,-1].argmax());generated.append(token)
            if token in stops:break
            if step==limit-1:break
            token_tensor=torch.tensor([[token]],device=device)
            out=model.model.language_model(input_ids=token_tensor,
                position_ids=torch.full((3,1,1),next_position+step,device=device,dtype=torch.long),
                attention_mask=torch.ones((1,embeddings.shape[1]+step+1),device=device,dtype=torch.long),
                past_key_values=cache,use_cache=True)
            cache=out.past_key_values;logits=model.lm_head(out.last_hidden_state[:,-1:])
        return generated,first_logits
    finally:
        del cache

@torch.no_grad()
def answer(model,processor,memory,q,arm,guard,limit=128):
    nodes=memory.select(arm,q);before=digest(nodes)
    emb,pos,ids=context(model,processor,nodes,question(q,0,'B'))
    generated,_=generate_context(model,emb,pos,limit=limit,guard=guard)
    assert digest(nodes)==before
    return dict(text=processor.tokenizer.decode(generated,skip_special_tokens=True),
        generated_ids=generated,input_tokens=ids.numel(),visual_tokens=sum(len(n.features) for n in nodes),
        nodes=[n.metadata() for n in nodes],read_state_unchanged=True,
        lexical_route=None if arm!='lexical' else lexical_route(q['question']))
