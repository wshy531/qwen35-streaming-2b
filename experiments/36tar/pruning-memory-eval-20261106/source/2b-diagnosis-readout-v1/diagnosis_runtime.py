"""Use the existing full-vision and hybrid readout, without any learned head."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/diagnosis_readout_prepare_20261105_01'
for name in ('streamingbench-router-v1','2b-midprune-memory-v1'):
    sys.path.insert(0,str(ROOT/'experiments'/name))
from v0_source import V0Pairs
from source import Packet,canvas
from pilot import runtime,make_builder,encode_pair,CoverageScorer,sparse_pair,context,digest,torch
sys.path.insert(0,str(ROOT/'experiments/2b-evidence-eval-v1'))
from response import question
from response_v4 import parse

ARMS=('recent','early','combined','text')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def choose(source,nodes,arm):
    ids=source['recent_pair_indices'] if arm=='recent' else source['early_pair_indices'] if arm=='early' else source['selected_pair_indices'] if arm=='combined' else []
    return [nodes[i] for i in ids]

def layout(model,builder,nodes,prompt,limit=128):
    if nodes:
        return context(model,builder,nodes,prompt,visual_limit=1728,context_limit=4096,max_new_tokens=min(limit,16))
    ids=torch.cat((builder.prefix_ids,builder.question_ids(prompt)),1)
    assert not bool((ids==builder.processor.video_token_id).any())
    device=model.get_input_embeddings().weight.device
    pos=torch.arange(ids.shape[1]).expand(4,1,-1).clone().to(device)
    return model.get_input_embeddings()(ids.to(device)),pos,ids,dict(visual_tokens=0,nodes=[])

@torch.no_grad()
def answer(env,model,builder,nodes,prompt,limit=128):
    before=digest(nodes);saved=model.model.rope_deltas;branch=None
    try:
        emb,pos,ids,meta=layout(model,builder,nodes,prompt,limit)
        assert ids.shape[1]+limit<=4096
        branch=env['HybridSequence'](model,max_tokens=4096,session_id='diagnosis-readout')
        for begin in range(0,emb.shape[1],512):
            end=min(begin+512,emb.shape[1])
            logits=branch.append(inputs_embeds=emb[:,begin:end],position_ids=pos[:,:,begin:end],logits_mode='last' if end==emb.shape[1] else 'none')
        eos=model.generation_config.eos_token_id
        stops=set(eos if isinstance(eos,list) else [] if eos is None else [eos]);stops.add(builder.processor.tokenizer.eos_token_id)
        generated,reason=env['greedy_decode'](branch,logits,limit,stops)
    finally:
        if branch is not None:branch.cache.reset()
        model.model.rope_deltas=saved
    assert digest(nodes)==before
    meta=dict(meta,input_tokens=int(ids.shape[1]))
    return dict(text=builder.processor.tokenizer.decode(generated,skip_special_tokens=True),generated_ids=generated,
        reason=reason,branch_released=True,read_state_unchanged=True,
        input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
        position_ids_sha256=hashlib.sha256(pos.cpu().numpy().tobytes()).hexdigest(),**meta)

def selected_pairs(env,path,source):
    """Seek exact frozen PTS/RGB, retaining only one contiguous evidence window."""
    import itertools
    wanted=source['selected_pair_indices']
    chunks=[]
    for _,group in itertools.groupby(enumerate(wanted),lambda item:item[1]-item[0]):
        chunks.append([value for _,value in group])
    with env['av'].open(str(path)) as container:
        video=container.streams.video[0];video.codec_context.thread_count=1
        first=next(container.decode(video));origin=float(first.time)
        for indices in chunks:
            targets=[p for i in indices for p in source['pairs'][i]]
            lo,hi=targets[0]['pts'],targets[-1]['pts']
            container.seek(int((origin+max(0.,lo-1))/video.time_base),stream=video,backward=True)
            found=[];cursor=0
            for frame in container.decode(video):
                if frame.time is None:continue
                pts=float(frame.time)-origin
                if pts<targets[cursor]['pts']-1e-7:continue
                assert abs(pts-targets[cursor]['pts'])<1e-7,(source['source_id'],pts,targets[cursor]['pts'])
                rgb=canvas(frame.to_ndarray(format='rgb24'))
                assert hashlib.sha256(rgb.tobytes()).hexdigest()==targets[cursor]['rgb_sha256']
                found.append(Packet(targets[cursor]['frame_id'],pts,rgb));cursor+=1
                if cursor==len(targets):break
            assert cursor==len(targets)
            for j,index in enumerate(indices):yield index,found[2*j],found[2*j+1]
