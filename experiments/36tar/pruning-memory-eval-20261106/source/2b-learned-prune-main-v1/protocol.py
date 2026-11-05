"""Fixed one-epoch pilot; no resource allocation on import."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/learned_prune_main_prepare_20261101_01'
DATA=ROOT/'runs/learned_prune_data_prepare_20261101_01'
for name in ['2b-content-midprune-v1','2b-learned-prune-profile-v1','2b-learned-prune-mechanism-v1']:
    sys.path.insert(0,str(ROOT/'experiments'/name))
from gpu_preflight import runtime,make_builder,decode,sha
from training import context,task_loss,question_text
from bias_prune import GroupScorer,encode as learned_encode
from content_prune import CoverageScorer,encode as reference_encode
ARMS=('full','fixed12','heuristic12','heuristic11','untrained','trained')
SEED=20261101

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rotate(q,shift):
    assert shift in range(5)
    return dict(q,options=q['options'][shift:]+q['options'][:shift],answer_index=(q['answer_index']-shift)%5)

def train_shift(q):return int(hashlib.sha256(('train-rotation-v1:'+q['id']).encode()).hexdigest(),16)%5

def encode_pair(model,batch,arm,scorer,state,heuristic):
    import torch
    visual=model.model.visual
    if arm in ('trained','untrained'):
        value,ids,_,state=learned_encode(visual,batch.pixel_values_videos,batch.video_grid_thw,scorer,previous=state)
        return value,ids,state
    if arm!='heuristic11':
        mode={'full':'full','fixed12':'mid_fixed','heuristic12':'mid_content'}[arm]
        value,ids,_=reference_encode(visual,batch.pixel_values_videos,batch.video_grid_thw,mode=mode,scorer=heuristic)
        return value,ids,None
    # Match the learned scorer's observation point, without its attention bias.
    grid=batch.video_grid_thw;pixels=batch.pixel_values_videos
    x=visual.patch_embed(pixels)+visual.fast_pos_embed_interpolate(grid)
    rope=visual.rot_pos_emb(grid).reshape(len(x),-1);emb=torch.cat((rope,rope),-1)
    positions=(emb.cos(),emb.sin());cu=torch.tensor([0,len(x)],device=x.device,dtype=torch.int32)
    for block in visual.blocks[:11]:x=block(x,cu_seqlens=cu,position_embeddings=positions)
    ids=heuristic.choose(x,(9,12))
    x=visual.blocks[11](x,cu_seqlens=cu,position_embeddings=positions)
    patches=(ids[:,None]*4+torch.arange(4,device=x.device)).flatten()
    x=x[patches];positions=tuple(p[patches] for p in positions)
    cu=torch.tensor([0,len(x)],device=x.device,dtype=torch.int32)
    for block in visual.blocks[12:]:x=block(x,cu_seqlens=cu,position_embeddings=positions)
    return visual.merger(x),ids,None
