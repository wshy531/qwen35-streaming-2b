"""Isolated original-native versus continued tiny FP32 control on the new grid."""
import argparse
import hashlib
import os
import numpy as np
import torch
from sember_common import ROOT,OUT,PREP,DATA,read,save,runtime
from sember_source import CausalSource
from sember_runtime import make_builder
from sember_session import Session


def main(stage):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1);env=runtime(stage);torch.manual_seed(20261025)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor);cfg=env['tiny_config'](processor)
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
    h=hashlib.sha256()
    for name,t in model.state_dict().items():h.update(name.encode());h.update(t.cpu().contiguous().numpy())
    question='What is visible?';q=read(PREP/'profiler_requests.json')[0]
    source=CausalSource(DATA/('videos/'+q['video_id']+'.mp4'))
    try:packets=list(source.advance(6.))
    finally:source.close()
    with torch.inference_mode(),env['fp32_reference']():
        if stage=='memory':
            ref=Session(env,model,builder,mode='reference')
            ref.append_pair(*packets[:2]);ref.append_pair(*packets[2:4])
            branch=ref.main.fork();ids=builder.question_ids(question)
            logits=branch.append(input_ids=ids,logits_mode='last');branch.cache.reset();ref.close()
        else:
            ids=builder.prefix_ids;emb=model.get_input_embeddings()(ids)
            pos=torch.arange(ids.shape[1]).expand(4,1,-1).clone();seen=ids.shape[1];rope=seen
            for i in (0,2):
                inputs,_=builder.pair(*packets[i:i+2],seen,rope)
                visual=torch.cat(model.model.get_video_features(inputs.pixel_values_videos,inputs.video_grid_thw,return_dict=True).pooler_output,0)
                block=model.get_input_embeddings()(inputs.input_ids)
                mask=(inputs.input_ids==processor.video_token_id).unsqueeze(-1).expand_as(block)
                emb=torch.cat((emb,block.masked_scatter(mask,visual.to(block))),1)
                pos=torch.cat((pos,inputs.position_ids),2);seen=emb.shape[1];rope=int(pos[1:].max())+1
            suffix=builder.question_ids(question);emb=torch.cat((emb,model.get_input_embeddings()(suffix)),1)
            suffix_pos=torch.stack((torch.arange(seen,seen+suffix.shape[1]),*(torch.arange(rope,rope+suffix.shape[1]) for _ in range(3))))[:,None,:]
            pos=torch.cat((pos,suffix_pos),2)
            result=model.model.language_model(inputs_embeds=emb,position_ids=pos,use_cache=False)
            logits=model.lm_head(result.last_hidden_state[:,-1:,:])
    assert torch.isfinite(logits).all()
    np.save(OUT/f'tiny_logits_{stage}.npy',logits.cpu().numpy())
    result=dict(status='output_saved_for_comparison',stage=stage,seed=20261025,state_dict_sha256=h.hexdigest(),
        real_frames=4,visual_tokens=216,pretrained_loads=0,GPU_jobs=0)
    save(OUT/f'tiny_native_{stage}.json',result);print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['native','memory'],required=True)
    main(parser.parse_args().stage)
