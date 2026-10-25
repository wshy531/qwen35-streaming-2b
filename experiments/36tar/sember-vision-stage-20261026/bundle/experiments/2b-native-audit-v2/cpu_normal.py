"""One short real prefix through the untouched default video processor and original model API."""
import argparse
import os
import sys
from audit_common import ROOT,OUT,MODEL,read,save,official_identity,tiny_config


def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    import torch
    from transformers import AutoProcessor,Qwen3_5Config,Qwen3_5ForConditionalGeneration
    from native_inputs import normal_video
    torch.set_num_threads(1)
    q=read(OUT/'inputs/requests.json')[0].copy()
    q['question_time']=6.
    path=ROOT/'datasets/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos'/(q['video_id']+'.mp4')
    inputs,metadata=normal_video(path,q)
    processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
    torch.manual_seed(20261025)
    model=Qwen3_5ForConditionalGeneration(Qwen3_5Config(**tiny_config(processor),attn_implementation='sdpa')).float().eval()
    with torch.inference_mode():
        result=model.generate(**inputs,do_sample=False,max_new_tokens=2,use_cache=True,
            logits_to_keep=1,pad_token_id=processor.tokenizer.eos_token_id)
    identity=official_identity(model)
    save(output,dict(status='passed',identity=identity,input=metadata,
        generated_ids=result[0,inputs.input_ids.shape[1]:].tolist(),real_prefix_seconds=6.,
        pretrained_loads=0,GPU_jobs=0,no_capability_claim=True))
    print(dict(status='passed',frames=metadata['selected_frames'],grid=metadata['grid'],tokens=metadata['input_tokens']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
