"""Fixed 30-source short-clip regression for the three frozen order endpoints."""
import argparse
import json
import os
import time
from pathlib import Path

import torch

from run_screen import (
    ROOT, PREP, WEIGHTS, GroupScorer, capture_pairs, make_builder,
    read, runtime, score, sha,
)
from run_order import decode

ARMS = {'initial':'initial.pt', 'short_then_delayed':'short_then_delayed_step8.pt', 'mixed':'mixed_step8.pt'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-tiny', action='store_true')
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to((ROOT/'runs').resolve())
    contract = read(PREP/'regression_lock.json')
    for path,digest in contract['files'].items():
        assert sha(ROOT/path)==digest,path
    rows = read(PREP/'regression30_LOCAL_ONLY.json')
    assert len(rows)==len({r['video_id'] for r in rows})==30
    torch.set_num_threads(1)
    torch.manual_seed(20261104)
    if args.cpu_tiny:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    else:
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
        assert torch.cuda.device_count()==1
    env = runtime()
    builder = make_builder(env, env['AutoProcessor'].from_pretrained(WEIGHTS,local_files_only=True))
    if args.cpu_tiny:
        config=env['tiny_config'](builder.processor)
        config['vision_config']['depth']=24
        model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config,attn_implementation='sdpa')).float()
    else:
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(
            WEIGHTS,local_files_only=True,dtype=torch.float32,device_map='cuda',
            attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.eval().requires_grad_(False)
    device=model.get_input_embeddings().weight.device
    scorer=GroupScorer(model.config.vision_config.hidden_size).to(device).eval().requires_grad_(False)
    states = {arm:torch.load(ROOT/contract['endpoint_directory']/name,map_location='cpu',weights_only=True)
              for arm,name in ARMS.items()} if not args.cpu_tiny else {}
    args.out.mkdir()
    started=time.monotonic()
    calls=answers=0
    with torch.no_grad(),env['fp32_reference'](),(args.out/'answers.jsonl').open('x',encoding='utf-8') as log:
        for row in rows:
            path=ROOT/row['path']
            assert sha(path)==row['sha256']
            batches=decode(env,path,row['audit'],builder)
            assert len(batches)==row['audit']['encoding_pairs']
            inputs=[(a,b,batch.pixel_values_videos,batch.video_grid_thw) for a,b,batch in batches]
            for arm in (*ARMS,'text'):
                nodes=None
                if arm!='text':
                    if states:
                        scorer.load_state_dict(states[arm])
                    capture=capture_pairs(model.model.visual,scorer,inputs)
                    nodes=capture.nodes
                    calls+=len(inputs)
                for shift in (0,2):
                    log.write(json.dumps(score(model,builder,row,nodes,shift,arm,[]))+'\n')
                    answers+=1
            log.flush()
    assert answers==240 and calls==1311
    receipt=dict(status='completed',cpu_tiny=args.cpu_tiny,answers=answers,visual_calls=calls,
                 sources=30,seconds=time.monotonic()-started,updates=0,generated_tokens=0)
    (args.out/'completed.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
