"""Exercise actual official from_pretrained loading metadata, using only a tiny synthetic checkpoint."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from audit_common import OUT,save,official_identity,tiny_config


def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    import torch
    from transformers import Qwen3_5Config,Qwen3_5ForConditionalGeneration
    torch.set_num_threads(1);torch.manual_seed(20261025)
    processor=SimpleNamespace(image_token_id=250,video_token_id=251,
        vision_start_token='start',vision_end_token='end',
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda s:252 if s=='start' else 253))
    config=tiny_config(processor);config['text_config'].update(vocab_size=256,eos_token_id=2,pad_token_id=0)
    model=Qwen3_5ForConditionalGeneration(Qwen3_5Config(**config,attn_implementation='sdpa')).float().eval()
    checkpoint=Path(output).with_suffix('')
    assert not checkpoint.exists();model.save_pretrained(checkpoint,safe_serialization=True)
    loaded,info=Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint,local_files_only=True,
        dtype=torch.float32,output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys']
    old_would_fail=False
    try:json.dumps(info)
    except TypeError:old_would_fail=True
    startup=Path(output).with_suffix('.startup.json')
    save(startup,dict(loading_info=info,deliberate_empty_set=set(),nonempty_set={'b','a'}))
    parsed=json.loads(startup.read_text())
    assert parsed['loading_info']['missing_keys']==[] and parsed['loading_info']['unexpected_keys']==[]
    assert parsed['deliberate_empty_set']==[] and parsed['nonempty_set']==['a','b']
    unsupported=Path(output).with_suffix('.unsupported.json')
    try:save(unsupported,dict(bad=object()))
    except TypeError:pass
    else:raise AssertionError('unsupported metadata must fail explicitly')
    assert not unsupported.exists(),'serialization must complete before creating the destination'
    with torch.inference_mode():
        generated=loaded.generate(input_ids=torch.tensor([[15,16]]),do_sample=False,max_new_tokens=2,
            pad_token_id=0,logits_to_keep=1)
    save(output,dict(status='passed',identity=official_identity(loaded),
        loading_info_types={k:type(v).__name__ for k,v in info.items()},
        reproduced_original_JSON_failure=old_would_fail,real_from_pretrained_metadata_roundtrip=True,
        unsupported_metadata_does_not_leave_partial_file=True,synthetic_checkpoint_loads=1,
        synthetic_checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.iterdir()),
        generated_ids=generated[0,2:].tolist(),real_2B_weights_loaded=0,GPU_jobs=0))
    print(dict(status='passed',original_failure_reproduced=old_would_fail,
        loading_types={k:type(v).__name__ for k,v in info.items()},real_2B_weights=0,GPU_jobs=0),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
