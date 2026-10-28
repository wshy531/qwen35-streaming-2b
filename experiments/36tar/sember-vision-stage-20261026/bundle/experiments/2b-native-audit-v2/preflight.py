"""CPU-only identities, 24 layouts, and real four-frame original/stream controls."""
import argparse
import hashlib
import os
import sys
from pathlib import Path
from audit_common import ROOT, OUT, MODEL, read, save, tiny_config, official_identity, paired_inputs, tensor_sha, template


def main(stage,output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    import torch
    import numpy as np
    torch.set_num_threads(1)
    if stage == 'stream':
        sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
        from sember_common import runtime
        env=runtime()
        processor=env['AutoProcessor'].from_pretrained(MODEL,local_files_only=True)
        model_class,config_class=env['Qwen3_5ForConditionalGeneration'],env['Qwen3_5Config']
    else:
        from transformers import AutoProcessor,Qwen3_5Config,Qwen3_5ForConditionalGeneration
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        model_class,config_class=Qwen3_5ForConditionalGeneration,Qwen3_5Config
    sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
    from sember_protocol import prompt
    from sember_source import CausalSource
    questions=read(OUT/'inputs/requests.json')
    expected=read(OUT/'inputs/expected_prefixes.json')
    marker=processor.vision_start_token+processor.video_token+processor.vision_end_token
    layouts={}
    for q in questions:
        prefix,suffix=template(processor,prompt(q)).split(marker)
        state=expected[q['question_id']]
        assert not state['true_EOF_tail_padded'], 'this profile fixes ordinary pre-EOF queries'
        text=prefix+''.join(f'<{i*4+1:.1f} seconds>'+processor.vision_start_token+
            processor.video_token*108+processor.vision_end_token for i in range(state['pairs']))+suffix
        ids=processor.tokenizer(text,add_special_tokens=False,return_tensors='pt').input_ids
        layouts[q['question_id']]=dict(input_ids_sha256=tensor_sha(ids),input_tokens=ids.shape[1],
            visual_tokens=state['pairs']*108,question_sha256=hashlib.sha256(prompt(q).encode()).hexdigest())
        assert ids.shape[1]+16<=262144
    first=questions[0]
    media=ROOT/'datasets/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23'/'videos'/ (first['video_id']+'.mp4')
    source=CausalSource(media)
    try:packets=list(source.advance(6.))
    finally:source.close()
    assert len(packets)==4
    torch.manual_seed(20261025)
    model=model_class(config_class(**tiny_config(processor),attn_implementation='sdpa')).float().eval()
    weight_hash=hashlib.sha256()
    for name,t in model.state_dict().items():
        weight_hash.update(name.encode());weight_hash.update(t.cpu().contiguous().numpy())
    question=prompt(first)
    inputs=paired_inputs(processor,[p.rgb for p in packets],[p.timestamp for p in packets],question)
    with torch.inference_mode():
        if stage=='stream':
            from sember_runtime import make_builder
            from sember_session import Session,emit_decode
            builder=make_builder(env,processor)
            session=Session(env,model,builder,mode='reference')
            for i in (0,2):session.append_pair(*packets[i:i+2])
            branch=session.main.fork()
            with env['fp32_reference']():
                logits=branch.append(input_ids=builder.question_ids(question),logits_mode='last')
                initial=logits.clone()
                generated=emit_decode(branch,logits,processor,2,set(),lambda x:None)['generated_ids']
            branch.cache.reset();session.close()
            identity=dict(implementation='current_stream',official_unmodified=False)
        else:
            identity=official_identity(model)
            from torch.nn.attention import sdpa_kernel,SDPBackend
            with sdpa_kernel(SDPBackend.MATH):
                result=model.generate(**inputs,do_sample=False,max_new_tokens=2,use_cache=True,
                    return_dict_in_generate=True,output_scores=True,pad_token_id=processor.tokenizer.eos_token_id)
            initial=result.scores[0][:,None,:]
            generated=result.sequences[0,inputs.input_ids.shape[1]:].tolist()
    np.save(Path(output).with_suffix('.npy'),initial.numpy())
    save(output,dict(status='passed',stage=stage,identity=identity,layouts=layouts,
        state_dict_sha256=weight_hash.hexdigest(),input_ids_sha256=tensor_sha(inputs.input_ids),
        pixels_sha256=tensor_sha(inputs.pixel_values_videos),generated_ids=generated,
        real_frames=4,real_grid=[9,12],CPU_only=True,pretrained_loads=0,GPU_jobs=0))
    print(dict(status='passed',stage=stage,layouts=len(layouts),real_frames=4,generated_ids=generated,
        max_layout_tokens=max(v['input_tokens'] for v in layouts.values())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['official530','official540','stream'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();main(args.stage,args.output)
