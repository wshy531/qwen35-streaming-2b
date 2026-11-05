"""Actual local tokenizer/template capacity check; no model instantiated."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

import math
from pathlib import Path
import sys

from prepare_natural_v2 import ROOT,OUT,read,save,sha
from score_memory_eval import prompt,digest


def main():
    sys.path.insert(0,str(ROOT/'experiments/08b-memory-eval'))
    from capability_pilot import imports
    env=imports();torch=env['torch'];torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-0.8B',local_files_only=True)
    builder=env['PairInputBuilder'](processor,canvas=(192,256),resize_mode='letterbox')
    rows={q['id']:q for q in read(OUT/'admitted74.json')}
    requests=read(OUT/'model_requests74.json');details=[]
    for request in requests:
        row=rows[request['id']];visible=request['model_visible']
        assert visible['question']==row['question'] and visible['options']==row['options']
        suffix=builder.question_ids(prompt(visible));suffix_length=suffix.shape[1]
        assert suffix_length<=128
        frames=visible['frames'];pairs=[frames[i:i+2] for i in range(0,len(frames),2)]
        total=builder.prefix_ids.shape[1]+suffix_length
        for pair in pairs:
            center=(pair[0]['pts_s']+pair[-1]['pts_s'])/2
            label=f'<{center:.1f} seconds>'
            ids=builder.tokenize(label+processor.vision_start_token+processor.video_token*48+processor.vision_end_token)
            total+=ids.shape[1]
        assert total+8<=8192 and len(pairs)<=128 and len(pairs)*48<=6144
        details.append(dict(id=row['id'],suffix_tokens=suffix_length,full_pairwise_input_tokens=total,
            pairs=len(pairs),visual_tokens=len(pairs)*48,causal_frames=len(frames),snapshot_padding=len(frames)%2==1))
    save(OUT/'cpu_input_preflight.json',dict(status='passed_actual_08b_tokenizer_and_template_no_model',
        questions=74,source_files=len({q['media_path'] for q in requests}),model_instances=0,pretrained_loads=0,cuda_visible=False,
        suffix_limit=128,max_suffix_tokens=max(q['suffix_tokens'] for q in details),
        context_limit=8192,max_full_pairwise_input_tokens=max(q['full_pairwise_input_tokens'] for q in details),
        max_full_visual_tokens=max(q['visual_tokens'] for q in details),max_pairs=max(q['pairs'] for q in details),
        processor_revision=env['profile']['model_revision'],profile_sha256=sha(ROOT/'runs/08b_fallback_preparation_20261018_01/profile.json'),
        prompt_template_sha256=digest([prompt(rows[q['id']]) for q in requests]),details=details,
        scope='No Qwen weights, model forward, numerical equivalence, QA answer or 2B validation. Full causal frames retained; no question truncation.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiments/08b-memory-eval/readout.py',
            ROOT/'experiments/08b-memory-eval/capability_pilot.py',ROOT/'experiments/2b-memory-eval/prefix_request.py']}))
    result=read(OUT/'cpu_input_preflight.json');print({k:v for k,v in result.items() if k not in ['details','source_sha256','scope']})


if __name__=='__main__':main()
