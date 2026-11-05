"""Actual0.8B tokenizer/template capacity check for locked holdout51, no model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

from pathlib import Path
import sys

from prepare_holdout_review import ROOT,OUT,read,save,sha
from score_memory_eval import digest,prompt


def main():
    lock=read(OUT/'DATA_ROSTER_LOCK.json')
    for p,h in lock['file_sha256'].items():assert sha(OUT/p)==h,p
    sys.path.insert(0,str(ROOT/'experiments/08b-memory-eval'))
    from capability_pilot import imports
    env=imports();torch=env['torch'];torch.set_num_threads(1)
    assert not torch.cuda.is_available()
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-0.8B',local_files_only=True)
    builder=env['PairInputBuilder'](processor,canvas=(192,256),resize_mode='letterbox')
    rows={q['id']:q for q in read(OUT/'admitted.json')};requests=read(OUT/'model_requests.json');details=[]
    for request in requests:
        q=rows[request['id']];v=request['model_visible']
        assert v['question']==q['question'] and v['options']==q['options']
        suffix=builder.question_ids(prompt(v)).shape[1];assert suffix<=128
        frames=v['frames'];pairs=[frames[i:i+2] for i in range(0,len(frames),2)]
        total=builder.prefix_ids.shape[1]+suffix
        for pair in pairs:
            center=(pair[0]['pts_s']+pair[-1]['pts_s'])/2
            ids=builder.tokenize(f'<{center:.1f} seconds>'+processor.vision_start_token+processor.video_token*48+processor.vision_end_token)
            assert int((ids==processor.video_token_id).sum())==48
            total+=ids.shape[1]
        assert total+8<=8192 and len(pairs)<=128 and len(pairs)*48<=6144
        details.append(dict(id=q['id'],cohort=q['curation']['cohort'],suffix_tokens=suffix,
            full_pairwise_input_tokens=total,pairs=len(pairs),visual_tokens=len(pairs)*48,real_frames=len(frames),
            snapshot_padding=len(frames)%2==1,exceeds_prior_native_matched_3072_guard=total+8>3072))
    result=dict(status='passed_08b_pairwise_tokenizer_capacity_only',questions=len(details),
        max_suffix_tokens=max(q['suffix_tokens'] for q in details),max_full_input_tokens=max(q['full_pairwise_input_tokens'] for q in details),
        max_pairs=max(q['pairs'] for q in details),max_visual_tokens=max(q['visual_tokens'] for q in details),
        exceeds_prior_native_matched_3072_guard=[q['id'] for q in details if q['exceeds_prior_native_matched_3072_guard']],
        natural_core_max_input_tokens=max(q['full_pairwise_input_tokens'] for q in details if q['cohort']=='natural_short_core'),
        model_instances=0,pretrained_loads=0,cuda_visible=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'experiments/08b-memory-eval/capability_pilot.py',
            ROOT/'experiments/08b-memory-eval/readout.py',ROOT/'experiments/2b-memory-eval/prefix_request.py']},
        processor_revision=env['profile']['model_revision'],
        profile_sha256=sha(ROOT/'runs/08b_fallback_preparation_20261018_01/profile.json'),
        prompt_template_sha256=digest([prompt(rows[q['id']]) for q in requests]),
        data_lock_sha256=sha(OUT/'DATA_ROSTER_LOCK.json'),details=details,
        limits='8192 context including8 generated tokens,6144 visual,128 nodes,128 suffix',
        limitation='No model, vision encoder, native processor pixels, position equivalence or answer. Native helper3072 guard requires separate adaptation/preflight for long supplementary inputs. No2B admission.')
    assert result['prompt_template_sha256']==read(OUT/'execution_bundle_draft.json')['input_protocol']['prompt_template_sha256']
    save(OUT/'cpu_input_preflight.json',result)
    print({k:v for k,v in result.items() if k not in ['source_sha256','details','limitation']})


if __name__=='__main__':main()
