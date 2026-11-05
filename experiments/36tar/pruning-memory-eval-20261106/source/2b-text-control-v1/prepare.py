"""Preserve all 478 original questions and build a literal visual-removal control."""
import os
import torch
from text_common import ROOT,OUT,MAIN,read,save,sha,runtime,make_builder,text_ids
from text_inference import answer


def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1);torch.manual_seed(20261025)
    env=runtime();processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor);qs=read(MAIN/'inputs/requests.json');assert len(qs)==478
    specs={};longest=None
    for q in qs:
        assert len(q['options'])==5
        ids,spec=text_ids(builder,q);specs[q['question_id']]=spec
        if longest is None or spec['input_tokens']>specs[longest['question_id']]['input_tokens']:longest=q
    cfg=env['tiny_config'](processor)
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
    calls=[]
    def forbid(*args,**kwargs):raise AssertionError('text control invoked vision')
    handle=model.model.visual.register_forward_pre_hook(forbid)
    try:
        first=None
        for q in (qs[0],longest,qs[0]):
            ids,_=text_ids(builder,q);events=[];row=answer(env,model,processor,ids,events.append,limit=2)
            assert row['generated_ids']==[e['token_id'] for e in events] and row['branch_released']
            if first is None:first=row['generated_ids']
            calls.append(dict(input_tokens=row['input_tokens'],vision_calls=0))
        assert first==row['generated_ids'],'question order leaked across fresh contexts'
    finally:handle.remove()
    if not (OUT/'inputs').exists():
        (OUT/'inputs').mkdir(parents=True)
        save(OUT/'inputs/requests.json',qs);save(OUT/'inputs/text_specs.json',specs)
        save(OUT/'inputs/protocol.json',dict(stage='same_478_questions_text_only_missing_modality_control',
            questions=478,sources=128,formal_answers=478,warmup_answers=1,max_answers=479,
            max_generated_tokens=7664,pretrained_loads=1,visual_encodings=0,video_decodes=0,
            dtype='FP32',max_new_tokens=16,prefill_chunk=512,context_limit=1024,
            retain_original_MCQs=True,retain_original_option_order=True,
            template='original PairInputBuilder template with only the video marker removed',
            retained_video_instruction=True,no_new_guessing_instruction=True,
            primary_denominator=478,format_not_multiplied_into_content=True,all_failures_preserved=True,
            comparison='all 12 frozen completed 218526 conditions; no filtering by outcomes',
            groups=['all478','ge120s85','ge300s25','eight_original_types'],
            uniform_random_chance=.2,expected_uniform_correct=95.6,
            fixed_letter_controls='report A/B/C/D/E separately; do not select a deployment rule from labels',
            report_prediction_marginals=True,report_gold_marginals_in_local_analysis_only=True,
            source_bootstrap_draws=5000,source_bootstrap_seed=20261025,
            one_round_no_prompt_or_order_sweep=True,not_a_proof_of_random_guessing=True,
            resource_budget=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,RAM_GiB=24,minutes=15),
            log_limit_bytes=16*2**20,worker_RSS_limit_bytes=20*2**30,GPU_allocated_limit_bytes=32*2**30,
            GPU_authorized=False,training=False,download=False,automatic_retry=False,
            independent_of_218537_outputs=True,does_not_modify_running_218537=True))
    else:
        assert qs==read(OUT/'inputs/requests.json') and specs==read(OUT/'inputs/text_specs.json')
    result=dict(status='passed',questions=478,option_counts={'5':478},literal_video_marker_removal=True,
        max_input_tokens=max(s['input_tokens'] for s in specs.values()),all_token_IDs_frozen=True,
        tiny_QAs=calls,repeat_first_question_identical=True,no_vision_or_video_calls=True,
        all_questions_without_truncation=True,pretrained_loads=0,GPU_jobs=0,no_capability_claim=True)
    save(output,result);print(result,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=__import__('pathlib').Path,required=True)
    main(p.parse_args().output)
