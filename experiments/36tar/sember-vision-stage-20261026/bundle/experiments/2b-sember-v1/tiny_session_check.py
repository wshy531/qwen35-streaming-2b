"""Real development pixels, random tiny model: grid/QA/resume/EOF; no accuracy."""
import os
import torch
from sember_common import ROOT,OUT,PREP,DATA,ARMS,runtime,read,save
from sember_source import CausalSource
from sember_runtime import make_builder,context
from sember_session import Session,signature


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1);torch.manual_seed(20261025)
    env=runtime();processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor)
    cfg=env['tiny_config'](processor)
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
    q=read(PREP/'profiler_requests.json')[0]
    source=CausalSource(DATA/('videos/'+q['video_id']+'.mp4'))
    try:packets=list(source.advance(12.))
    finally:source.close()
    assert len(packets)==7
    inputs,record=builder.pair(*packets[:2],0,0)
    assert record['grid']==[[1,18,24]] and (inputs.input_ids==processor.video_token_id).sum()==108
    answers=[];question='What is visible?'
    with torch.inference_mode(),env['fp32_reference']():
        session=Session(env,model,builder)
        for i in (0,2):session.append_pair(*packets[i:i+2])
        initial=session.bank.fingerprint();all_tokens=[]
        for arm in ARMS:
            tokens=[];row=session.ask(arm,question,tokens.append,max_new_tokens=2)
            assert row['generated_ids']==[t['token_id'] for t in tokens]
            assert row['read_state_unchanged'] and row['branch_released']
            import hashlib
            assert row['question_suffix_sha256']==hashlib.sha256(builder.question_ids(question).numpy().tobytes()).hexdigest()
            answers.append(dict(arm=arm,input_tokens=row['input_tokens'],visual_tokens=row['visual_tokens'],generated=row['generated_ids']))
            all_tokens.extend(tokens)
        assert session.bank.fingerprint()==initial and not hasattr(session,'main')
        repeat=session.ask(ARMS[0],question,lambda x:None,max_new_tokens=2)
        assert repeat['generated_ids']==answers[0]['generated']
        session.append_pair(*packets[4:6]);session.append_pair(packets[6],packets[6],padding=True)
        try:session.append_pair(*packets[:2])
        except RuntimeError:pass
        else:raise AssertionError('EOF tail session continued')
        session.close();assert not session.bank.banks
        # A distinct full-history session has main cache and no bounded bank.
        reference=Session(env,model,builder,mode='reference')
        for i in (0,2):reference.append_pair(*packets[i:i+2])
        initial_signature=signature(reference.main);tokens=[]
        row=reference.ask('R_all_history_stream',question,tokens.append,max_new_tokens=2)
        assert signature(reference.main)==initial_signature and not hasattr(reference,'bank')
        assert row['bounded_context_ids_sha256'] is None
        assert row['question_suffix_sha256']==repeat['question_suffix_sha256']
        assert row['generated_ids']==answers[0]['generated'],(row['generated_ids'],answers[0]['generated'])
        reference.append_pair(*packets[4:6]);reference.append_pair(packets[6],packets[6],padding=True)
        reference.close();assert reference.main.seen_tokens==0
        # All frozen full MCQs must fit without truncation, including actual special suffix.
        max_suffix=0
        for r in read(PREP/'selection/requests.json'):
            from sember_protocol import prompt
            ids=builder.question_ids(prompt(r));max_suffix=max(max_suffix,ids.shape[1]);assert ids.shape[1]<=512
    result=dict(status='passed',real_frames=7,grid=[9,12],visual_tokens_per_pair=108,
        bounded_answers=12,reference_answers=1,QA=answers,max_real_question_suffix_tokens=max_suffix,
        all_11_arms_stream_callbacks=True,all_banks_unchanged=True,post_QA_continuation=True,
        bounded_no_main_KV=True,reference_no_feature_archive=True,reference_cache_SHA_unchanged=True,
        full_bank_and_reference_same_short_answer=True,EOF_tail_finalizes=True,
        pretrained_loads=0,random_tiny_model_loads=1,GPU_jobs=0,no_capability_or_speed_claim=True)
    save(OUT/'tiny_session_check.json',result);print({k:v for k,v in result.items() if k!='QA'},flush=True)


if __name__=='__main__':main()
