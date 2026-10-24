"""Real RGB random tiny model: rollover, matched reads, continuation, cleanup."""
import os
import torch
from common import ROOT,OUT,ARMS,read,save,runtime
from session import Session

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=runtime();torch.set_num_threads(1);torch.manual_seed(321)
builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),canvas=(192,256),resize_mode='letterbox')
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](builder.processor),attn_implementation='sdpa')).float().eval()
from replay import VideoSource
group=next(g for g in read(OUT/'inputs/groups.json') if g['cohort']=='stream34')
key='cluster_media' if (ROOT/'repo').exists() else 'local_media'
source=VideoSource(ROOT/group[key],dict(frames=group['frames'],media_sha256=group['sha256']),env['FramePacket'])
packets=[];iterator=iter(source.frames())
while len(packets)<38:
    first,second=next(iterator),next(iterator)
    packets.extend(source.take_pair([first.frame_id,second.frame_id]))
iterator.close()
session=Session(env,model,builder);answers=[]
with torch.inference_mode(),env['fp32_reference']():
    for i in range(0,34,2):session.append_pair(*packets[i:i+2])
    assert session.pairs==17
    before=session.bank.fingerprint();selector=session.bank.selector_digest()
    for arm in ARMS:
        tokens=[];answer=session.ask_arm(arm,'What is visible?',tokens.append)
        assert answer['generated_ids']==[t['token_id'] for t in tokens]
        assert session.bank.fingerprint()==before and session.bank.selector_digest()==selector
        answers.append(dict(arm=arm,generated_tokens=len(tokens),input_tokens=answer['input_tokens'],
                            visual_tokens=answer['visual_tokens'],bank_and_selector_unchanged=True))
    session.append_pair(*packets[34:36]);assert session.pairs==18
    session.snapshot_tail(packets[36]);assert session.tail_snapshot_only
    try:session.append_pair(*packets[36:38])
    except RuntimeError:pass
    else:raise AssertionError('query-only tail committed')
    timing=session.timing.copy();session.close()
    assert not session.bank.banks and session.bank.previous is None
save(OUT/'tiny_session_check.json',dict(status='passed',real_frames=38,vision_calls=19,QA=answers,
    post_QA_real_pair_continuation=True,
    timing=timing,no_main_language_cache=True,all_closed=True,pretrained_loads=0,GPU_jobs=0))
print(dict(status='passed',tiny_QA=len(answers)),flush=True)
