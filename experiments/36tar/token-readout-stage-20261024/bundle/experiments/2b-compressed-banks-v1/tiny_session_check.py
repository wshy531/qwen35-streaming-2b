"""Real RGB in a random tiny model: observe, nine isolated reads, continue, clear."""
import os,time,torch
from common import ROOT,OUT,ARMS,read,save,runtime
from session import Session

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=runtime();torch.set_num_threads(1);torch.manual_seed(321)
build=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),canvas=(192,256),resize_mode='letterbox')
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](build.processor),attn_implementation='sdpa')).float().eval()
from replay import VideoSource
g=next(g for g in read(ROOT/'runs/2b_temporal24_prepare_20261023_01/inputs/groups.json') if g['cohort']=='stream34')
key='cluster_media' if (ROOT/'repo').exists() else 'local_media'
source=VideoSource(ROOT/g[key],dict(frames=g['frames'],media_sha256=g['sha256']),env['FramePacket'])
packets=[];it=iter(source.frames())
while len(packets)<22:
    first,second=next(it),next(it)
    packets.extend(source.take_pair([first.frame_id,second.frame_id]))
it.close()
session=Session(env,model,build);answers=[]
with torch.inference_mode(),env['fp32_reference']():
    for i in range(0,18,2):session.append_pair(*packets[i:i+2])
    before=session.bank.fingerprint();selector=session.bank.selector_digest()
    for arm in ARMS:
        tokens=[];a=session.ask_arm(arm,'What is visible?',tokens.append)
        assert a['generated_ids']==[t['token_id'] for t in tokens]
        assert before==session.bank.fingerprint() and selector==session.bank.selector_digest()
        answers.append(dict(arm=arm,generated_tokens=len(tokens),bank_and_selector_unchanged=True))
    session.append_pair(*packets[18:20]);assert session.pairs==10
    trace=session.snapshot_tail(packets[20]);assert trace['selection_trace']['pair_index']==10
    assert session.tail_snapshot_only
    try:session.append_pair(*packets[20:22])
    except RuntimeError:pass
    else:raise AssertionError('query-only tail committed')
    timing=session.timing.copy();session.close()
    assert not session.bank.banks and session.bank.previous is None and session.bank.descriptor is None
save(OUT/'tiny_session_check.json',dict(status='passed',real_frames=22,vision_calls=11,
    QA=answers,timing=timing,no_main_language_cache=True,all_closed=True,pretrained_loads=0,GPU_jobs=0))
print(dict(status='passed',tiny_QA=len(answers),timing=timing))
