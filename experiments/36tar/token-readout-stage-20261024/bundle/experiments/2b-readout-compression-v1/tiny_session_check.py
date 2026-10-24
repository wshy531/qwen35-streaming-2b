"""Real file input, tiny random weights, QA isolation and resumed observation."""
import os,torch
from common import ROOT,OUT,ARMS,read,save,runtime
from io_source import BoundedAVSource
from session import Session
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
torch.set_num_threads(1);torch.manual_seed(321);env=runtime()
builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),canvas=(192,256),resize_mode='letterbox')
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](builder.processor),attn_implementation='sdpa')).float().eval()
g=next(g for g in read(OUT/'inputs/groups.json') if g['cohort']=='T98')
key='cluster_media' if (ROOT/'repo').exists() else 'local_media'
source=BoundedAVSource(ROOT/g[key],dict(g,frames=g['frames'][:7]),env['FramePacket'],config='auto4')
packets=list(source.packets());assert len(packets)==7
QA=[]
with torch.inference_mode(),env['fp32_reference']():
    s=Session(env,model,builder);s.append_pair(*packets[:2]);s.append_pair(*packets[2:4]);tokens=[]
    before=s.bank.fingerprint();result=s.ask('What is visible?',read_from='readout_banks',on_token=tokens.append)
    assert before==s.bank.fingerprint() and set(result['arms'])==set(ARMS)
    for arm in ARMS:
        a=result['arms'][arm]
        assert a['generated_ids']==[t['token_id'] for t in tokens if t['arm']==arm]
        assert a['banks_unchanged'] and a['branch_released']
        QA.append(dict(arm=arm,input_tokens=a['input_tokens'],visual_tokens=a['visual_tokens'],generated_tokens=len(a['generated_ids'])))
    s.append_pair(*packets[4:6]);s.snapshot_tail(packets[6]);assert s.vision_encodings==4
    try:s.append_pair(*packets[:2])
    except RuntimeError:pass
    else:raise AssertionError('query-only tail continued')
    s.close();assert not s.bank.banks and s.bank.previous is None and s.bank.descriptor is None
save(OUT/'tiny_session_check.json',dict(status='passed',QA=QA,vision_calls=4,real_frames=7,
    all_closed=True,pretrained_loads=0,GPU_jobs=0,post_QA_continuation=True,source_profile=source.status()))
print(dict(status='passed',QA=6,vision_calls=4),flush=True)
