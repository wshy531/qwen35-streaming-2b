"""Real input, random tiny weights: single-arm QA, resume and bounded cleanup."""
import os
import torch
from common import ROOT,OUT,ARMS,read,save,runtime
from source import ExactPackets
from session import Session

assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=runtime();torch.set_num_threads(1);torch.manual_seed(321)
builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),canvas=(192,256),resize_mode='letterbox')
model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](builder.processor),attn_implementation='sdpa')).float().eval()
group=next(g for g in read(OUT/'inputs/groups.json') if g['cohort']=='stream34')
key='cluster_media' if (ROOT/'repo').exists() else 'local_media'
source=ExactPackets(env,ROOT/group[key],group['frames'][:22],group['sha256'])
packets=list(source.packets());assert len(packets)==22
answers=[]
with torch.inference_mode(),env['fp32_reference']():
    for arm in ARMS:
        session=Session(env,model,builder,arm)
        for i in range(0,18,2):session.append_pair(*packets[i:i+2])
        before=session.bank.fingerprint();selector=session.bank.selector_digest();tokens=[]
        result=session.ask('What is visible?',read_from='standalone',on_token=tokens.append)
        assert set(result['arms'])=={arm} and result['arms'][arm]['generated_ids']==[r['token_id'] for r in tokens]
        assert session.bank.fingerprint()==before and session.bank.selector_digest()==selector
        session.append_pair(*packets[18:20]);session.snapshot_tail(packets[20])
        assert session.vision_encodings==11 and session.audit_seconds>=0
        try:session.append_pair(*packets[20:22])
        except RuntimeError:pass
        else:raise AssertionError('query-only tail committed')
        answers.append(dict(arm=arm,generated_tokens=len(tokens),input_tokens=result['arms'][arm]['input_tokens'],
                            visual_tokens=result['arms'][arm]['visual_tokens'],post_QA_continuation=True))
        session.close();assert not session.bank.banks and session.bank.previous is None
save(OUT/'tiny_session_check.json',dict(status='passed',real_frames=22,vision_calls=33,QA=answers,
    exact_source_decode_verified=True,source_profile=source.status(),one_active_bank=True,
    no_main_language_cache=True,all_closed=True,pretrained_loads=0,GPU_jobs=0))
print(dict(status='passed',tiny_QA=3),flush=True)
