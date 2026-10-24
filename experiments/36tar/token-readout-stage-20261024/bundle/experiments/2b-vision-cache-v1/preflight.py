"""606 fixed readout layouts and four tiny causal questions; no real weights."""
import argparse
import hashlib
import os
import time
import torch
from common import ROOT, OUT, ARMS, read, save, sha, runtime
from banks import Banks
from bank_readout import context, Limits
from io_source import BoundedAVSource
from cacher import PairVisionCache
from session import Session

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
parser=argparse.ArgumentParser()
parser.add_argument('--receipt',default='CPU_preflight.json')
args=parser.parse_args()
assert args.receipt.endswith('.json') and '/' not in args.receipt and '\\' not in args.receipt
torch.set_num_threads(1)
torch.manual_seed(317)
started = time.monotonic()
env = runtime()
assert not torch.cuda.is_available()
builder = env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B', local_files_only=True),
    canvas=(192,256), resize_mode='letterbox')
model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](
    **env['tiny_config'](builder.processor), attn_implementation='sdpa')).float().eval()
questions, groups = read(OUT/'inputs/questions.json'), read(OUT/'inputs/groups.json')
expected = {r['id']:r for r in read(OUT/'historical_references_LOCAL_ONLY.json')['records']}
from score_memory_eval import prompt
layouts = []
accounting = {}
with torch.inference_mode(), env['fp32_reference']():
    for group in groups:
        for arm in ARMS:
            bank = Banks(arm)
            for q in sorted([q for q in questions if q['group']==group['id']], key=lambda q:(q['pairs'],q['id'])):
                while bank.pairs < q['pairs']:
                    i=bank.pairs
                    first,second=group['frames'][2*i],group['frames'][min(2*i+1,len(group['frames'])-1)]
                    bank.append(torch.zeros(48,32),i,first['pts_s'],second['pts_s'])
                before=bank.fingerprint()
                _,_,ids,meta=context(model,builder,bank.export(arm),prompt(q),
                    limits=Limits(2048,384,8,128),max_new_tokens=8)
                digest=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                assert digest==expected[q['id']]['input_ids_sha256']
                assert bank.fingerprint()==before and meta['visual_tokens']<=384
                layouts.append(dict(id=q['id'],arm=arm,input_tokens=meta['input_tokens'],
                    visual_tokens=meta['visual_tokens'],input_ids_sha256=digest))
            bank.clear()
    group=next(g for g in sorted(groups,key=lambda g:g['sha256']) if g['cohort']=='T98')
    path=ROOT/group['local_media' if not (ROOT/'repo').exists() else 'cluster_media']
    source=BoundedAVSource(path,dict(group,frames=group['frames'][:7]),env['FramePacket'],config='auto4')
    packets=list(source.packets())
    assert len(packets)==7
    QA=[]
    with PairVisionCache(model.model.visual) as cache:
        session=Session(env,model,builder,cache)
        session.append_pair(*packets[:2]);session.append_pair(*packets[2:4])
        for index in range(2):
            before={i:{k:v.clone() for k,v in r.items()} for i,r in cache.references.items()}
            tokens=[]
            result=session.ask('What is visible?',read_from='visual_cache',on_token=tokens.append)
            assert result['visual_reference_unchanged_by_QA']
            assert all(torch.equal(v,before[i][k]) for i,r in cache.references.items() for k,v in r.items())
            for arm,answer in result['arms'].items():
                assert answer['generated_ids']==[t['token_id'] for t in tokens if t['arm']==arm]
                assert answer['banks_unchanged'] and answer['branch_released']
                QA.append(dict(arm=arm,input_tokens=answer['input_tokens'],visual_tokens=answer['visual_tokens'],
                               generated_tokens=len(answer['generated_ids'])))
            if index==0:
                session.append_pair(*packets[4:6]);session.snapshot_tail(packets[6])
        assert session.vision_encodings==8 and cache.full_pairs==cache.selective_pairs==2
        session.close()
        assert cache.reference_bytes()==0
    for arm in ARMS:
        bank=Banks(arm)
        for i in range(9):bank.append(torch.zeros(48,2048),i,i*2.,i*2.+1)
        accounting[arm]=bank.status()['banks'][arm]['resident_tensor_bytes']
        bank.clear()
assert len(layouts)==606 and accounting==dict.fromkeys(ARMS,3156480)
peak=getattr(env['psutil'].Process().memory_info(),'peak_wset',env['psutil'].Process().memory_info().rss)
assert peak<1536*2**20 and time.monotonic()-started<300
save(OUT/args.receipt,dict(status='passed',layouts=606,layout_rows=layouts,QA=QA,
    real_sampled_frames=7,tiny_visual_calls=8,reference_unchanged_by_QA=True,
    bank_bytes_2048=accounting,process_peak_RSS_bytes=peak,
    pretrained_loads=0,GPU_jobs=0,seconds=time.monotonic()-started,
    protocol_SHA256=sha(OUT/'inputs/protocol.json'),cacher_SHA256=sha(ROOT/'experiments/2b-vision-cache-v1/cacher.py')))
print(dict(status='passed',layouts=606,tiny_QA=4,tiny_visual_calls=8,seconds=time.monotonic()-started),flush=True)
