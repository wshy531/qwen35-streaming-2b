"""Bounded state regression and all P3 layouts; CPU only, no pretrained model."""
import os,time,importlib.util,json
from pathlib import Path
import torch
from common import ROOT,OUT,ARMS,read,save,runtime,sha
from banks import Banks
from policy_adapters import BoundedForest,BoundedShared
from memory384 import Forest384
from policy import DepthForest384
from shared import SharedHistory384
from features import raw_block
from online_bank import nodes_digest
from bank_readout import context,Limits

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1);torch.manual_seed(321)
    started=time.monotonic();checks=0
    for scenario in ('one_event','every_pair','mixed'):
        refs=[Forest384(),DepthForest384(),SharedHistory384()]
        candidates=[BoundedForest(),BoundedForest(True),BoundedShared()]
        banks=Banks()
        for i in range(96):
            flag=i==0 or scenario=='every_pair' or scenario=='mixed' and i%7==0
            x=torch.randn(48,32);block=raw_block(x,i,float(i*2),float(i*2+1))
            for ref,candidate in zip(refs,candidates):
                ref.append(block,starts_event=flag);candidate.append(block,starts_event=flag)
                assert nodes_digest(ref.export())==nodes_digest(candidate.export())
                checks+=1
            trace=banks.append(x,i,float(i*2),float(i*2+1))
            assert trace['selected_indices']['change24']==sorted(trace['selected_indices']['change24'])
            for arm in ARMS:
                state=banks.status()['banks'][arm]
                assert state['visual_tokens']<=384
                assert len(banks.export(arm))<=16
                if not arm.startswith('M0'):assert state['details']['recent_tokens']<=192
                before=banks.fingerprint();digest=banks.selector_digest()
                nodes=banks.export(arm);nodes[0].features.zero_()
                assert before==banks.fingerprint() and digest==banks.selector_digest()
            assert time.monotonic()-started<180
        banks.clear();assert not banks.banks and banks.previous is None
    print(dict(stage='storage_regressions_passed',checks=checks,seconds=time.monotonic()-started),flush=True)
    # Actual 2048-dimensional accounting and constant-state retention.
    banks=Banks()
    for i in range(32):banks.append(torch.randn(48,2048),i,float(i*2),float(i*2+1))
    status=banks.status();assert status['shared_selector_bytes']==393216
    assert status['shared_descriptor_bytes']==8192
    assert status['banks']['M0_change24']['visual_tokens']==384
    assert status['banks']['M0_full48']['visual_tokens']==384
    assert len(banks.export('M0_change24'))==16 and len(banks.export('M0_full48'))==8
    banks.clear()
    # Reject an incorrect pair size before changing an individual forest.
    one=BoundedForest(pair_tokens=24);block=raw_block(torch.ones(48,32),0,0.,1.)
    try:one.append(block,starts_event=True)
    except ValueError:pass
    else:raise AssertionError('mixed pair size accepted')
    assert not one.nodes()
    env=runtime();assert not torch.cuda.is_available()
    build=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](build.processor),attn_implementation='sdpa')).float().eval()
    print(dict(stage='tiny_model_created',seconds=time.monotonic()-started),flush=True)
    source=ROOT/'runs/2b_temporal24_prepare_20261023_01/inputs'
    groups=read(source/'groups.json');questions=read(source/'questions.json');layouts=[]
    from score_memory_eval import prompt
    with torch.inference_mode(),env['fp32_reference']():
        for g in groups:
            banks=Banks()
            for q in sorted([q for q in questions if q['group']==g['id']],key=lambda q:(q['pairs'],q['id'])):
                while banks.pairs<q['pairs']:
                    i=banks.pairs;first,second=g['frames'][i*2],g['frames'][min(i*2+1,len(g['frames'])-1)]
                    banks.append(torch.randn(48,32),i,first['pts_s'],second['pts_s'])
                before=banks.fingerprint();selector=banks.selector_digest()
                for arm in ARMS:
                    emb,pos,ids,meta=context(model,build,banks.export(arm),prompt(q),limits=Limits(2048,384,16,128),max_new_tokens=8)
                    assert meta['visual_tokens']<=384 and meta['input_tokens']+8<=2048
                    layouts.append(dict(id=q['id'],arm=arm,input_tokens=meta['input_tokens'],visual_tokens=meta['visual_tokens']))
                assert before==banks.fingerprint() and selector==banks.selector_digest()
            banks.clear()
            assert time.monotonic()-started<300
            if len(layouts)%90==0:print(dict(stage='layouts',done=len(layouts)),flush=True)
    assert len(layouts)==2727
    receipt=dict(status='passed',full48_step_regressions=checks,storage_scenarios=3,
        layouts=len(layouts),max_input_tokens=max(x['input_tokens'] for x in layouts),
        accounting_2048=status,export_and_query_preparation_unchanged=True,
        old_merge_and_boundary_unchanged=True,pretrained_loads=0,GPU_jobs=0,
        seconds=time.monotonic()-started,layout_rows=layouts,
        protocol_sha256=sha(OUT/'inputs/protocol.json'))
    save(OUT/'CPU_preflight.json',receipt)
    print({k:v for k,v in receipt.items() if k not in ('accounting_2048','layout_rows')})

if __name__=='__main__':main()
