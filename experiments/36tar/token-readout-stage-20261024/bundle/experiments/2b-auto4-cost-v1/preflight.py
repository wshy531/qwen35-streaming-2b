"""CPU: active-bank isolation, old readout equivalence, all909 layouts."""
from dataclasses import replace
import importlib.util,os,time
import torch
from common import ROOT,OUT,ARMS,read,save,sha,runtime
from banks import Banks
from features import raw_block
from selection import select
from online_bank import nodes_digest
from bank_readout import context,Limits


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1);torch.manual_seed(631);started=time.monotonic()
    spec=importlib.util.spec_from_file_location('legacy_recent_controls',ROOT/'experiments/2b-compressed-banks-v1/banks.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    checks=0;accounting={}
    for arm in ARMS:
        bank=Banks(arm);reference=old.Recent();previous=None
        for i in range(96):
            features=torch.randn(48,32);block=raw_block(features,i,i*2.,i*2.+1)
            chosen,_=select(block.features,previous);mode=arm.split('_')[0]
            selected=block if mode=='full48' else replace(block,features=block.features[chosen[mode]].clone(),mass=block.mass[chosen[mode]].clone(),xy=block.xy[chosen[mode]].clone())
            reference.append(selected,starts_event=False);bank.append(features,i,i*2.,i*2.+1)
            assert nodes_digest(bank.export(arm))==nodes_digest(reference.export()[-8:]);checks+=1
            assert set(bank.banks)=={arm}
            assert (bank.previous is None)==(arm!='change24_8')
            previous=block.features.clone()
        before=bank.fingerprint();bank.export(arm)[0].features.zero_();assert bank.fingerprint()==before
        bank.clear();assert not bank.banks and bank.previous is None
        bank=Banks(arm)
        for i in range(12):bank.append(torch.randn(48,2048),i,i*2.,i*2.+1)
        accounting[arm]=bank.status()['banks'][arm]
        bank.clear()
    assert accounting['full48_8']['standalone_persistent_tensor_bytes']==3156480
    assert accounting['odd24_8']['standalone_persistent_tensor_bytes']==1578240
    assert accounting['change24_8']['standalone_persistent_tensor_bytes']==1971456
    env=runtime();assert not torch.cuda.is_available()
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](builder.processor),attn_implementation='sdpa')).float().eval()
    from score_memory_eval import prompt
    groups=read(OUT/'inputs/groups.json');questions=read(OUT/'inputs/questions.json');jobs=read(OUT/'inputs/jobs.json')
    assert len(jobs)==6 and sum(j['questions'] for j in jobs)==228
    for arm in ARMS:assert sorted(g for j in jobs if j['arm']==arm for g in j['groups'])==sorted([g['id'] for g in groups]*2)
    old_rows=read(ROOT/'runs/2b_matched_coverage_prepare_20261023_01/CPU_preflight.json')['layout_rows']
    expected={(r['id'],r['arm']):r for r in old_rows if r['arm'] in ARMS};layouts=[]
    with torch.inference_mode(),env['fp32_reference']():
        for group in groups:
            for arm in ARMS:
                bank=Banks(arm)
                for q in sorted([q for q in questions if q['group']==group['id']],key=lambda q:(q['pairs'],q['id'])):
                    while bank.pairs<q['pairs']:
                        i=bank.pairs;f,s=group['frames'][2*i],group['frames'][min(2*i+1,len(group['frames'])-1)]
                        bank.append(torch.randn(48,32),i,f['pts_s'],s['pts_s'])
                    before=bank.fingerprint();selector=bank.selector_digest()
                    _,_,_,meta=context(model,builder,bank.export(arm),prompt(q),limits=Limits(2048,384,8,128),max_new_tokens=8)
                    row=dict(id=q['id'],arm=arm,input_tokens=meta['input_tokens'],visual_tokens=meta['visual_tokens'])
                    assert row==expected[q['id'],arm] and before==bank.fingerprint() and selector==bank.selector_digest()
                    layouts.append(row)
                bank.clear()
    assert len(layouts)==114 and time.monotonic()-started<300
    save(OUT/'CPU_preflight.json',dict(status='passed',layouts=114,legacy_step_regressions=checks,
        layout_rows=layouts,accounting_2048=accounting,one_active_bank=True,no_unused_previous_state=True,
        all8_groups_twice_per_arm=True,all6_jobs_checked=True,pretrained_loads=0,GPU_jobs=0,
        seconds=time.monotonic()-started,protocol_sha256=sha(OUT/'inputs/protocol.json')))
    print(dict(status='passed',layouts=114,old_regressions=checks,seconds=time.monotonic()-started),flush=True)


if __name__=='__main__':main()
