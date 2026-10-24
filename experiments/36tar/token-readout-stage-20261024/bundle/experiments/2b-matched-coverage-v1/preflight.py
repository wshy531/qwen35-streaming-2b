"""CPU geometry, bounded storage, legacy controls and matched layouts."""
import importlib.util
import os
import time
import torch
from common import ROOT,OUT,ARMS,read,save,sha,runtime
from banks import Banks
from features import raw_block
from dataclasses import replace
from selection import select
from online_bank import nodes_digest
from bank_readout import context,Limits


def geometry(bank):
    nodes={a:bank.export(a) for a in ARMS}
    times=lambda ns:[(n.first,n.last,n.begin,n.end,n.center) for n in ns]
    assert times(nodes['full48_8'])==times(nodes['odd24_8'])==times(nodes['change24_8'])
    for mode in ('odd24','change24'):
        assert nodes_digest(nodes[mode+'_8'])==nodes_digest(nodes[mode+'_16'][-8:])
    assert all(len(n.features)==48 for n in nodes['full48_8'])
    assert all(len(n.features)==24 for a,ns in nodes.items() if a!='full48_8' for n in ns)
    return nodes


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1);torch.manual_seed(491)
    started=time.monotonic()
    spec=importlib.util.spec_from_file_location('old_p3_bank_controls',ROOT/'experiments/2b-compressed-banks-v1/banks.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    bank=Banks();references={a:old.Recent() for a in ('full48_8','odd24_16','change24_16')}
    previous=None;regressions=0
    for i in range(96):
        features=torch.randn(48,32)
        block=raw_block(features,i,i*2.,i*2.+1)
        selected,_=select(block.features,previous)
        bank.append(features,i,i*2.,i*2.+1)
        for a,ref in references.items():
            mode=a.split('_')[0]
            b=block if mode=='full48' else replace(block,features=block.features[selected[mode]].clone(),
                mass=block.mass[selected[mode]].clone(),xy=block.xy[selected[mode]].clone())
            ref.append(b,starts_event=False)
            assert nodes_digest(bank.export(a))==nodes_digest(ref.export())
            regressions+=1
        nodes=geometry(bank)
        for mode in ('odd24','change24'):
            # Latest selected values and original coordinates match directly.
            last=nodes[mode+'_8'][-1]
            assert torch.equal(last.features,block.features[selected[mode]])
            assert torch.equal(last.coords[:,1:],block.xy[selected[mode]])
        previous=block.features.clone()
    before=bank.fingerprint();digest=bank.selector_digest()
    nodes=bank.export('full48_8');nodes[0].features.zero_()
    assert bank.fingerprint()==before
    try:bank.append(torch.zeros(48,32),98,192.,193.)
    except ValueError:pass
    else:raise AssertionError('nonconsecutive pair accepted')
    assert bank.fingerprint()==before and bank.selector_digest()==digest
    bank.clear();assert not bank.banks and bank.previous is None
    accounting=Banks()
    for i in range(20):accounting.append(torch.randn(48,2048),i,i*2.,i*2.+1)
    state=accounting.status()
    assert state['shared_selector_bytes']==393216 and state['shared_descriptor_bytes']==0
    assert state['banks']['change24_8']['visual_tokens']==192
    accounting.clear()
    env=runtime();assert not torch.cuda.is_available()
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](builder.processor),attn_implementation='sdpa')).float().eval()
    from score_memory_eval import prompt
    groups=read(OUT/'inputs/groups.json');questions=read(OUT/'inputs/questions.json');layouts=[]
    with torch.inference_mode(),env['fp32_reference']():
        for group in groups:
            bank=Banks()
            for q in sorted([q for q in questions if q['group']==group['id']],key=lambda q:(q['pairs'],q['id'])):
                while bank.pairs<q['pairs']:
                    i=bank.pairs;f,s=group['frames'][2*i],group['frames'][min(2*i+1,len(group['frames'])-1)]
                    bank.append(torch.randn(48,32),i,f['pts_s'],s['pts_s'])
                nodes=geometry(bank);before=bank.fingerprint();selector=bank.selector_digest();local={}
                for arm in ARMS:
                    emb,pos,ids,meta=context(model,builder,nodes[arm],prompt(q),limits=Limits(2048,384,16,128),max_new_tokens=8)
                    assert meta['input_tokens']+8<=2048
                    local[arm]=meta['input_tokens']
                    layouts.append(dict(id=q['id'],arm=arm,input_tokens=meta['input_tokens'],visual_tokens=meta['visual_tokens']))
                assert local['odd24_8']==local['change24_8']==local['full48_8']-24*min(q['pairs'],8)
                assert local['odd24_16']==local['change24_16']
                assert bank.fingerprint()==before and bank.selector_digest()==selector
            bank.clear()
    assert len(layouts)==1515 and time.monotonic()-started<300
    save(OUT/'CPU_preflight.json',dict(status='passed',legacy_step_regressions=regressions,layouts=len(layouts),
        max_input_tokens=max(r['input_tokens'] for r in layouts),layout_rows=layouts,
        matched_times_and_pair_ids=True,selected_values_and_xy_exact=True,accounting_2048=state,
        export_isolation_and_clear=True,pretrained_loads=0,GPU_jobs=0,seconds=time.monotonic()-started,
        protocol_sha256=sha(OUT/'inputs/protocol.json')))
    print(dict(status='passed',layouts=len(layouts),legacy_checks=regressions,seconds=time.monotonic()-started),flush=True)


if __name__=='__main__':main()
