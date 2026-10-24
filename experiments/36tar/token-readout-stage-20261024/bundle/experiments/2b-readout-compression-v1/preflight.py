"""CPU legacy bank equivalence, ephemeral pruning and all1818 layouts."""
import importlib.util,os,time
from dataclasses import replace
from collections import deque
import torch
from common import ROOT,OUT,POLICIES,ARMS,read,save,sha,runtime
from banks import Banks,half_nodes
from features import raw_block
from policy_adapters import pair_node
from online_bank import nodes_digest
from bank_readout import context,Limits

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1);torch.manual_seed(631);started=time.monotonic()
    spec=importlib.util.spec_from_file_location('original_P3_banks',ROOT/'experiments/2b-compressed-banks-v1/banks.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.ARMS=tuple(p+'_full48' for p in POLICIES)
    bank=Banks();reference=old.Banks();checks=read_checks=0;history=deque(maxlen=8);m0_checks=0
    for i in range(128):
        # Repeated visual events plus changes exercise within/between-event merges.
        torch.manual_seed(100+i//7);features=torch.randn(48,32)+i*.0001
        bank.append(features,i,i*2.,i*2.+1);reference.append(features,i,i*2.,i*2.+1)
        for p in POLICIES:
            full=bank.export(p+'_full48');assert nodes_digest(full)==nodes_digest(reference.export(p+'_full48'));checks+=1
            before=bank.fingerprint();selected=bank.export(p+'_read24')
            assert len(full)==len(selected)
            for a,b in zip(full,selected):
                assert torch.equal(b.features,a.features[1::2]) and torch.equal(b.mass,a.mass[1::2]) and torch.equal(b.coords,a.coords[1::2])
                assert (a.first,a.last,a.begin,a.end,a.center,a.groups,a.depth)==(b.first,b.last,b.begin,b.end,b.center,b.groups,b.depth)
                assert b.features.data_ptr()!=a.features.data_ptr();read_checks+=1
            assert sum(len(n.features) for n in selected)*2==sum(len(n.features) for n in full)
            selected[0].features.zero_();assert bank.fingerprint()==before
        block=raw_block(features,i,i*2.,i*2.+1);history.append(pair_node(block))
        late=half_nodes(list(history))
        early=[pair_node(replace(raw_block(n.features,n.first,n.begin,n.end),
             features=n.features[1::2].clone(),mass=n.mass[1::2].clone(),xy=n.coords[1::2,1:].clone())) for n in history]
        assert nodes_digest(late)==nodes_digest(early);m0_checks+=1
    bad=replace(bank.export('M2_full48')[0],features=bank.export('M2_full48')[0].features[:47],
        mass=bank.export('M2_full48')[0].mass[:47],coords=bank.export('M2_full48')[0].coords[:47])
    try:half_nodes([bad])
    except ValueError:pass
    else:raise AssertionError('odd layout accepted')
    bank.clear();reference.clear()
    env=runtime();assert not torch.cuda.is_available()
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](builder.processor),attn_implementation='sdpa')).float().eval()
    from score_memory_eval import prompt
    groups=read(OUT/'inputs/groups.json');questions=read(OUT/'inputs/questions.json');layouts=[]
    with torch.inference_mode(),env['fp32_reference']():
        for g in groups:
            b=Banks();ref=old.Banks()
            for q in sorted([q for q in questions if q['group']==g['id']],key=lambda q:(q['pairs'],q['id'])):
                while b.pairs<q['pairs']:
                    i=b.pairs;a=g['frames'][2*i];z=g['frames'][min(2*i+1,len(g['frames'])-1)]
                    torch.manual_seed(400+i//5);f=torch.randn(48,32)
                    b.append(f,i,a['pts_s'],z['pts_s']);ref.append(f,i,a['pts_s'],z['pts_s'])
                before=b.fingerprint();selector=b.selector_digest()
                for arm in ARMS:
                    nodes=b.export(arm)
                    _,pos,ids,meta=context(model,builder,nodes,prompt(q),limits=Limits(2048,384,16,128),max_new_tokens=8)
                    assert meta['visual_tokens']<=(384 if arm.endswith('full48') else 192)
                    if arm.endswith('full48'):
                        _,rpos,rids,rmeta=context(model,builder,ref.export(arm),prompt(q),limits=Limits(2048,384,16,128),max_new_tokens=8)
                        assert torch.equal(pos,rpos) and torch.equal(ids,rids) and meta==rmeta
                    layouts.append(dict(id=q['id'],arm=arm,input_tokens=meta['input_tokens'],visual_tokens=meta['visual_tokens'],
                        input_ids_sha256=sha_tensor(ids),storage_node_digest=before[arm.split('_')[0]]))
                assert before==b.fingerprint() and selector==b.selector_digest()
            b.clear();ref.clear()
    assert len(layouts)==1818 and time.monotonic()-started<240
    save(OUT/'CPU_preflight.json',dict(status='passed',layouts=1818,layout_rows=layouts,
        legacy_step_regressions=checks,readout_checks=read_checks,M0_equivalence_checks=m0_checks,
        main_bank_unchanged=True,no_training=True,pretrained_loads=0,GPU_jobs=0,
        seconds=time.monotonic()-started,protocol_sha256=sha(OUT/'inputs/protocol.json')))
    print(dict(status='passed',layouts=1818,legacy_steps=checks,readout_checks=read_checks,M0_equivalence_checks=m0_checks),flush=True)

def sha_tensor(t):
    import hashlib
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()

if __name__=='__main__':main()
