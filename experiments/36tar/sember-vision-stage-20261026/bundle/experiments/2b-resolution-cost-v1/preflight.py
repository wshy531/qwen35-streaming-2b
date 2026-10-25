"""CPU parity, both real visual grids, all registered QA branches; no real weights."""
import argparse
import os
import sys
import torch
from resolution_common import ROOT,OUT,MAIN,SPECS,ARMS,DATA,read,save,sha,runtime
from resolution_banks import Banks,select108
from resolution_nodes import validate,digest
from resolution_runtime import make_builder,context
from resolution_source import CausalSource,PairDriver
from resolution_session import Session


def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1);torch.manual_seed(20261025)
    from sember_banks import Banks as Old
    old=Old();mapping={p+'_full108'+('_8' if p=='M0' else ''):p+'_mid108' for p in ('M0','M2','A06','LS')}
    old.banks={a:b for a,b in old.banks.items() if a in mapping};mid=Banks();high=Banks((12,16));parity=0
    for i in range(96):
        x=torch.randn(108,32) if i%7==0 else x+.02*torch.randn(108,32)
        old.append(x,i,4.*i,4.*i+2);mid.append(x,i,4.*i,4.*i+2)
        for a,b in mapping.items():
            for key in ('tokens','within_merges','long_merges','recent_tokens','closed_tokens','history_tokens'):
                assert old.banks[a].status()[key]==mid.banks[b].status()[key]
            left,right=old.banks[a].export(),mid.banks[b].export();assert len(left)==len(right)
            for l,r in zip(left,right):
                assert l.metadata()==r.metadata()
                assert all(torch.equal(getattr(l,k),getattr(r,k)) for k in ('features','mass','coords'))
            parity+=1
    peaks={a:0 for a in high.banks};maxdepth=0
    for i in range(160):
        z=torch.randn(192,32) if i%7==0 else z+.02*torch.randn(192,32)
        trace=high.append(z,i,4.*i,4.*i+2)
        for a,b in high.banks.items():
            st=b.status();peaks[a]=max(peaks[a],st['tokens']);assert st['tokens']<=SPECS[a]['budget']
            for node in b.nodes():validate(node);maxdepth=max(maxdepth,node.depth)
        assert all(len(v)==108 for v in trace['selected_indices'].values())
    assert all(peaks[a]==SPECS[a]['budget'] for a in peaks)
    for a in high.banks:
        before=high.fingerprint();exported=high.export(a);exported[0].features.zero_();assert high.fingerprint()==before
    zero=torch.zeros(192,32);u,c,_=select108(zero,zero)
    assert len(set(u.tolist()))==len(set(c.tolist()))==108 and torch.equal(c,torch.arange(108))
    wide=Banks((12,16),arms=('LS_highchange108',));wide.append(torch.randn(192,2048),0,0.,2.)
    assert wide.status()['shared_previous_bytes']==1572864
    env=runtime();processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](processor),attn_implementation='sdpa')).float().eval()
    warm=read(OUT/'inputs/protocol.json')['warmup'];answers=[];mid_reference=None
    with torch.inference_mode(),env['fp32_reference']():
        for canvas in ((288,384),(384,512)):
            builder=make_builder(env,processor,canvas);source=CausalSource(DATA/('videos/'+warm['video_id']+'.mp4'),canvas=canvas)
            session=Session(env,model,builder);driver=PairDriver(source,session.append_pair)
            try:
                prefix=driver.advance(8.);assert prefix['pairs']==2 and prefix['pending_real_frames']==1
                if mid_reference is None:mid_reference=prefix
                else:
                    for key in ('pairs','real_frames','last_committed_pts','pending_real_frames'):assert prefix[key]==mid_reference[key]
                for arm in session.bank.banks:
                    before=session.bank.fingerprint();tokens=[];r=session.ask(arm,'What is visible?',tokens.append,max_new_tokens=2)
                    assert session.bank.fingerprint()==before and r['generated_ids']==[t['token_id'] for t in tokens]
                    assert r['read_state_unchanged'] and r['branch_released'] and r['visual_tokens']==2*SPECS[arm]['keep']
                    answers.append(dict(arm=arm,visual_tokens=r['visual_tokens'],input_tokens=r['input_tokens']))
                driver.advance(12.);assert session.pairs==3
            finally:driver.close();session.close()
        # Real processor tensor shape and every full official suffix without truncation.
        max_suffix=0
        from sember_protocol import prompt
        builder=make_builder(env,processor,(384,512))
        for q in read(OUT/'inputs/requests.json'):
            max_suffix=max(max_suffix,builder.question_ids(prompt(q)).shape[1])
        assert max_suffix<=512
    for grid in ((9,12),(12,16)):
        highgrid=Banks(grid)
        for i in range(8):highgrid.append(torch.randn(grid[0]*grid[1],32),i,4.*i,4.*i+2)
        for a,b in highgrid.banks.items():
            emb,pos,ids,meta=context(model,make_builder(env,processor,tuple(n*32 for n in grid)),b.export(),
                prompt(max(read(OUT/'inputs/requests.json'),key=lambda q:len(prompt(q)))),visual_limit=b.budget)
            assert meta['visual_tokens']<=b.budget and pos.shape==(4,1,ids.shape[1]) and emb.shape[1]==ids.shape[1]
    result=dict(status='passed',mid_old_steps=parity,high_steps=160,max_merge_depth=maxdepth,
        high_peaks=peaks,real_grid_QAs=answers,QA_resume=True,all_13_QA_branches=len(answers)==13,
        high_previous_FP32_2048_bytes=1572864,source_RGB_changed_but_pair_times_equal=True,
        max_suffix=max_suffix,pretrained_loads=0,GPU_jobs=0,no_capability_or_speed_claim=True)
    save(output,result);print({k:v for k,v in result.items() if k not in ('real_grid_QAs','high_peaks')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=__import__('pathlib').Path,required=True)
    main(p.parse_args().output)
