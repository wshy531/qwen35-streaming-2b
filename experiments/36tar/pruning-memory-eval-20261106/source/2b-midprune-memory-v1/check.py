"""CPU mechanism only: old full-bank parity, sparse provenance, bounds and QA isolation."""
import argparse,json,sys
from adapter import *
sys.path.insert(0,str(ROOT/'experiments/2b-prune-checkpoints-v1'))
from worker import runtime,make_builder,records_for,old_worker,GroupScorer
from resolution_banks import Banks
from resolution_nodes import pair_node

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1);torch.manual_seed(20261102)
    memories={v:Memory(v) for v in VIEWS};old=Banks((9,12),arms=tuple(p+'_mid108' for p in POLICIES))
    regressions=0;checks=0
    for i in range(48):
        x=torch.randn(108,16)+(i//6)*.2
        ids=torch.arange(108);old.append(x,i,float(2*i),float(2*i+1))
        for v,m in memories.items():
            chosen=ids if v=='full' else torch.arange(1,108,2)
            n=sparse_pair(x[chosen],chosen,i,float(2*i),float(2*i+1))
            assert digest([n])==digest([pair_node(x,i,float(2*i),float(2*i+1),(9,12),chosen)])
            m.append(x[chosen],chosen,i,float(2*i),float(2*i+1));checks+=1
            if v=='full':
                for policy in POLICIES:
                    assert digest(m.export(policy))==digest(old.export(policy+'_mid108'));regressions+=1
            for policy,bank in m.banks.items():
                status=bank.status();assert status['tokens']<=bank.budget and status['metadata_bytes']<=65536
                assert all(n.last<=i and n.end<=2*i+1 for n in bank.nodes())
                exported=m.export(policy);fingerprint=m.fingerprint();exported[0].features.zero_();assert m.fingerprint()==fingerprint
                if policy=='M0_same8':assert [n.first for n in m.export(policy)]==list(range(max(0,i-7),i+1))
        if i>=15:
            assert len(memories['full'].export('M0'))==8
            assert len(memories['fixed12'].export('M0'))==16
    invalid=0
    for bad in (torch.zeros(54,dtype=torch.long),torch.arange(54)+80,torch.arange(54).flip(0)):
        try:sparse_pair(torch.zeros(54,16),bad,0,0.,1.)
        except ValueError:invalid+=1
    assert invalid==3
    env=runtime();proc=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](proc);cfg['vision_config']['depth']=24
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
    builder=make_builder(env,proc);head=GroupScorer(model.config.vision_config.hidden_size)
    # Real model architecture and patched visual paths with random tiny weights, not trained capability.
    from transformers.feature_extraction_utils import BatchFeature
    batch=BatchFeature(dict(pixel_values_videos=torch.randn(432,1536),video_grid_thw=torch.tensor([[1,18,24]])))
    tiny={v:Memory(v) for v in VIEWS};qa=[]
    with torch.no_grad(),env['fp32_reference']():
        for v,m in tiny.items():
            rec=records_for(model,[(0.,.5,batch)],head,'gather') if v=='mask_ce' else old_worker.records_for(model,[(0.,.5,batch)],v,None)
            _,begin,end,feat,chosen=rec[0]
            for i in range(18):m.append(feat,chosen,i,float(i*2),float(i*2+1))
            for policy in m.banks:
                r=ask(env,model,builder,m,policy,'What is visible now? Answer briefly.',limit=1)
                assert r['read_state_unchanged'] and r['branch_released'];qa.append(dict(view=v,policy=policy,visual_tokens=r['visual_tokens'],input_tokens=r['input_tokens']))
            # Query does not invalidate continued causal appends.
            m.append(feat,chosen,18,36.,37.);m.close();assert m.previous_descriptor is None and not m.banks
    assert len(qa)==19
    for m in memories.values():m.close();assert not m.banks and m.previous_descriptor is None
    result=dict(passed=True,old_full_bank_parity=regressions,sparse_pair_checks=checks,invalid_coordinates_rejected=invalid,
        micro_QA=qa,causal_bounds_export_isolation=True,query_continuation_close=True,pretrained_loads=0,GPU=0,
        limitation='Random tiny/CPU mechanism evidence only; no capability or speed claim')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
