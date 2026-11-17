import argparse,sys
from common import *
from native_cpu import make_tiny
from transformers import AutoProcessor

def main(out):
    torch.set_num_threads(1);torch.manual_seed(7110)
    model=make_tiny(real_tokens=True).float().eval().requires_grad_(False)
    p=AutoProcessor.from_pretrained((ROOT/'runs/7b_transfer_prepare_20261107_01/processor_assets' if (ROOT/'runs/7b_transfer_prepare_20261107_01/processor_assets').exists() else ROOT/'models/Qwen2.5-VL-7B-Instruct'),local_files_only=True)
    frames=torch.randint(0,256,(4,280,392,3),dtype=torch.uint8).numpy()
    memories={mode:Memory(mode,archive=mode=='full') for mode in ('full','fixed','content')}
    results=[]
    with torch.no_grad():
        for i in range(2):
            batch=readout.pair_inputs(p,frames[2*i:2*i+2])
            official=model.get_video_features(**batch,return_dict=True).pooler_output[0]
            _,_,shared=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],'full',capture_prefix=True)
            for mode,m in memories.items():
                import copy
                duplicate=copy.deepcopy(m.scorer)
                value,ids,trace=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],mode,m.scorer,trace=True)
                cached,cached_ids,_=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],mode,duplicate,prefix=shared['prefix'])
                torch.testing.assert_close(value,cached,atol=0,rtol=0);assert torch.equal(ids,cached_ids)
                if mode=='full':torch.testing.assert_close(value,official,atol=0,rtol=0)
                assert all(x['tokens']==560 for x in trace['layers'][:16])
                assert all(x['tokens']==(560 if mode=='full' else 280) for x in trace['layers'][16:])
                m.append(value,ids,i,2.*i,2.*i+1)
                results.append(dict(pair=i,mode=mode,shape=list(value.shape),ids=ids.tolist(),trace=trace))
            # Post-only fixed selection must exactly match official features.
            post,ids,_=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],'post_fixed')
            torch.testing.assert_close(post,official[ids],atol=0,rtol=0)
        q=read(PREP/'evaluation.json')[0];answers=[]
        for mode,m in memories.items():
            nodes=m.select('recent8');emb,pos,ids=readout.context(model,p,nodes,readout.question(q,0,'B'))
            full=model.lm_head(model.model.language_model(inputs_embeds=emb,position_ids=pos,attention_mask=torch.ones(emb.shape[:2],dtype=torch.long),use_cache=False).last_hidden_state[:,-1:])[0,-1]
            generated,chunk=readout.generate_context(model,emb,pos,limit=2)
            torch.testing.assert_close(full,chunk,atol=1e-6,rtol=1e-5)
            answers.append(dict(mode=mode,ids=generated,token_sha=tsha(ids),pos_sha=tsha(pos),max_abs=float((full-chunk).abs().max())))
        fullnodes=memories['full'].select('recent8');ids,pos,_=readout.layout(p,fullnodes,readout.question(q,0,'B'))
        native=readout.native_inputs(p,frames,readout.question(q,0,'B'))
        expected,_=model.model.get_rope_index(native['input_ids'],video_grid_thw=native['video_grid_thw'],second_per_grid_ts=native['second_per_grid_ts'])
        assert torch.equal(ids,native['input_ids']) and torch.equal(pos,expected)
        from run_prune import audit
        audit_q=next(q for q in read(PREP/'evaluation.json') if q['id'] in read(PREP/'audit_ids.json'))
        audit_result=audit(model,p,memories['full'],audit_q,frames,out.parent,limit=2)
        assert audit_result['passed'],audit_result
    # The head recipe is dimension-adapted rather than padding a 2B checkpoint.
    from selector import Head,descriptor_features,choices_mask
    bank=Memory();
    for i in range(12):bank.append(torch.zeros(140,64),torch.arange(140),i,2*i,2*i+1)
    recent,history=bank.parts();f=descriptor_features(torch.zeros(64),recent,history)
    h=Head(64);choices,mask=choices_mask(len(history));values=h(*(f[k][None] for k in ('query','recent','history','metadata','node_mask')),mask)
    assert values.shape==(1,16) and float(values[0,0])==0
    values[:,1:].square().sum().backward();assert any(p.grad is not None for p in h.parameters())
    # Verify the modified head is algebraically the OLD head at the OLD width.
    from utility_head import HistoryUtilityHead
    torch.manual_seed(7);old=HistoryUtilityHead();new=Head(2048);new.load_state_dict(old.state_dict())
    choices,mask=choices_mask(4)
    inputs=[torch.randn(2,2048),torch.randn(2,4096),torch.randn(2,4,4096),torch.randn(2,4,8),torch.ones(2,4,dtype=torch.bool),mask.expand(2,-1,-1)]
    for blind in (False,True):torch.testing.assert_close(old(*inputs,blind_history=blind),new(*inputs,blind_history=blind),atol=0,rtol=0)
    result=dict(passed=True,GPU=0,model_weights='random tiny; official geometry/depth',vision=results,micro_answers=answers,official_positions_exact=True,native_readout_audit=audit_result,head_recipe_exact_at_old_width=True,head_params_7b=sum(p.numel() for p in Head(3584).parameters()))
    save(out,result);print(json.dumps(dict(passed=True,vision_checks=len(results),micro_answers=answers,head_params=result['head_params_7b'])))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);main(ap.parse_args().out)
