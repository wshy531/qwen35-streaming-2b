import argparse,hashlib,json,os
from main_train import *
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    env=runtime();torch.set_num_threads(1);torch.manual_seed(20261102)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=24
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
    builder=make_builder(env,processor);scorer=GroupScorer(model.config.vision_config.hidden_size)
    train=read(OLD_PREP/'train.json');evaluation=read(OLD_PREP/'eval.json')
    audit={r['video_id']:r for r in read(DATA/'CPU_media_audit.json')['rows']}
    digest=hashlib.sha256();layouts=0;maximum=0
    for q in train+evaluation:
        for count in (108,54):
            samples=audit[q['video_id']]['samples'];records=[]
            ids=torch.arange(108) if count==108 else torch.arange(1,108,2)
            for i in range(0,len(samples),2):
                records.append((i//2,samples[i]['pts_s'],samples[min(i+1,len(samples)-1)]['pts_s'],
                                torch.zeros(count,model.config.text_config.hidden_size),ids))
            for shift in ([q['shift']] if q['role']=='train' else [0,2]):
                rq=rotate(q,shift);_,pos,tokens=context(model,builder,records,question_text(rq))
                assert tokens.shape[1]+8<=4096 and rq['options'][rq['answer_index']]==q['options'][q['answer_index']]
                digest.update(tokens.numpy().tobytes());digest.update(pos.numpy().tobytes());layouts+=1
                maximum=max(maximum,tokens.shape[1])
    assert layouts==5082
    from transformers.feature_extraction_utils import BatchFeature
    batch=BatchFeature(dict(pixel_values_videos=torch.randn(432,1536),video_grid_thw=torch.tensor([[1,18,24]])))
    pairs=[(0.,.5,batch)];q=evaluation[0];qa=[];gradients=[]
    with torch.no_grad(),env['fp32_reference']():
        for arm in BASELINES+ARMS:
            records=old_worker.records_for(model,pairs,arm,None) if arm in BASELINES else records_for(model,pairs,scorer,'gather')
            result=old_worker.answer(env,model,builder,records,q,limit=1)
            qa.append(dict(arm=arm,generated=len(result['generated_ids'])))
    options=option_ids(builder,torch.device('cpu'));initial={n:p.detach().clone() for n,p in scorer.state_dict().items()}
    for arm in ('ce','ce_kl'):
        scorer.load_state_dict(initial);optimizer=torch.optim.AdamW(scorer.parameters(),lr=1e-4,weight_decay=0.)
        optimizer.zero_grad(set_to_none=True)
        with env['fp32_reference']():
            records=records_for(model,pairs,scorer,'mask');logits,_=last_logits(model,builder,records,q)
            teacher=torch.zeros(1,5);teacher[0,q['answer_index']]=2
            loss,_,_,_=loss_parts(logits,options,q['answer_index'],teacher if arm=='ce_kl' else None);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(scorer.parameters(),1.);assert torch.isfinite(norm) and norm>0
        optimizer.step();assert all(p.grad is None for p in model.parameters())
        assert any(not torch.equal(initial[n],p) for n,p in scorer.state_dict().items())
        gradients.append(dict(arm=arm,gradient=float(norm)))
    out=dict(status='passed_random_tiny_only',layouts=layouts,layout_sha256=digest.hexdigest(),maximum_tokens=maximum,
             micro_QA=qa,two_objective_updates=gradients,GPU=0,pretrained_loads=0)
    save(a.out,out);print(json.dumps(out))
if __name__=='__main__':main()
