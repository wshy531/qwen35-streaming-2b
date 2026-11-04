"""Fixed one-epoch real experiment; requires a separate explicit GPU authorization."""
import argparse
import gc
from pathlib import Path
import time
import torch
from torch import nn
from transformers import AutoProcessor,VideoMAEImageProcessor,VideoMAEForVideoClassification,Qwen3_5ForConditionalGeneration
from common import (ROOT,PREP,STUDENT,TEACHER,LABELS,read,save,sha,decode,supervised,
    generation_inputs,attach,adapter_state,restore,training_forward,prompt,official_identity)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=False,parents=True)
    started=time.monotonic();torch.set_num_threads(4);torch.manual_seed(20261026)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    protocol=read(PREP/'main_protocol.json')
    lock=read(PREP/'main_execution_lock.json')
    for name,digest in lock['files_sha256'].items():assert sha(ROOT/name)==digest,name
    rows=read(PREP/'media_frames_LOCAL_ONLY.json');by_id={r['id']:r for r in rows}
    train=[by_id[i] for i in protocol['train_order']]
    assert len(train)==653 and all(r['split']=='train' for r in train)
    processor=AutoProcessor.from_pretrained(STUDENT,local_files_only=True)
    ids=[processor.tokenizer(s,add_special_tokens=False).input_ids for s in LABELS]
    assert all(len(i)==1 for i in ids);candidate_ids=torch.tensor([i[0] for i in ids],device='cuda')
    teacher_processor=VideoMAEImageProcessor.from_pretrained(TEACHER,local_files_only=True)
    teacher=VideoMAEForVideoClassification.from_pretrained(TEACHER,local_files_only=True,
        dtype=torch.float32,device_map={'':'cuda:0'},attn_implementation='eager').eval().requires_grad_(False)
    features={}
    for index,row in enumerate(train):
        if time.monotonic()-started>84*60:raise TimeoutError('bounded worker time exceeded')
        rgb=decode(row);pixels=teacher_processor(rgb,return_tensors='pt')['pixel_values'].to('cuda')
        with torch.inference_mode():feature=teacher.fc_norm(teacher.videomae(pixel_values=pixels).last_hidden_state.mean(1))
        features[row['id']]=feature.detach().cpu().clone()
        if (index+1)%100==0:print(dict(phase='teacher',done=index+1),flush=True)
        del rgb,pixels,feature
    torch.save(features,out/'teacher_features_train_only.pt')
    del teacher;gc.collect();torch.cuda.empty_cache()
    net=Qwen3_5ForConditionalGeneration.from_pretrained(STUDENT,local_files_only=True,
        dtype=torch.float32,device_map={'':'cuda:0'},attn_implementation='eager')
    save(out/'startup.json',dict(status='real_main_started',identity=official_identity(net),protocol=protocol))
    predictions=[];training={}
    def evaluate(arm):
        net.eval()
        for mode in ('video','text_only'):
            selected=[r for r in rows if r['split'] in (('validation','test') if mode=='video' else ('test',))]
            for r in selected:
                if time.monotonic()-started>84*60:raise TimeoutError('bounded worker time exceeded')
                if mode=='video':
                    rgb=decode(r);batch,boundary=supervised(processor,r,rgb);batch=generation_inputs(batch,boundary)
                    del rgb
                else:
                    rendered=processor.apply_chat_template([dict(role='user',content=[dict(type='text',text=prompt(r))])],
                        tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    batch=dict(processor.tokenizer(rendered,add_special_tokens=False,return_tensors='pt'))
                batch={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                with torch.inference_mode():logits=net(**batch,use_cache=False,logits_to_keep=1).logits[0,-1]
                assert torch.isfinite(logits).all()
                scores=logits[candidate_ids].float().cpu().tolist()
                predictions.append(dict(id=r['id'],split=r['split'],arm=arm,mode=mode,
                    label=r['label'],predicted=LABELS[max(range(7),key=lambda k:scores[k])],scores=scores,
                    unconstrained_greedy_token_id=int(logits.argmax())))
                del batch,logits
        save(out/(arm+'_predictions.json'),predictions[-526:])
        print(dict(phase='evaluation_done',arm=arm),flush=True)
    evaluate('original')
    targets=attach(net);initial=adapter_state(net)
    for arm in ('task_only','expert_aligned','expert_shuffled'):
        restore(net,initial);torch.manual_seed(protocol['seed'])
        head=nn.Linear(net.config.text_config.hidden_size,768,bias=False,device='cuda')
        params=[p for p in net.parameters() if p.requires_grad]
        if arm!='task_only':params+=list(head.parameters())
        optimizer=torch.optim.AdamW(params,lr=protocol['learning_rate'],weight_decay=0.)
        log=[];torch.cuda.reset_peak_memory_stats();updates=0
        for begin in range(0,len(train),4):
            group=train[begin:begin+4];net.train();optimizer.zero_grad(set_to_none=True)
            for row in group:
                if time.monotonic()-started>84*60:raise TimeoutError('bounded worker time exceeded')
                rgb=decode(row);batch,boundary=supervised(processor,row,rgb);del rgb
                batch={k:v.to('cuda') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                donor=protocol['teacher_donors'][row['id']] if arm=='expert_shuffled' else row['id']
                feature=features[donor].to('cuda')
                loss,task,aux=training_forward(net,batch,boundary,head if arm!='task_only' else None,
                    feature if arm!='task_only' else None,weight=protocol['expert_weight'])
                assert torch.isfinite(loss)
                (loss/len(group)).backward()
                log.append(dict(id=row['id'],task_loss=float(task.detach()),auxiliary_loss=float(aux.detach())))
                del batch,feature,loss,task,aux
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
            assert all(p.grad is None for p in net.parameters() if not p.requires_grad)
            torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();updates+=1
            if updates%40==0:print(dict(phase='training',arm=arm,updates=updates),flush=True)
        assert updates==164
        training[arm]=dict(examples=len(log),optimizer_steps=updates,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            task_loss_mean=sum(r['task_loss'] for r in log)/len(log),
            auxiliary_loss_mean=sum(r['auxiliary_loss'] for r in log)/len(log))
        torch.save(adapter_state(net),out/(arm+'_adapters.pt'))
        save(out/(arm+'_training.json'),log)
        del optimizer,head,params;gc.collect();torch.cuda.empty_cache()
        evaluate(arm)
    assert len(predictions)==2104
    save(out/'completed.json',dict(status='fixed_main_complete',elapsed_s=time.monotonic()-started,
        training=training,total_classifications=len(predictions),generated_tokens=0,
        teacher_loads=1,student_loads=1,adapter_modules=targets,heldout_did_not_select_hyperparameters=True))
    assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<128*2**20
    print(dict(status='fixed_main_complete'),flush=True)


if __name__=='__main__':main()
