"""One AdamW proposal per source; accept using SAME training source hard CE.

No validation-selected updates, backtracking search, or changed surrogate.
Rejecting restores both parameters and optimizer moments/step counter.
"""
import argparse, copy, gc, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-training-order-main-v2'))
import run_main as m
import torch
PREP=ROOT/'runs/training_repair_prepare_20261104_01'

def restore(scorer,optimizer,parameters,state):
    scorer.load_state_dict(parameters)
    optimizer.load_state_dict(state)
    optimizer.zero_grad(set_to_none=True)

def equal_tree(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
    return a==b

def unit_checks():
    torch.manual_seed(71);scorer=torch.nn.Linear(3,1);opt=torch.optim.AdamW(scorer.parameters(),lr=1e-4,weight_decay=0.)
    x=torch.ones(1,3)
    # Both empty and initialized Adam states must survive rejection exactly.
    for step in range(2):
        p=m.snapshot(scorer);state=copy.deepcopy(opt.state_dict())
        opt.zero_grad();scorer(x).square().sum().backward();opt.step()
        restore(scorer,opt,p,state)
        assert equal_tree(p,m.snapshot(scorer)) and equal_tree(state,opt.state_dict())
        opt.zero_grad();scorer(x).square().sum().backward();opt.step()
    return dict(empty_and_initialized_rollback=True)

@torch.no_grad()
def source_loss(model,builder,captured,questions):
    rows=[m.answer(model,builder,q,captured.nodes,m.train_shift(q)) for q in questions]
    return sum(r['ce'] for r in rows)/len(rows),rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cpu-check',action='store_true');a=ap.parse_args()
    assert a.out.resolve().is_relative_to((ROOT/'runs').resolve()) and not a.out.exists()
    for path,h in m.read(PREP/'guard_lock.json').items():assert m.sha(ROOT/path)==h,path
    torch.set_num_threads(1);unit_checks();torch.manual_seed(20261104)
    if a.cpu_check:assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    else:assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4' and torch.cuda.device_count()==1
    env=m.runtime();b=m.make_builder(env,env['AutoProcessor'].from_pretrained(m.WEIGHTS,local_files_only=True));panel=m.read(m.PREP/'panel_LOCAL_ONLY.json');groups=panel['groups']
    if a.cpu_check:
        cfg=env['tiny_config'](b.processor);cfg['vision_config']['depth']=24
        model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float()
    else:
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(m.WEIGHTS,local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.eval().requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
    torch.manual_seed(20261104);scorer=m.GroupScorer(model.config.vision_config.hidden_size).to(next(model.parameters()).device)
    if not a.cpu_check:scorer.load_state_dict(torch.load(ROOT/'runs/training-order-main-220483/initial.pt',map_location='cpu',weights_only=True))
    opt=torch.optim.AdamW(scorer.parameters(),lr=1e-4,weight_decay=0.)
    a.out.mkdir();start=time.monotonic();counts=dict(questions_trained=0,proposals=0,accepted=0,rejected=0,answers=0,check_answers=0,visual_calls=0,regressions=0)
    def guard():
        if time.monotonic()-start>108*60:raise TimeoutError('108 minute fixed stop')
        assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file())<128*2**20
    def log_answer(file,r,**kwargs):file.write(json.dumps(dict(r,**kwargs))+'\n')
    schedule=panel['schedules']['sequential']
    if a.cpu_check:
        schedule=[next(k for k,g in groups.items() if g['role']==r) for r in ('short_train','long_train')]
    with env['fp32_reference'](),(a.out/'updates.jsonl').open('x',encoding='utf-8') as log,(a.out/'answers.jsonl').open('x',encoding='utf-8') as answers:
        if not a.cpu_check:
            ref={}
            for line in (ROOT/'runs/training-order-main-220483/answers.jsonl').open(encoding='utf-8'):
                r=json.loads(line)
                if r['stage']=='initial':ref[(r['id'],r['view'],r['shift'])]=r
            # Twelve fixed development sources, one SHA-selected question each,
            # both shifts; exact old endpoint agreement before any updates.
            for s in m.read(PREP/'plan.json')['sources']:
                if s['role'] not in ('old_development','long_development'):continue
                g=groups[s['video_id']];pairs=m.load_pairs(env,b,g);short=g['role']=='old_development';c=m.capture(model,scorer,pairs,short);counts['visual_calls']+=len(pairs)
                q=next(q for q in g['questions'] if q['id']==s['questions'][0]);views={'full_short' if short else 'LS':c.nodes}
                if not short:views['recent8']=m.all_nodes(c,pairs)[-8:]
                for view,nodes in views.items():
                    for shift in (0,2):
                        r=m.answer(model,b,q,nodes,shift);old=ref[(q['id'],view,shift)]
                        assert r['choice_original']==old['choice_original'] and abs(r['ce']-old['ce'])<=1e-5+1e-5*abs(old['ce'])
                        counts['regressions']+=1;log_answer(answers,r,stage='initial_regression',view=view,role=g['role'])
                del c,pairs;gc.collect()
            assert counts['regressions']==36
            print(json.dumps(dict(stage='regression_passed',**counts)),flush=True)
        for step,vid in enumerate(schedule,1):
            guard();tick=time.monotonic();g=groups[vid];pairs=m.load_pairs(env,b,g);qs=g['questions']
            if a.cpu_check:pairs=pairs[:18];qs=qs[:2]
            short=g['role']=='short_train';opt.zero_grad(set_to_none=True);c=m.capture(model,scorer,pairs,short)
            rows,recomputed,grad_l1=m.batch_gradient(model,b,scorer,c,pairs,qs)
            gradient_ce=sum(r['ce'] for r in rows)/len(rows)
            # Compare proposals under the exact same no-grad readout; do not
            # confound tiny autograd/no-grad kernel differences with progress.
            pre,prerows=source_loss(model,b,c,qs)
            assert abs(pre-gradient_ce)<=1e-5+5e-5*abs(pre)
            norm=torch.nn.utils.clip_grad_norm_(scorer.parameters(),1.,error_if_nonfinite=True)
            assert all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
            before=m.snapshot(scorer);state=copy.deepcopy(opt.state_dict());oldids={i:p.ids.clone() for i,p in c.pairs.items()}
            opt.step();after=m.capture(model,scorer,pairs,short);post,postrows=source_loss(model,b,after,qs)
            tolerance=1e-6+1e-6*abs(pre);accept=post<=pre+tolerance
            changes=sum(54-int(torch.isin(p.ids,oldids[i]).sum()) for i,p in after.pairs.items())
            if not accept:
                restore(scorer,opt,before,state)
                assert equal_tree(before,m.snapshot(scorer)) and equal_tree(state,opt.state_dict())
            counts['proposals']+=1;counts['accepted' if accept else 'rejected']+=1;counts['questions_trained']+=len(qs);counts['check_answers']+=2*len(qs);counts['visual_calls']+=2*len(pairs)+recomputed
            log.write(json.dumps(dict(step=step,video_id=vid,role=g['role'],questions=rows,pre_questions=prerows,post_questions=postrows,gradient_ce=gradient_ce,pre_ce=pre,post_ce=post,tolerance=tolerance,accepted=accept,changed_groups=changes,total_groups=54*len(pairs),grad_norm=float(norm),score_grad_l1=grad_l1,seconds=time.monotonic()-tick))+'\n');log.flush()
            if step in (68,135,150,203,270):torch.save(m.snapshot(scorer),a.out/f'scorer_step{step}.pt')
            del c,after,pairs,state;gc.collect()
            if step%10==0 or a.cpu_check:print(json.dumps(dict(stage='train',**counts,seconds=time.monotonic()-start)),flush=True)
        torch.save(m.snapshot(scorer),a.out/'final.pt')
        if not a.cpu_check:
            assert counts['questions_trained']==2351 and counts['proposals']==270
            for index,vid in enumerate(panel['development']):
                guard();g=groups[vid];pairs=m.load_pairs(env,b,g);short=g['role']=='old_development';c=m.capture(model,scorer,pairs,short);counts['visual_calls']+=len(pairs)
                views={'full_short' if short else 'LS':c.nodes}
                if not short:views['recent8']=m.all_nodes(c,pairs)[-8:]
                for view,nodes in views.items():
                    for q in g['questions']:
                        for shift in (0,2):
                            r=m.answer(model,b,q,nodes,shift);counts['answers']+=1;log_answer(answers,r,stage='final',view=view,role=g['role'])
                answers.flush();del c,pairs;gc.collect()
                if (index+1)%10==0:print(json.dumps(dict(stage='evaluate',sources=index+1,**counts,seconds=time.monotonic()-start)),flush=True)
            assert counts['answers']==1774
    assert all(p.grad is None and p._version==versions[n] for n,p in model.named_parameters())
    m.save(a.out/'completed.json',dict(status='cpu_passed' if a.cpu_check else 'completed',**counts,seconds=time.monotonic()-start,generated_tokens=0,base_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated() if not a.cpu_check else 0))
if __name__=='__main__':main()
