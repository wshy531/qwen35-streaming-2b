"""Frozen checkpoint trajectory; fixed train/dev probes, no fitting."""
import sys, os, json, argparse, time, gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-training-order-main-v2'))
import run_main as m
import torch
PREP=ROOT/'runs/training_repair_prepare_20261104_01'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cpu-check',action='store_true');a=ap.parse_args()
    assert a.out.resolve().is_relative_to((ROOT/'runs').resolve()) and not a.out.exists()
    for path,h in m.read(PREP/'lock.json').items():assert m.sha(ROOT/path)==h,path
    torch.set_num_threads(1);torch.manual_seed(20261104)
    if not a.cpu_check:assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    else:assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    env=m.runtime();b=m.make_builder(env,env['AutoProcessor'].from_pretrained(m.WEIGHTS,local_files_only=True))
    if a.cpu_check:
        cfg=env['tiny_config'](b.processor);cfg['vision_config']['depth']=24
        model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float()
    else:
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(m.WEIGHTS,local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.eval().requires_grad_(False);scorer=m.GroupScorer(model.config.vision_config.hidden_size).to(next(model.parameters()).device)
    plan=m.read(PREP/'plan.json');groups=m.read(m.PREP/'panel_LOCAL_ONLY.json')['groups'];a.out.mkdir();start=time.monotonic();count=0;vision=0
    checkpoints=plan['checkpoints'] if not a.cpu_check else {'tiny_initial':None,'tiny_repeat':None}
    selected=plan['sources'] if not a.cpu_check else [next(s for s in plan['sources'] if s['role']=='long_train')]
    with torch.no_grad(),env['fp32_reference'](),(a.out/'answers.jsonl').open('x',encoding='utf-8') as answers,(a.out/'selection.jsonl').open('x',encoding='utf-8') as selections:
        for index,s in enumerate(selected):
            g=groups[s['video_id']];pairs=m.load_pairs(env,b,g)
            if a.cpu_check:pairs=pairs[:18]
            short=s['role'] in ('short_train','old_development');qs=[q for q in g['questions'] if q['id'] in s['questions']]
            if a.cpu_check:qs=qs[:1]
            initial_ids=None
            for name,path in checkpoints.items():
                if time.monotonic()-start>plan['max_seconds']:raise TimeoutError('fixed deadline')
                if path:scorer.load_state_dict(torch.load(ROOT/path,map_location='cpu',weights_only=True))
                stats=[]
                def hook(module,args,out):
                    scores=out[0].detach().float().cpu();v=scores.sort(descending=True).values
                    stats.append(dict(mean=float(scores.mean()),std=float(scores.std()),margin=float(v[53]-v[54]),absmax=float(scores.abs().max())))
                h=scorer.register_forward_hook(hook)
                try:c=m.capture(model,scorer,pairs,short)
                finally:h.remove()
                vision+=len(pairs);ids=[c.pairs[i].ids for i in sorted(c.pairs)]
                if initial_ids is None:initial_ids=[x.clone() for x in ids]
                overlap=sum(int(torch.isin(x,y).sum()) for x,y in zip(ids,initial_ids))/(54*len(ids))
                selections.write(json.dumps(dict(checkpoint=name,video_id=s['video_id'],role=s['role'],pairs=len(pairs),overlap_initial=overlap,stats=stats,ids=[x.tolist() for x in ids]))+'\n');selections.flush()
                views={'full_short' if short else 'LS':c.nodes}
                if not short:views['recent8']=m.all_nodes(c,pairs)[-8:]
                for view,nodes in views.items():
                    for q in qs:
                        shift=m.train_shift(q) if s['role'].endswith('_train') else 0
                        row=m.answer(model,b,q,nodes,shift);count+=1
                        answers.write(json.dumps(dict(row,checkpoint=name,role=s['role'],view=view))+'\n')
                answers.flush();del c;gc.collect()
            del pairs;gc.collect();print(json.dumps(dict(sources=index+1,answers=count,visual_calls=vision,seconds=time.monotonic()-start)),flush=True)
    m.save(a.out/'completed.json',dict(status='cpu_passed' if a.cpu_check else 'completed',answers=count,visual_calls=vision,sources=len(selected),checkpoints=len(checkpoints),seconds=time.monotonic()-start,updates=0,generated_tokens=0))
if __name__=='__main__':main()
