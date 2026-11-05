"""Frozen previous CE scorer, identical EOF readout and questions; no fitting."""
import argparse,json,os,time,gc
from pathlib import Path
from run_main import (ROOT,PREP,WEIGHTS,read,sha,runtime,torch,make_builder,GroupScorer,
    load_pairs,capture,all_nodes,answer,save)
CHECKPOINT=ROOT/'runs/hard-mask-main-220062/mask_ce_step1237.pt'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    assert args.out.resolve().is_relative_to((ROOT/'runs').resolve()) and not args.out.exists()
    for path,h in read(PREP/'execution_lock.json').items():assert sha(ROOT/path)==h,path
    for path,h in read(PREP/'reference_lock.json').items():assert sha(ROOT/path)==h,path
    torch.set_num_threads(1);torch.manual_seed(20261104);assert torch.cuda.device_count()==1
    env=runtime();builder=make_builder(env,env['AutoProcessor'].from_pretrained(WEIGHTS,local_files_only=True))
    model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(WEIGHTS,local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model.eval().requires_grad_(False);scorer=GroupScorer(model.config.vision_config.hidden_size).to('cuda')
    scorer.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True));scorer.requires_grad_(False)
    args.out.mkdir();panel=read(PREP/'panel_LOCAL_ONLY.json');counts=dict(answers=0,visual_calls=0);start=time.monotonic()
    with torch.no_grad(),env['fp32_reference'](),(args.out/'answers.jsonl').open('x',encoding='utf-8') as log:
        for index,vid in enumerate(panel['development']):
            if time.monotonic()-start>28*60:raise TimeoutError('fixed reference deadline')
            g=panel['groups'][vid];pairs=load_pairs(env,builder,g);short=g['role']=='old_development'
            c=capture(model,scorer,pairs,short);counts['visual_calls']+=len(pairs)
            views={'full_short' if short else 'LS':c.nodes}
            if not short:views['recent8']=all_nodes(c,pairs)[-8:]
            for view,nodes in views.items():
                for q in g['questions']:
                    for shift in (0,2):
                        r=answer(model,builder,q,nodes,shift);counts['answers']+=1
                        log.write(json.dumps(dict(r,stage='frozen_reference',view=view,role=g['role'],arm='old_CE'))+'\n')
            log.flush();del pairs,c;gc.collect()
            if (index+1)%10==0:print(json.dumps(dict(sources=index+1,**counts,seconds=time.monotonic()-start)),flush=True)
    assert counts['answers']==1774 and counts['visual_calls']==2275
    save(args.out/'completed.json',dict(status='completed',**counts,questions=616,updates=0,generated_tokens=0,
        checkpoint_sha256=sha(CHECKPOINT),seconds=time.monotonic()-start))
if __name__=='__main__':main()
