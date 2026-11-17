"""Same frozen SB150/short60 readouts; all 840 content endpoints must reproduce."""
import argparse,json,time,os,hashlib,sys,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/7b-content-residual-repair-v1'))
from residual import ContentResidual,encode,torch
sys.path.insert(0,str(ROOT/'experiments/7b-trained-prune-completion-v1'))
from training import pair_inputs,CausalPairs
sys.path.insert(0,str(ROOT/'experiments/7b-prune-selector-transfer-v1'))
from common import Memory,load_model
from selector import forward,prompt,option_ids
sys.path.insert(0,str(Path(__file__).parent))
from verify import verify,PREP,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cpu',action='store_true');a=ap.parse_args()
    verify();assert (a.cpu or os.environ.get('SLURM_JOB_ID')) and not a.out.exists()
    result=json.loads((a.training/'completed.json').read_text());assert result['cpu']==a.cpu
    assert json.loads((a.training/'method.json').read_text())['architecture']=='ContentResidual'
    if not a.cpu:assert result['sources']==270 and result['questions']==2351
    qs=json.loads((ROOT/'runs/7b_prune_head_prepare_20261111_01/questions.json').read_text(encoding='utf8'));assert len(qs)==210
    refs={(r['id'],r['route'],r['shift']):r for r in json.loads((PREP/'content_references.json').read_text())}
    assert len(refs)==840
    if a.cpu:
        from native_cpu import make_tiny
        from transformers import AutoProcessor
        torch.set_num_threads(1);torch.manual_seed(7113);model=make_tiny(real_tokens=True).float()
        assets=ROOT/'runs/7b_transfer_prepare_20261107_01/processor_assets'
        if not assets.exists():assets=ROOT/'models/Qwen2.5-VL-7B-Instruct'
        p=AutoProcessor.from_pretrained(assets,local_files_only=True)
        qs=[next(q for q in qs if q['suite']==s) for s in sorted({q['suite'] for q in qs})]
    else:model,p=load_model()
    device=next(model.parameters()).device;a.out.mkdir();scorers={};weight_hash={}
    for name,file in [('initial','initial.pt'),('guard','final.pt')]:
        path=a.training/file;weight_hash[name]=sha(path)
        scorer=ContentResidual(model.config.vision_config.hidden_size).to(device).eval().requires_grad_(False)
        ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['retained']==70 and ck['cut']==16 and ck['cpu']==a.cpu
        scorer.load_state_dict(ck['scorer']);scorers[name]=scorer
    assert all(torch.count_nonzero(v)==0 for v in (scorers['initial'].correction.net[-1].weight,scorers['initial'].correction.net[-1].bias))
    groups={}
    for q in qs:groups.setdefault((q['suite'],q['source_id']),[]).append(q)
    started=time.monotonic();counts=dict(questions=0,classifications=0,vision=0,content_reference_matches=0)
    try:
        with torch.no_grad(),(a.out/'answers.jsonl').open('x') as log:
            for key,questions in sorted(groups.items(),key=lambda item:min(q['query_time_s'] for q in item[1])):
                media=ROOT/questions[0]['media_path']
                if not a.cpu:assert sha(media)==questions[0]['media_sha256']
                memories={k:Memory('fixed') for k in scorers};past={k:None for k in scorers};index=0
                def commit(x,y,padding=False):
                    nonlocal index
                    batch=pair_inputs(p,[x.rgb,y.rgb],device=device)
                    for k,scorer in scorers.items():
                        features,ids,_,past[k]=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],scorer,past[k],'gather')
                        memories[k].append(features,ids,index,x.timestamp,y.timestamp);counts['vision']+=1
                        assert counts['vision']<=53930,'declared visual call budget exceeded'
                    index+=1
                driver=None if a.cpu else CausalPairs(media,commit)
                try:
                    for q in sorted(questions,key=lambda q:q['query_time_s']):
                        assert time.monotonic()-started<175*60
                        if a.cpu:
                            from types import SimpleNamespace
                            for i in range(6):commit(SimpleNamespace(timestamp=2.*i,rgb=torch.zeros(280,392,3,dtype=torch.uint8).numpy()),SimpleNamespace(timestamp=2.*i+1,rgb=torch.ones(280,392,3,dtype=torch.uint8).numpy()))
                        else:driver.advance(q['query_time_s'])
                        for k,memory in memories.items():
                            for route in ('recent8','LS'):
                                nodes=memory.select(route)
                                for shift in (0,2):
                                    v=forward(model,p,nodes,prompt(q,shift))[option_ids(p,len(q['options']))]
                                    row=dict(id=q['id'],suite=q['suite'],source_id=q['source_id'],method=k,route=route,shift=shift,
                                        option_logits=v.tolist(),selected_original=(int(v.argmax())+shift)%len(q['options']),visual_tokens=sum(len(n.features) for n in nodes))
                                    if k=='initial' and not a.cpu:
                                        ref=refs[(q['id'],route,shift)]
                                        for field in ('selected_original','visual_tokens'):assert row[field]==ref[field],(q['id'],route,shift,field,row[field],ref[field])
                                        counts['content_reference_matches']+=1
                                    log.write(json.dumps(row)+'\n');counts['classifications']+=1
                        counts['questions']+=1;log.flush();print(json.dumps(counts),flush=True)
                finally:
                    if driver is not None:driver.close()
                del memories;gc.collect()
        assert counts['questions']==len(qs) and counts['classifications']==len(qs)*8
        if not a.cpu:assert counts['content_reference_matches']==840
        (a.out/'completed.json').write_text(json.dumps(dict(**counts,cpu_synthetic=a.cpu,weight_hash=weight_hash,seconds=time.monotonic()-started),indent=2))
    except BaseException as e:
        import traceback
        (a.out/'failed.json').write_text(json.dumps(dict(**counts,error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
