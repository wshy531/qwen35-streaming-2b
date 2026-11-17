import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/7b-prune-selector-transfer-v1'))
from recovery_common import check_control, specification, verify_recovery, REC
from state_contract import compare_state,NEW_CANVAS
"""Resume only 150 missing StreamingBench questions; no whole-prefix GPU audit."""
import argparse,os,time,gc,traceback
from common import *
from selector import forward,prompt,option_ids

def check_pair(spec,i,first,second):
    if isinstance(spec.get('pairs'),list):
        r=spec['pairs'][i]
        for p,k in [(first,'first'),(second,'second')]:
            assert p.frame_id==r[k]['frame_id'] and abs(p.timestamp-r[k]['pts_s'])<1e-6
            if 'rgb_sha256' in r[k]:assert hashlib.sha256(p.rgb.tobytes()).hexdigest()==r[k]['rgb_sha256']
    elif spec.get('sampled_trace'):
        for p,r in zip((first,second),spec['sampled_trace'][2*i:2*i+2]):
            assert p.frame_id==r['frame_id'] and abs(p.timestamp-r['pts_s'])<1e-6

def main(out):
    assert os.environ.get('SLURM_JOB_ID') and not out.exists();verify();verify_recovery();out.mkdir()
    recovery_spec=specification('prune');control_checks=0;state_checks=0
    start=time.monotonic();counts=dict(answers=0,vision=0,audits=0,regressions=0,generated_tokens=0,pairs=0)
    plan=read(PREP/'plan.json');budget=read(REC/'recovery.json')
    def guard():
        if time.monotonic()-start>237*60:raise TimeoutError('fixed 237 minute internal stop')
        assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<128*2**20
        assert counts['vision']<=budget['visual_calls_bound']
    try:
        model,p=load_model();versions={n:x._version for n,x in model.named_parameters()}
        qs=recovery_spec['questions'];traces=read(PREP/'traces.json');groups={}
        for q in qs:groups.setdefault((q['suite'],q['source_id']),[]).append(q)
        with torch.no_grad(),(out/'answers.jsonl').open('x',encoding='utf-8') as log:
            for key,tasks in sorted(groups.items(),key=lambda item:(not any(q['id']==recovery_spec['control_id'] for q in item[1]),item[0])):
                guard();tasks.sort(key=lambda q:(q['query_time_s'],q['id']));media=path(tasks[0]);assert sha(media)==tasks[0]['media_sha256']
                spec=traces['|'.join(key)]
                memories={m:Memory(m,archive=False) for m in plan['prune_modes']}
                index=0
                def commit(first,second,padding=False):
                    nonlocal index
                    guard();check_pair(spec,index,first,second)
                    assert first.rgb.shape==second.rgb.shape==NEW_CANVAS
                    batch=readout.pair_inputs(p,[first.rgb,second.rgb],device='cuda')
                    official,full_ids,full_meta=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],'full',capture_prefix=True);counts['vision']+=1
                    if counts['pairs']==0:
                        keep=model.get_video_features(**batch,return_dict=True).pooler_output[0]
                        torch.testing.assert_close(keep,official,atol=0,rtol=0)
                        save(out/'all_keep_gate.json',dict(passed=True,max_abs=0));counts['vision']+=1
                    for mode,m in memories.items():
                        if mode=='full':value,ids=official,torch.arange(140,device='cuda')
                        else:
                            value,ids,_=encode(model.model.visual,batch['pixel_values_videos'],batch['video_grid_thw'],mode,m.scorer,prefix=full_meta['prefix']);counts['vision']+=1
                        m.append(value,ids,index,first.timestamp,second.timestamp)
                    index+=1;counts['pairs']+=1
                driver=CausalPairs(media,commit)
                try:
                    for q in tasks:
                        state=driver.advance(q['query_time_s']);guard()
                        assert q['id']==recovery_spec['control_id'] or control_checks==12
                        if spec.get('states',{}).get(q['id']):
                            result=compare_state(state,spec['states'][q['id']]);state_checks+=1
                            with (out/'state_checks.jsonl').open('a',encoding='utf8') as sf:
                                sf.write(json.dumps(dict(id=q['id'],**result))+'\n')
                        for mode,m in memories.items():
                            for route in plan['routes']:
                                guard();torch.cuda.synchronize();tick=time.monotonic()
                                nodes=m.select(route);before=digest(nodes)
                                for shift in (0,2):
                                    values=forward(model,p,nodes,prompt(q,shift))[option_ids(p,len(q['options']))]
                                    result=dict(id=q['id'],suite=q['suite'],source_id=q['source_id'],mode=mode,route=route,shift=shift,
                                        option_logits=values.tolist(),selected_original=(int(values.argmax())+shift)%len(q['options']),
                                        visual_tokens=sum(len(n.features) for n in nodes),nodes=[n.metadata() for n in nodes])
                                    if q['id']==recovery_spec['control_id']:check_control(result,recovery_spec['references'],'prune');control_checks+=1
                                    log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush();counts['answers']+=1
                                assert digest(nodes)==before
                        print(json.dumps(dict(id=q['id'],elapsed=time.monotonic()-start,**counts)),flush=True)
                finally:driver.close()
                del memories;gc.collect();torch.cuda.empty_cache()
        assert counts['answers']==1812 and state_checks==150 and counts['audits']==0 and counts['regressions']==0 and control_checks==12
        assert counts['vision']<=budget['visual_calls_bound'] and counts['generated_tokens']<=plan['generated_tokens_max']
        assert all(x._version==versions[n] and x.grad is None for n,x in model.named_parameters())
        verify();save(out/'completed.json',dict(**counts,seconds=time.monotonic()-start,base_updates=0,state_checks=state_checks,control_checks=control_checks,quality_protocol='two option permutations; normalized single-letter scoring',shared_first16_visual_layers=True,standalone_cost=False))
    except BaseException as e:
        save(out/'failed.json',dict(error=repr(e),traceback=traceback.format_exc(),**counts));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);main(ap.parse_args().out)
