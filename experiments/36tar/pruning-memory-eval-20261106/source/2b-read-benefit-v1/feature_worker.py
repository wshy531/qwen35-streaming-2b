"""One approved GPU job extracts 240 vectors. Fitting and labels stay offline."""
from collections import deque,defaultdict
from pathlib import Path
import argparse,gc,hashlib,json,os,sys,time
from prepare import ROOT,read,save,digest
from scheduler import schedule
from feature_runtime import extract
EXEC=ROOT/'runs/read_benefit_execution_prepare_20261028_01'
sys.path.insert(0,str(ROOT/'experiments/2b-recent-evidence-v1'))
from gate_common import MEDIA,shared
from evidence_runtime import recent_pairs,projected_pair

def main(output,lock):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':raise RuntimeError('approved Slurm 1GPU/4CPU required')
    from freeze_features import verify
    verify(lock)
    if not output.resolve().is_relative_to((ROOT/'runs').resolve()):raise ValueError('owned output required')
    env=shared();torch=env['torch'];psutil=env['psutil'];torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info()[0]<24*2**30 or psutil.virtual_memory().available<12*2**30:raise RuntimeError('memory margin')
    requests=read(EXEC/'feature_requests.json');plans=schedule(requests,read(EXEC/'media_resolution.json'),read(EXEC/'sources.json'))
    output.mkdir(exist_ok=False);start=time.monotonic();stats=dict(vision=0,lm_head=0);features={};timings=[];warm=[]
    def guard():
        if time.monotonic()-start>1000 or psutil.Process().memory_info().rss>20*2**30:raise RuntimeError('wall/RSS limit')
        if torch.cuda.max_memory_allocated()>24*2**30:raise RuntimeError('GPU limit')
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file())>16*2**20:raise RuntimeError('output size limit')
    model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('weight loading mismatch')
    model.eval();model.requires_grad_(False)
    if model.get_input_embeddings().embedding_dim!=2048:raise ValueError('real 2048-D base required')
    from sember_runtime import make_builder
    from sember_nodes import pair_node,digest as bank_digest
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    def hook(key):
        def fn(*args):stats[key]+=1
        return fn
    vh=model.model.visual.register_forward_pre_hook(hook('vision'));lh=model.lm_head.register_forward_pre_hook(hook('lm_head'))
    def take(nodes,q,evidence):
        guard();torch.cuda.synchronize();t=time.monotonic();v,meta=extract(env,model,builder,list(nodes),q['question'],evidence);torch.cuda.synchronize()
        return v.numpy(),dict(id=q['id'],evidence=evidence,seconds=time.monotonic()-t,**meta)
    with (output/'timings.jsonl').open('x',encoding='utf-8') as stream:
        for si,plan in enumerate(plans):
            source=plan['source'];media=MEDIA/source['name'];guard()
            if media.resolve().parent!=MEDIA.resolve() or media.stat().st_size!=source['bytes'] or digest(media)!=source['sha256']:raise ValueError('fixed media changed')
            at=defaultdict(list)
            for job in plan['queries']:at[job['last_pair']].append(job)
            nodes=deque(maxlen=4);delivered=set();t=time.monotonic()
            for index,first,second in recent_pairs(media,source):
                guard();value=projected_pair(env,model,builder,first,second)
                nodes.append(pair_node(value,index,first.timestamp,second.timestamp,(9,12)));del value
                for job in at[index]:
                    q=job['request'];before=bank_digest(nodes)
                    if [n.first for n in nodes]!=job['pair_indices']:raise ValueError('recent cache drift')
                    if not warm:
                        for evidence in (True,False):
                            _,row=take(nodes,q,evidence);warm.append(row)
                    data={}
                    for mode,evidence in (('recent_video',True),('no_video',False)):
                        v,row=take(nodes,q,evidence);data[mode]=v;timings.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    if bank_digest(nodes)!=before or model.model.rope_deltas is not None:raise RuntimeError('read mutated state')
                    if q['id'] in features:raise ValueError('duplicate feature extraction')
                    features[q['id']]=data;delivered.add(q['id'])
            if delivered!={j['request']['id'] for j in plan['queries']}:raise ValueError('missing request')
            print(dict(sources=si+1,queries=len(features),vision_calls=stats['vision'],source_seconds=time.monotonic()-t),flush=True)
            del nodes;gc.collect();torch.cuda.empty_cache()
    guard();vh.remove();lh.remove()
    if len(features)!=120 or len(timings)!=240 or len(warm)!=2 or stats!={'vision':453,'lm_head':0}:raise RuntimeError('work exceeds frozen budget')
    import numpy as np
    arrays={mode:np.stack([features[q['id']][mode] for q in requests]) for mode in ('recent_video','no_video')}
    for v in arrays.values():assert v.shape==(120,2048) and v.dtype==np.float32 and np.isfinite(v).all()
    np.savez_compressed(output/'features.npz',ids=np.asarray([q['id'] for q in requests]),**arrays)
    save(output/'warmup.json',warm)
    save(output/'completed.json',dict(status='complete_real_read_features',tasks=120,formal_feature_prefills=240,warmup_feature_prefills=2,vision_calls=453,lm_head_calls=0,
        loads=1,generated_tokens=0,new_QA_answers=0,base_model_training_updates=0,head_fit_done=False,
        maximum_visual_pairs_resident=4,recent_visual_payload_bytes=4*108*2048*4,no_full_feature_archive=True,
        features_sha256=digest(output/'features.npz'),wall_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        RSS_bytes=psutil.Process().memory_info().rss,execution_lock_sha256=digest(lock),partner_writes=0,
        extra_feature_prefill_cost_not_free=True,not_complete_two_stage_speed_test=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);a=p.parse_args();main(a.output,a.lock)
