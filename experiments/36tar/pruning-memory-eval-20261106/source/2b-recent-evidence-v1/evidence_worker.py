"""GPU task requires a new approval; targeted recent reads, never writes a bank."""
import argparse
from collections import deque
import gc
import json
import os
from pathlib import Path
import time
from gate_common import ROOT,INPUTS,read,save,sha,shared,MEDIA
from evidence_policy import render,aggregate
from evidence_runtime import recent_pairs,projected_pair,score

def main(out,lock_path):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':
        raise RuntimeError('approved one-GPU/four-CPU Slurm allocation required')
    if not out.resolve().is_relative_to((ROOT/'runs').resolve()):raise ValueError('owned output required')
    from freeze_evidence import verify
    verify(lock_path)
    env=shared();torch=env['torch'];psutil=env['psutil'];torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info()[0]<24*2**30 or psutil.virtual_memory().available<12*2**30:
        raise RuntimeError('fixed device/system memory margin failed')
    requests=read(INPUTS/'requests.json');sources=read(INPUTS/'sources.json');protocol=read(INPUTS/'protocol.json')
    budget=protocol['budget'];out.mkdir(exist_ok=False)
    started=time.monotonic();calls=[0];forwards=0;routes=0
    def guard():
        if time.monotonic()-started>budget['worker_seconds'] or psutil.Process().memory_info().rss>20*2**30:
            raise RuntimeError('time or system-memory bound exceeded')
        if torch.cuda.max_memory_allocated()>24*2**30:raise RuntimeError('GPU allocation bound exceeded')
        if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>32*2**20:raise RuntimeError('output bound exceeded')
    model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise RuntimeError('pretrained loading mismatch')
    model.eval()
    from sember_runtime import make_builder
    from sember_nodes import pair_node,digest
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    handle=model.model.visual.register_forward_pre_hook(lambda *a:calls.__setitem__(0,calls[0]+1))
    save(out/'startup.json',dict(precision='float32',loads=1,execution_lock_sha256=sha(lock_path),
        no_cached_answers_loaded=True,no_historical_features_loaded=True))
    warm=False;warm_rows=[]
    with (out/'gate_predictions.jsonl').open('x',encoding='utf-8') as f:
        for si,source in enumerate(sources):
            guard();media=MEDIA/source['name']
            if media.resolve().parent!=MEDIA.resolve() or sha(media)!=source['sha256']:
                raise ValueError('owned fixed media mismatch')
            selected=[q for q in requests if q['source_name']==source['name']]
            at=__import__('collections').defaultdict(list)
            for q in selected:at[q['committed_pairs']-1].append(q)
            delivered=set();nodes=deque(maxlen=4)
            for index,first,second in recent_pairs(media,source):
                guard();value=projected_pair(env,model,builder,first,second)
                nodes.append(pair_node(value,index,first.timestamp,second.timestamp,(9,12)));del value
                if index not in at:continue
                if [n.first for n in nodes]!=list(range(index-3,index+1)):raise ValueError('recent suffix incomplete')
                for q in at[index]:
                    if [[n.begin,n.end] for n in nodes]!=[[q['recent_frames'][2*i]['pts_s'],q['recent_frames'][2*i+1]['pts_s']] for i in range(4)]:
                        raise ValueError('query recent frame timestamps drift')
                    before=digest(nodes)
                    if not warm:
                        for evidence in (True,False):
                            for view in (0,1):warm_rows.append(score(env,model,builder,list(nodes),render(q['question'],q['options'],view),evidence))
                        save(out/'warmup.json',warm_rows);warm=True
                    for shift in range(5):
                        options=q['options'][shift:]+q['options'][:shift]
                        for mode,evidence in (('recent_video',True),('no_video',False)):
                            views=[]
                            for view in (0,1):
                                guard();v=score(env,model,builder,list(nodes),render(q['question'],options,view),evidence)
                                views.append(v);forwards+=1
                            row=dict(task_id=q['id'],shift=shift,mode=mode,views=views,
                                decision=aggregate(views),seconds=sum(v['seconds'] for v in views),
                                branches_released=True,no_history_input=True)
                            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();routes+=1
                    if digest(nodes)!=before:raise RuntimeError('gate mutated recent features')
                    delivered.add(q['id'])
            if delivered!={q['id'] for q in selected}:raise ValueError('missing query')
            del nodes;gc.collect();torch.cuda.empty_cache()
            print(dict(sources=si+1,routes=routes,classifier_forwards=forwards,vision_calls=calls[0]),flush=True)
    guard();handle.remove()
    if routes!=1200 or forwards!=2400 or calls[0]!=budget['visual_forwards'] or len(warm_rows)!=4:
        raise RuntimeError('work differs from frozen budget')
    save(out/'completed.json',dict(status='complete',routes=routes,formal_classifier_forwards=forwards,
        warmup_classifier_forwards=4,visual_forwards=calls[0],loads=1,generated_tokens=0,new_QA_answers=0,
        wall_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        RSS_bytes=psutil.Process().memory_info().rss,no_full_feature_archive=True,
        recent_feature_payload_bytes=4*108*2048*4,not_full_causal_stream_replay=True,
        training_steps=0,default_modified=False,partner_writes=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--lock',type=Path,required=True)
    a=p.parse_args();main(a.out,a.lock)
