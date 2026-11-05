"""GPU features only; no base updates, answers, labels-in-prompt or API."""
import os,sys,time,gc,json,math
from pathlib import Path
import numpy as np
from prepare_data import ROOT,PREP,read,save,sha

def shared():
    sys.path.insert(0,str(ROOT/'experiments/2b-evidence-selection-v1'))
    import worker as base
    env,builder,Bank,distance,pair_node,digest,*_=base.setup()
    sys.path.insert(0,str(ROOT/'experiments/2b-read-benefit-v1'))
    from feature_runtime import extract
    return base,env,builder,Bank,distance,pair_node,digest,extract

def short_pairs(source,guard):
    import av,numpy as np,hashlib
    from PIL import Image
    from types import SimpleNamespace
    p=ROOT/'datasets/VideoCon-support-v1/videos'/(source['video_id']+'.mp4')
    if p.stat().st_size!=source['media_bytes'] or sha(p)!=source['media_sha256']:raise ValueError('short media SHA mismatch')
    wanted={f['frame_id']:f for f in source['full_frames']};pending=None;seen=set();origin=None
    with av.open(str(p)) as c:
        stream=c.streams.video[0];stream.codec_context.thread_count=1
        for i,f in enumerate(c.decode(stream)):
            guard()
            if origin is None:origin=float(f.time)
            if i not in wanted:continue
            im=f.to_image();im.thumbnail((384,384),Image.Resampling.BICUBIC);scale=min(384/im.width,288/im.height)
            im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.BICUBIC)
            canvas=Image.new('RGB',(384,288));canvas.paste(im,((384-im.width)//2,(288-im.height)//2))
            t=float(f.time)-origin;r=wanted[i]
            if hashlib.sha256(canvas.tobytes()).hexdigest()!=r['rgb_sha256'] or abs(t-r['pts_s'])>1e-7:raise ValueError('frozen short RGB/PTS mismatch')
            packet=SimpleNamespace(frame_id=i,timestamp=t,rgb=np.asarray(canvas).copy());seen.add(i)
            if pending is None:pending=packet
            else:yield pending,packet;pending=None
            if len(seen)==len(wanted):break
    if pending is not None or seen!=set(wanted):raise ValueError('short paired prefix incomplete')

def main(out):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':raise RuntimeError('approved Slurm allocation required')
    if out.exists() or not out.resolve().is_relative_to((ROOT/'runs').resolve()):raise ValueError('new owned output required')
    from selector_freeze import verify
    verify()
    base,env,make_builder,Bank,distance,pair_node,digest,extract=shared();torch=env['torch'];psutil=env['psutil'];torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info()[0]<24*2**30 or psutil.virtual_memory().available<12*2**30:raise RuntimeError('memory margin')
    out.mkdir();start=time.monotonic();features={};traces=[];counts=dict(vision=0,text_prefills=0)
    def guard():
        if time.monotonic()-start>2550 or psutil.Process().memory_info().rss>20*2**30 or torch.cuda.max_memory_allocated()>24*2**30:raise RuntimeError('resource bound')
    model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise RuntimeError('base loading mismatch')
    model.eval();model.requires_grad_(False);builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    from selector_model import metadata
    with np.load(ROOT/'runs/router-head-219466/features.npz',allow_pickle=False) as old:
        all_questions=old['no_video']
        cached={i:all_questions[j].copy() for j,i in enumerate(old['ids'].tolist())}
    def desc(nodes):
        x=torch.cat([n.features for n in nodes]);return torch.cat((x.mean(0),x.max(0).values)).numpy()
    def record(task,nodes,q=None):
        before=digest(nodes);recent=nodes[-4:];history=nodes[:-4]
        if len(recent)!=4 or len(history)>4:raise ValueError('bounded complete bank required')
        if q is None:
            q,_=extract(env,model,builder,recent,task['question'],False);q=q.numpy();counts['text_prefills']+=1
        if q.shape!=(2048,) or not np.isfinite(q).all():raise ValueError('finite fixed 2B question representation')
        h=np.zeros((4,4096),np.float32);m=np.zeros((4,8),np.float32);mask=np.zeros(4,bool)
        for i,n in enumerate(history):h[i]=desc([n]);m[i]=metadata(n.metadata(),task['query_time_s'],recent[0].begin);mask[i]=True
        features[task['id']]=dict(query=q,recent=desc(recent),history=h,metadata=m,mask=mask)
        traces.append(dict(id=task['id'],history=[n.metadata() for n in history],recent_begin=recent[0].begin))
        if digest(nodes)!=before:raise RuntimeError('feature extraction modified bank')
    short_tasks=read(PREP/'short_requests.json');long_tasks=read(PREP/'long_requests.json')
    with torch.inference_mode():
        for s in read(PREP/'short_sources.json'):
            bank=Bank('LS',108);previous=None
            for i,(a,b) in enumerate(short_pairs(s,guard)):
                v=base.project(env,model,builder,a,b);counts['vision']+=1;d=v.mean(0)
                bank.append(pair_node(v,i,a.timestamp,b.timestamp,(9,12)),starts_event=previous is None or distance(previous,d)>.047536224126815796);previous=d.clone();del v
            for task in [t for t in short_tasks if t['source_id']==s['id']]:record(task,bank.nodes(),cached[task['id']])
            del bank,previous;gc.collect();torch.cuda.empty_cache()
        for s in read(PREP/'long_sources.json'):
            tasks=[t for t in long_tasks if t['source_id']==s['video_id']]
            targets={t['id']:(math.floor(t['query_time_s'])+1)//2 for t in tasks};pairs=max(targets.values())
            spec=dict(check_frames=[],cutoff_s=2*pairs-1,max_pairs=pairs)
            media=ROOT/'datasets/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23'/s['relative_path']
            if media.stat().st_size!=s['bytes'] or sha(media)!=s['sha256']:raise ValueError('long media SHA mismatch')
            bank=Bank('LS',108);previous=None;delivered=set()
            for i,(a,b) in enumerate(base.sampled_pairs(media,spec,guard)):
                v=base.project(env,model,builder,a,b);counts['vision']+=1;d=v.mean(0)
                bank.append(pair_node(v,i,a.timestamp,b.timestamp,(9,12)),starts_event=previous is None or distance(previous,d)>.047536224126815796);previous=d.clone();del v
                for task in [t for t in tasks if targets[t['id']]==i+1]:
                    if b.timestamp>task['query_time_s']:raise ValueError('future visual evidence')
                    record(task,bank.nodes());delivered.add(task['id'])
            if delivered!=set(targets):raise ValueError('missing query checkpoint')
            del bank,previous;gc.collect();torch.cuda.empty_cache()
            print(json.dumps(dict(long_source=s['video_id'],features=len(features),**counts)),flush=True)
    ids=[t['id'] for t in short_tasks+long_tasks]
    if set(features)!=set(ids) or counts!={'vision':7640,'text_prefills':104}:raise RuntimeError('frozen workload mismatch')
    np.savez(out/'features.npz',ids=np.array(ids),**{k:np.stack([features[i][k] for i in ids]) for k in ('query','recent','history','metadata','mask')})
    if (out/'features.npz').stat().st_size>64*2**20:raise RuntimeError('feature artifact limit')
    save(out/'nodes.json',traces);verify()
    save(out/'completed.json',dict(status='complete',features=len(ids),**counts,pretrained_loads=1,generated_tokens=0,base_updates=0,
        wall_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),automatic_retry=False,
        cached_short_questions_from_previous_completed_job=True,history_full_feature_archive=False))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
