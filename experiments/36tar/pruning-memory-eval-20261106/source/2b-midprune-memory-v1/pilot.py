"""One bounded causal memory-combination pilot; labels remain offline."""
import argparse,gc,hashlib,json,os,sys,time
from dataclasses import dataclass
from adapter import *
sys.path.insert(0,str(ROOT/'experiments/2b-learned-prune-main-v1'))
from protocol import runtime,make_builder,encode_pair,CoverageScorer,read,sha
from frozen_pixels import canvas
sys.path.insert(0,str(ROOT/'experiments/2b-hard-mask-prune-v1'))
from hard_prune import GroupScorer,encode as hard_encode
sys.path.insert(0,str(ROOT/'experiments/2b-evidence-eval-v1'))
from response import question
from response_v4 import parse
from resolution_banks import Banks
P=ROOT/'runs/midprune_memory_prepare_20261102_01'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
@dataclass
class Packet:
    frame_id:int
    timestamp:float
    rgb:object
def exact_pairs(env,path,trace):
    assert trace and len(trace)%2==0
    cursor=0;pending=None;origin=None
    with env['av'].open(str(path)) as c:
        s=c.streams.video[0];s.codec_context.thread_count=1
        for index,frame in enumerate(c.decode(s)):
            if origin is None:origin=float(frame.time)
            target=trace[cursor]
            if index<target['frame_id']:continue
            pts=float(frame.time)-origin;im=canvas(frame)
            assert index==target['frame_id'] and abs(pts-target['pts_s'])<1e-7
            assert hashlib.sha256(im.tobytes()).hexdigest()==target['mid_rgb_sha256']
            packet=Packet(index,pts,env['np'].asarray(im).copy())
            if pending is None:pending=packet
            else:yield pending,packet;pending=None
            cursor+=1
            if cursor==len(trace):break
    assert cursor==len(trace) and pending is None
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    for n,h in read(P/'execution_lock.json').items():assert sha(ROOT/n)==h,n
    auth=read(P/'GPU_authorization.json');assert auth['approved'] and auth['execution_lock_sha256']==sha(P/'execution_lock.json')
    for n in ('CPU_local.json','CPU_cluster.json','CPU_layout_local.json','CPU_layout_cluster.json','CPU_media_cluster.json'):assert read(P/n)['passed']
    assert not a.out.exists() and a.out.resolve().is_relative_to((ROOT/'runs').resolve())
    a.out.mkdir();start=time.monotonic();answers=vision=generated=parity=0
    spec=read(P/'protocol.json');qs=read(P/'questions.json');sources=read(P/'sources.json')
    def guard():
        if time.monotonic()-start>43*60:raise TimeoutError('fixed deadline, preserve partial results')
        assert sum(p.stat().st_size for p in a.out.iterdir() if p.is_file())<64*2**20
    try:
        env=runtime();torch.set_num_threads(1)
        assert torch.cuda.device_count()==1 and env['psutil'].virtual_memory().available>=12*2**30
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
            dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False);builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
        head=GroupScorer(model.config.vision_config.hidden_size).to('cuda').eval().requires_grad_(False)
        assert sha(ROOT/spec['checkpoint'])==spec['checkpoint_sha256']
        head.load_state_dict(torch.load(ROOT/spec['checkpoint'],map_location='cpu',weights_only=True))
        with torch.no_grad(),env['fp32_reference'](),(a.out/'answers.jsonl').open('x') as out,(a.out/'event_trace.jsonl').open('x') as events:
            for source in sources:
                guard();path=Path(spec['media_root'])/source['name']
                assert path.resolve().parent==Path(spec['media_root']).resolve() and sha(path)==source['media_sha256']
                memory={v:Memory(v) for v in VIEWS};old=Banks((9,12),arms=tuple(p+'_mid108' for p in POLICIES))
                state=None;selector=CoverageScorer();delivered=set()
                tasks=[q for q in qs if q['source_name']==source['name']]
                for i,(first,second) in enumerate(exact_pairs(env,path,source['sampled_trace'])):
                    guard();batch,_=builder.pair(first,second,0,0);batch=batch.to('cuda')
                    for view,m in memory.items():
                        if view=='mask_ce':features,ids,_,state=hard_encode(model.model.visual,batch.pixel_values_videos,batch.video_grid_thw,head,previous=state,mode='gather')
                        else:features,ids,_=encode_pair(model,batch,view,None,None,selector)
                        vision+=1;trace=m.append(features,ids,i,first.timestamp,second.timestamp)
                        if view=='full':
                            old.append(features,i,first.timestamp,second.timestamp)
                            for policy in POLICIES:
                                assert digest(m.export(policy))==digest(old.export(policy+'_mid108'));parity+=1
                        events.write(json.dumps(dict(source_name=source['name'],view=view,**trace))+'\n')
                        del features,ids
                    events.flush();del batch
                    for q in tasks:
                        if q['query_time_s']!=second.timestamp:continue
                        assert q['id'] not in delivered;delivered.add(q['id'])
                        for view,m in memory.items():
                            for policy in m.banks:
                                guard();tick=time.monotonic();r=ask(env,model,builder,m,policy,question(q,0,'B'),limit=128)
                                r.update(parse(r['text'],len(q['options']),'B'));answers+=1;generated+=len(r['generated_ids'])
                                descriptor_bytes=0 if state is None else state.numel()*state.element_size()
                                extra=descriptor_bytes if view=='mask_ce' else selector.resident_bytes() if view=='heuristic11' else 0
                                r.update(id=q['id'],case_id=q['case_id'],role=q['role'],stage=q['stage'],source_name=q['source_name'],
                                    query_time_s=q['query_time_s'],view=view,policy=policy,visual_pairs_received=i+1,
                                    bank_status=m.status()['banks'][policy],selector_previous_bytes=extra,QA_seconds=time.monotonic()-tick)
                                out.write(json.dumps(r)+'\n');out.flush()
                        print(json.dumps(dict(questions_done=len(delivered),source=source['name'],answers=answers,vision=vision,seconds=time.monotonic()-start)),flush=True)
                assert delivered=={q['id'] for q in tasks}
                for m in memory.values():m.close()
                old.clear();del memory,old,state,selector;gc.collect();torch.cuda.empty_cache()
        assert answers==spec['answers'] and vision==spec['visual_calls'] and generated<=spec['max_generated_tokens'] and parity==4*spec['real_pairs']
        save(a.out/'completed.json',dict(status='complete',answers=answers,visual_calls=vision,generated_tokens=generated,legacy_full_parity=parity,
            model_loads=1,updates=0,seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            comparison_controls_extra_bounded_reference_banks=True,shared_quality_not_standalone_cost=True))
    except Exception as exc:save(a.out/'failed.json',dict(error=repr(exc),answers=answers,vision=vision,generated_tokens=generated));raise
if __name__=='__main__':main()
