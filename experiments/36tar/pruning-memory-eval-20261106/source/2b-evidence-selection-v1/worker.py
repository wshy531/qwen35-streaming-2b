"""Fixed same-bank evidence diagnostic, no training or automatic retrieval."""
import argparse,gc,hashlib,json,math,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from prepare import ROOT,PREP,read,save,sha

def setup():
    for path in ['2b-sember-v1','2b-perception-transition-v1','2b-temporal-router-v1']:
        sys.path.insert(0,str(ROOT/'experiments'/path))
    from sember_common import runtime
    from sember_runtime import make_builder
    from sember_banks import Bank,distance
    from sember_nodes import pair_node,digest
    from canonical_prompt import question_prompt,rotate
    from temporal_policy import lexical_route
    return runtime(),make_builder,Bank,distance,pair_node,digest,question_prompt,rotate,lexical_route

def sampled_pairs(media,spec,guard):
    import av,numpy as np
    from PIL import Image
    expected={f['frame_id']:f for f in spec['check_frames']};seen=set();origin=None;due=0.;pending=None;count=0;previous=-1.
    with av.open(str(media)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        for index,frame in enumerate(container.decode(stream)):
            guard()
            if frame.time is None:raise ValueError('display PTS required')
            if origin is None:origin=float(frame.time)
            pts=float(frame.time)-origin
            if not math.isfinite(pts) or pts<=previous:raise ValueError('monotonic PTS required')
            previous=pts
            if pts>spec['cutoff_s']+1e-9:break
            if pts+1e-9<due:continue
            image=frame.to_image();image.thumbnail((384,384),Image.Resampling.BICUBIC)
            scale=min(384/image.width,288/image.height)
            image=image.resize((max(1,min(384,round(image.width*scale))),max(1,min(288,round(image.height*scale)))),Image.Resampling.BICUBIC)
            canvas=Image.new('RGB',(384,288));canvas.paste(image,((384-image.width)//2,(288-image.height)//2))
            if index in expected:
                f=expected[index]
                if abs(pts-f['pts_s'])>1e-7 or hashlib.sha256(canvas.tobytes()).hexdigest()!=f['mid_rgb_sha256']:
                    raise ValueError('frozen recent RGB/PTS mismatch')
                seen.add(index)
            packet=SimpleNamespace(frame_id=index,timestamp=pts,rgb=np.asarray(canvas).copy())
            if pending is None:pending=packet
            else:
                yield pending,packet
                pending=None;count+=1
            due=math.floor(pts+1e-9)+1
    if count!=spec['max_pairs'] or pending is not None or seen!=set(expected):raise ValueError('sampled prefix/check-frame count drift')

def project(env,model,builder,first,second):
    torch=env['torch'];batch,_=builder.pair(first,second,0,0)
    batch=batch.to(model.get_input_embeddings().weight.device);saved=model.model.rope_deltas
    try:
        with env['fp32_reference']():
            out=model.model.get_video_features(batch.pixel_values_videos,batch.video_grid_thw,return_dict=True)
            value=torch.cat(out.pooler_output,0).detach().to(device='cpu',dtype=torch.float32)
        if value.shape!=(108,2048) or not torch.isfinite(value).all():raise ValueError('fixed real projection required')
        return value
    finally:model.model.rope_deltas=saved

def infer(env,model,builder,nodes,text,selected,full=False):
    from evidence_selection_readout import compose
    torch=env['torch'];saved=model.model.rope_deltas;branch=None
    def sync():
        if model.get_input_embeddings().weight.device.type=='cuda':torch.cuda.synchronize()
    sync();start=time.monotonic()
    try:
        with env['fp32_reference']():
            emb,pos,ids,meta=compose(model,builder,nodes,selected,allow_full=full,question=text)
            branch=env['HybridSequence'](model,max_tokens=8192,session_id='whole-evidence-diagnostic')
            logits=None
            for b in range(0,emb.shape[1],512):
                e=min(b+512,emb.shape[1]);logits=branch.append(inputs_embeds=emb[:,b:e],position_ids=pos[:,:,b:e],logits_mode='last' if e==emb.shape[1] else 'none')
            if logits is None or not torch.isfinite(logits).all():raise FloatingPointError('invalid logits')
            eos=model.generation_config.eos_token_id;stops=set(eos if isinstance(eos,list) else [eos]);stops.add(builder.processor.tokenizer.eos_token_id)
            generated,reason=env['greedy_decode'](branch,logits,8,stops)
            sync();meta.pop('full_context_columns')
            return dict(text=builder.processor.tokenizer.decode(generated,skip_special_tokens=True),generated_ids=generated,
                stop_reason=reason,input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                position_ids_sha256=hashlib.sha256(pos.cpu().numpy().tobytes()).hexdigest(),
                QA_seconds=time.monotonic()-start,branch_released=True,**meta)
    finally:
        if branch is not None:branch.cache.reset()
        model.model.rope_deltas=saved

def main(out,media):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK')!='4':raise RuntimeError('approved Slurm allocation required')
    if out.exists() or not out.resolve().is_relative_to((ROOT/'runs').resolve()):raise ValueError('new owned output required')
    from execution_freeze import verify
    verify()
    env,make_builder,Bank,distance,pair_node,digest,prompt,rotate,lexical=setup()
    torch=env['torch'];psutil=env['psutil'];torch.set_num_threads(1)
    if torch.cuda.device_count()!=1 or torch.cuda.mem_get_info()[0]<24*2**30 or psutil.virtual_memory().available<12*2**30:
        raise RuntimeError('declared GPU/system free-memory margin required')
    plan=read(PREP/'runtime_plan.json');tasks={t['id']:t for t in read(PREP/'inputs/requests.json')}
    allowed=[ROOT/'datasets/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos',
        Path('/home/u2025010462/share/czy/wsy/data/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos')]
    if media.resolve() not in [p.resolve() for p in allowed]:raise ValueError('authorized exact media path required')
    for s in plan['sources']:
        p=media/s['name']
        if p.resolve().parent!=media.resolve() or p.stat().st_size!=s['bytes'] or sha(p)!=s['sha256']:raise ValueError('media SHA mismatch')
    out.mkdir();start=time.monotonic();stats=dict(vision=0,model_answers=0,generated_tokens=0,reference_matches=0,warmup_answers=0)
    def guard():
        if time.monotonic()-start>2550 or psutil.Process().memory_info().rss>20*2**30:raise RuntimeError('wall/RSS budget exceeded')
        if torch.cuda.max_memory_allocated()>24*2**30:raise RuntimeError('GPU memory bound exceeded')
        if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>64*2**20:raise RuntimeError('output budget exceeded')
    model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise RuntimeError('weight load mismatch')
    model.eval();model.requires_grad_(False)
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    from transition_scoring import parse_original
    records=[];checkpoints=[]
    try:
        with torch.inference_mode(),(out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for source in plan['sources']:
                scheduled=[r for r in plan['tasks'] if tasks[r['task_id']]['source_name']==source['name']]
                bank=Bank('LS',108);previous=None;delivered=set()
                for i,(a,b) in enumerate(sampled_pairs(media/source['name'],source,guard)):
                    guard();value=project(env,model,builder,a,b);stats['vision']+=1;desc=value.mean(0)
                    bank.append(pair_node(value,i,a.timestamp,b.timestamp,(9,12)),starts_event=previous is None or distance(previous,desc)>0.047536224126815796)
                    previous=desc.clone();del value
                    for item in [r for r in scheduled if r['committed_pairs']==i+1]:
                        t=tasks[item['task_id']];nodes=bank.nodes();before=digest(nodes)
                        for real,expected in zip(nodes,item['nodes']):
                            actual=real.metadata()
                            if any(actual[k]!=expected[k] for k in ('first','last','begin','end','center','tokens','exact_pair')):
                                raise ValueError('recorded real node partition differs')
                        if len(nodes)!=8 or b.timestamp!=item['cutoff_s']:raise ValueError('bank/query checkpoint differs')
                        for shift in range(5):
                            text=prompt(rotate(t,shift)[0]);base={}
                            for arm,sel in [('recent',[]),('full',[0,1,2,3])]:
                                guard()
                                if stats['warmup_answers']<2:
                                    warm=infer(env,model,builder,nodes,text,sel,full=arm=='full')
                                    stats['warmup_answers']+=1;stats['generated_tokens']+=len(warm['generated_ids'])
                                result=infer(env,model,builder,nodes,text,sel,full=arm=='full')
                                stats['model_answers']+=1;stats['generated_tokens']+=len(result['generated_ids'])
                                ref=next(r for r in item['references'] if r['shift']==shift and r['arm']==arm)
                                if result['input_ids_sha256']!=ref['input_ids_sha256'] or parse_original(result['text'],shift)!=ref['expected_choice']:
                                    raise RuntimeError('old input/choice reference failed; stop before analysis')
                                for k in ('generated_ids','position_ids_sha256'):
                                    if k in ref and result[k]!=ref[k]:raise RuntimeError('old transition token/position reference failed')
                                stats['reference_matches']+=1;base[arm]=result
                            for arm in ('recent','full','lexical','time_matched_nodes','nonmatching_nodes'):
                                if arm in base:result=dict(base[arm],reused_from=None)
                                elif arm=='lexical':
                                    route=lexical(t['question'])['route'];result=dict(base[route],reused_from=route)
                                else:
                                    sel=item['selection']['selected_history_indices'] if arm=='time_matched_nodes' else item['selection']['control_history_indices']
                                    if sel is None:result=dict(available=False,reason='no_equal_token_nonmatching_time_control')
                                    elif sel==[]:result=dict(base['recent'],reused_from='recent')
                                    else:
                                        guard();result=infer(env,model,builder,nodes,text,sel)
                                        stats['model_answers']+=1;stats['generated_tokens']+=len(result['generated_ids'])
                                row=dict(task_id=t['id'],arm=arm,shift=shift,available=result.get('available',True),**{k:v for k,v in result.items() if k!='available'})
                                records.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                            if digest(bank.nodes())!=before:raise RuntimeError('QA mutated bank')
                        delivered.add(t['id']);checkpoints.append(dict(task_id=t['id'],nodes=[n.metadata() for n in nodes],selection=item['selection']))
                if delivered!={r['task_id'] for r in scheduled}:raise RuntimeError('missing query')
                del bank,previous;gc.collect();torch.cuda.empty_cache()
                print(json.dumps(dict(source=source['name'],rows=len(records),**stats)),flush=True)
        if len(records)!=1600 or stats['reference_matches']!=640 or stats['vision']!=2796:raise RuntimeError('fixed workload mismatch')
        exact=read(PREP/'exact_workload.json')
        if stats['model_answers']!=exact['formal_model_answers'] or stats['warmup_answers']!=2 or stats['generated_tokens']>exact['maximum_generated_tokens']:
            raise RuntimeError('answer/token bound')
        verify();save(out/'completed.json',dict(status='complete',**stats,wall_seconds=time.monotonic()-start,
            records=len(records),peak_allocated_bytes=torch.cuda.max_memory_allocated(),RSS_snapshot_bytes=psutil.Process().memory_info().rss,
            checkpoints=checkpoints,pretrained_loads=1,trained=False,automatic_retry=False))
    except Exception as e:
        save(out/'failed.json',dict(error_type=type(e).__name__,error=str(e),**stats,rows=len(records)))
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--media-root',type=Path,required=True)
    a=p.parse_args();main(a.out,a.media_root)
