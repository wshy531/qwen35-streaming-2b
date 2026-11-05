"""12 fixed NExT-QA clips, full/fixed, answer/evidence. No paid API."""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
sys.path.insert(0,str(R/'experiments/2b-learned-prune-main-v1'))
from protocol import runtime,make_builder,decode,read,sha,DATA,context,encode_pair,CoverageScorer
from response import question,parse
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def records_for(model,pairs,arm):
    result=[];scorer=CoverageScorer()
    for i,(begin,end,batch) in enumerate(pairs):
        batch=batch.to(model.get_input_embeddings().weight.device)
        feat,ids,_=encode_pair(model,batch,arm,None,None,scorer)
        result.append((i,begin,end,feat,ids))
    return result
def generate(env,model,builder,records,q,protocol):
    emb,pos,ids=context(model,builder,records,question(q,0,protocol))
    branch=env['HybridSequence'](model,max_tokens=4096,session_id='evidence-profile')
    saved=model.model.rope_deltas;limit=8 if protocol=='A' else 128
    try:
        for start in range(0,emb.shape[1],512):
            end=min(start+512,emb.shape[1]);logits=branch.append(inputs_embeds=emb[:,start:end],position_ids=pos[:,:,start:end],logits_mode='last' if end==emb.shape[1] else 'none')
        eos=model.generation_config.eos_token_id
        stops=set(eos if isinstance(eos,list) else [] if eos is None else [eos]);stops.add(builder.processor.tokenizer.eos_token_id)
        generated,reason=env['greedy_decode'](branch,logits,limit,stops)
    finally:branch.cache.reset();model.model.rope_deltas=saved
    text=builder.processor.tokenizer.decode(generated,skip_special_tokens=True)
    return dict(text=text,generated_ids=generated,reason=reason,input_tokens=ids.shape[1],**parse(text,len(q['options']),protocol))
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    for n,h in read(P/'execution_lock.json').items():assert sha(R/n)==h,n
    assert not a.out.exists();a.out.mkdir();started=time.monotonic();answers=calls=generated=0
    try:
        env=runtime();t=env['torch'];t.set_num_threads(1);assert t.cuda.device_count()==1
        model,info=env['Qwen3_5ForConditionalGeneration'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True,dtype=t.float32,
            device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        assert not any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False)
        builder=make_builder(env,env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True))
        audit={r['video_id']:r for r in read(DATA/'CPU_media_audit.json')['rows']}
        frame_root=a.out/'sampled_frames';frame_root.mkdir()
        from PIL import Image,ImageDraw
        materials=[]
        with t.no_grad(),env['fp32_reference'](),(a.out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for qi,q in enumerate(read(P/'profile.json')):
                if time.monotonic()-started>27*60:raise TimeoutError('profile deadline')
                path=R/q['media_path'];assert sha(path)==q['media_sha256'];au=audit[q['video_id']]
                pairs=decode(env,path,au,builder)
                frames=[];desired={s['frame_index']:s for s in au['samples']}
                from audit_media import canvas
                with env['av'].open(str(path)) as container:
                    video=container.streams.video[0];video.thread_count=1
                    for index,f in enumerate(container.decode(video)):
                        if index not in desired:continue
                        image=canvas(f);s=desired[index];digest=hashlib.sha256(image.tobytes()).hexdigest();assert digest==s['rgb_sha256']
                        dest=frame_root/f'{q["source_id"]}-{index}.png';image.save(dest)
                        frames.append(dict(frame_id=index,pts_s=s['pts_s'],path=dest.relative_to(a.out).as_posix(),rgb_sha256=digest))
                materials.append(dict(material_id=q['id'],frames=frames,complete=True,query_time_s=q['query_time_s']))
                # All actual sampled inputs, small contact sheet only for manual inspection.
                sheet=Image.new('RGB',(384*4,312*((len(frames)+3)//4)),(35,35,35));draw=ImageDraw.Draw(sheet)
                for j,f in enumerate(frames):
                    with Image.open(a.out/f['path']) as im:sheet.paste(im,((j%4)*384,(j//4)*312))
                    draw.text(((j%4)*384+4,(j//4)*312+289),f'{f["frame_id"]} / {f["pts_s"]:.3f}s',fill='white')
                sheet.save(a.out/f'contact-{q["source_id"]}.jpg',quality=85)
                arms=('full','fixed12') if qi%2==0 else ('fixed12','full')
                for arm in arms:
                    records=records_for(model,pairs,arm);calls+=len(pairs)
                    for protocol in ('A','B'):
                        tick=time.monotonic();result=generate(env,model,builder,records,q,protocol);t.cuda.synchronize()
                        answers+=1;generated+=len(result['generated_ids'])
                        result.update(id=q['id'],source_id=q['source_id'],arm=arm,protocol=protocol,seconds=time.monotonic()-tick)
                        stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                    del records;gc.collect()
                del pairs;t.cuda.empty_cache();save(a.out/'materials.json',materials)
                assert sum(x.stat().st_size for x in a.out.rglob('*') if x.is_file())<128*2**20
                print(json.dumps(dict(sources=qi+1,answers=answers,seconds=time.monotonic()-started)),flush=True)
        assert answers==48 and generated<=3264
        save(a.out/'completed.json',dict(status='complete_interface_profile',answers=answers,vision_calls=calls,generated_tokens=generated,
            seconds=time.monotonic()-started,API=0,model_loads=1,training_updates=0,peak_allocated_bytes=t.cuda.max_memory_allocated()))
    except Exception as e:save(a.out/'failed.json',dict(error=repr(e),answers=answers,vision_calls=calls));raise
if __name__=='__main__':main()
