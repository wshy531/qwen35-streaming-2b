"""Fixed development pilot: full uncompressed / recent / text-only, not a frozen benchmark.

No labels or evidence annotations are read by the inference worker. Reuses the
pinned pair encoder and fresh external-memory reader. Full means the shared
pairwise protocol, not the model's native whole-video processor protocol.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def prepare(out):
    out.mkdir(exist_ok=False)
    plan=read(ROOT/'runs/dataset_build_20261018_01/handoff/development_plan_v2_not_frozen.json')
    ids=[r['id'] for r in plan['rows']]
    raw=[json.loads(s) for s in (ROOT/'runs/08b_fallback_preparation_20261018_01/draft_inputs/model_requests.jsonl').read_text(encoding='utf-8').splitlines()]
    byid={r['id']:r for r in raw}
    rows=[byid[k] for k in ids]
    if len(rows)!=24 or len(set(ids))!=24:raise ValueError('fixed development24 required')
    requests=[]
    for r in rows:
        # Deliberate allowlist; no official answer, task, evidence, or reviewer text.
        requests.append(dict(id=r['id'],media_path=r['media_path'],media_sha256=r['media_sha256'],model_visible=r['model_visible']))
    save(out/'requests.json',requests)
    profile=read(ROOT/'runs/08b_fallback_preparation_20261018_01/profile.json')
    save(out/'protocol.json',dict(status='preregistered_development_diagnostic_not_frozen',
        model_id=profile['model_id'],revision=profile['model_revision'],questions=len(rows),
        source_files=len(set(r['media_path'] for r in rows)),request_sha256=sha(out/'requests.json'),
        arms=['full_pairwise','recent8_pairs','text_only'],sampling_fps=.5,canvas=[192,256],
        max_new_tokens=8,context_tokens=8192,prefill_chunk=512,
        maximum_pretrained_loads=1,gpu_wall_seconds=1200,minimum_free_ram_gib=3,
        minimum_free_gpu_gib=5.5,maximum_tree_rss_gib=5,maximum_gpu_allocated_gib=7.25,
        source_profile_sha256=sha(ROOT/'runs/08b_fallback_preparation_20261018_01/profile.json'),
        selection='existing24 development IDs, before any model outputs; no holdout',
        full_reference_scope='all causal .5FPS pairs, independent pair encoding and pinned readout; not native whole-video baseline',
        stop='one bounded pilot; retain missing/format/errors; no prompt or threshold sweep'))
    print(json.dumps(dict(prepared=True,questions=len(rows),sources=len(set(r['media_path'] for r in rows)))))


def imports():
    import torch
    import numpy as np
    import av
    import psutil
    from transformers import AutoProcessor
    profile=read(ROOT/'runs/08b_fallback_preparation_20261018_01/profile.json')
    active=Path(profile['repository'])/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    sys.path.insert(0,str(active));sys.path.append(str(ROOT/'experiments/2b-memory-eval'))
    from legacy_sources import load
    from qwen35_local import Qwen3_5ForConditionalGeneration,Qwen3_5Config
    from streaming.online import PairInputBuilder,fp32_reference,greedy_decode
    from streaming.hybrid import HybridSequence
    from streaming.window import FramePacket
    from audit_sampling import canvas_bytes
    from score_memory_eval import prompt
    from readout import context,Limits
    return locals()


def prefill(env,model,emb,pos,chunk):
    torch=env['torch'];session=env['HybridSequence'](model,max_tokens=8192,session_id='capability-pilot')
    try:
        for i in range(0,emb.shape[1],chunk):
            end=min(i+chunk,emb.shape[1])
            logits=session.append(inputs_embeds=emb[:,i:end],position_ids=pos[:,:,i:end],
                logits_mode='last' if end==emb.shape[1] else 'none')
        return session,logits
    except BaseException:
        session.cache.reset();raise


def text_context(env,model,builder,question):
    torch=env['torch']
    rendered=builder.processor.apply_chat_template([dict(role='user',content=question)],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=builder.tokenize(rendered).to(model.get_input_embeddings().weight.device)
    n=ids.shape[1]
    pos=torch.arange(n,device=ids.device)[None,None,:].expand(4,1,-1)
    return model.get_input_embeddings()(ids),pos,ids,dict(input_tokens=n,visual_tokens=0,nodes=[])


def answer(env,model,builder,nodes,question):
    torch=env['torch'];saved=model.model.rope_deltas
    if nodes:
        emb,pos,ids,meta=env['context'](model,builder,nodes,question,
            limits=env['Limits'](8192,6144,128),max_new_tokens=8)
    else:emb,pos,ids,meta=text_context(env,model,builder,question)
    if emb.shape[1]+8>8192:raise ValueError('context budget; no truncation')
    session=None
    try:
        session,logits=prefill(env,model,emb,pos,512)
        if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite logits')
        configured=model.generation_config.eos_token_id
        stops=set(configured if isinstance(configured,list) else [configured])
        stops.add(builder.processor.tokenizer.eos_token_id)
        generated,reason=env['greedy_decode'](session,logits,8,stops)
        return dict(text=builder.processor.tokenizer.decode(generated,skip_special_tokens=True),
            generated_ids=generated,stop_reason=reason,input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            input_tokens=meta['input_tokens'],visual_tokens=meta['visual_tokens'],
            pair_indices=[n.first for n in nodes],context_tokens=session.seen_tokens)
    finally:
        if session:session.cache.reset()
        model.model.rope_deltas=saved


def check_chunk(env,model,builder,nodes,question):
    torch=env['torch'];saved=model.model.rope_deltas
    emb,pos,_,_=env['context'](model,builder,nodes[:2],question,
        limits=env['Limits'](8192,6144,128),max_new_tokens=8)
    all_logits=[]
    for chunk in [8192,64]:
        session,logits=prefill(env,model,emb,pos,chunk)
        all_logits.append(logits.detach().cpu());session.cache.reset();model.model.rope_deltas=saved
    error=float((all_logits[0]-all_logits[1]).abs().max())
    same=int(all_logits[0].argmax())==int(all_logits[1].argmax())
    return dict(max_absolute_logit_error=error,first_argmax_same=same,passed=error<1e-3 and same,
        scope='first two actual pairs, one complete development question; not all-input equivalence')


def worker(out):
    import psutil
    initial=psutil.virtual_memory().available
    if initial<3*2**30:
        save(out/'resource_guard.json',dict(status='blocked_before_import',available_ram_bytes=initial));return
    env=imports();torch=env['torch'];legacy=env['load']()
    torch.set_num_threads(1);torch.manual_seed(123)
    profile=env['profile'];modeldir=ROOT/'models/Qwen3.5-0.8B'
    for f in profile['model_assets']:
        p=modeldir/f['name']
        if p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:raise ValueError('model asset mismatch')
    free,total=torch.cuda.mem_get_info()
    guard=dict(available_ram_bytes=psutil.virtual_memory().available,free_gpu_bytes=free,
        minimum_ram_bytes=3*2**30,minimum_gpu_bytes=5.5*2**30)
    save(out/'resource_guard.json',guard)
    if guard['available_ram_bytes']<3*2**30 or free<5.5*2**30:
        print('RESOURCE_GUARD_BLOCKED',flush=True);return
    source=Path(sys.modules[env['Qwen3_5ForConditionalGeneration'].__module__].__file__)
    save(out/'load_started.json',dict(real_pretrained_load_attempts=1,source=str(source),source_sha256=sha(source),guard=guard))
    started=time.monotonic()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(modeldir,local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    if next(model.parameters()).dtype!=torch.float32:raise ValueError('FP32 required')
    save(out/'loaded.json',dict(real_pretrained_loads=1,load_seconds=time.monotonic()-started,
        gpu_name=torch.cuda.get_device_name(),revision=profile['model_revision']))
    processor=env['AutoProcessor'].from_pretrained(modeldir,local_files_only=True)
    builder=env['PairInputBuilder'](processor,canvas=(192,256),resize_mode='letterbox')
    requests=read(out/'requests.json');protocol=read(out/'protocol.json')
    if sha(out/'requests.json')!=protocol['request_sha256']:raise ValueError('requests changed')
    groups={}
    for r in requests:groups.setdefault(r['media_path'],[]).append(r)
    written=0;pair_count=0;integrity=[];chunk_checked=False
    predictions=out/'predictions.jsonl'
    with predictions.open('x',encoding='utf-8') as f,torch.inference_mode(),env['fp32_reference']():
        for media,rows in groups.items():
            if time.monotonic()-started>1100:break
            if sha(ROOT/media)!=rows[0]['media_sha256']:raise ValueError('media changed')
            wanted={q['frame_id']:q for r in rows for q in r['model_visible']['frames']}
            frames={};origin=None
            with env['av'].open(str(ROOT/media)) as container:
                stream=container.streams.video[0];stream.codec_context.thread_count=1
                for index,frame in enumerate(container.decode(stream)):
                    if origin is None:origin=float(frame.time)
                    if index not in wanted:continue
                    q=wanted[index];pts=float(frame.time)-origin;rgb=env['canvas_bytes'](frame)
                    if abs(pts-q['pts_s'])>1e-7 or hashlib.sha256(rgb).hexdigest()!=q['rgb_sha256']:
                        raise ValueError('actual RGB/PTS differs from registered shared input')
                    frames[index]=env['FramePacket'](index,pts,env['np'].frombuffer(rgb,dtype='uint8').reshape(192,256,3).copy(),index+1)
                    if len(frames)==len(wanted):break
            if len(frames)!=len(wanted):raise ValueError('missing registered frames')
            encoded={}
            for r in rows:
                specs=r['model_visible']['frames'];nodes=[]
                if any(q['pts_s']>r['model_visible']['query_time_s']+1e-9 for q in specs):raise ValueError('future frame')
                for i in range(0,len(specs),2):
                    a,b=specs[i],specs[min(i+1,len(specs)-1)];key=(a['frame_id'],b['frame_id'])
                    if key not in encoded:
                        inputs,_=builder.pair(frames[key[0]],frames[key[1]],0,0,synthetic_padding=key[0]==key[1])
                        visual=torch.cat(model.model.get_video_features(inputs.pixel_values_videos.to('cuda'),
                            inputs.video_grid_thw.to('cuda'),return_dict=True).pooler_output,0)
                        node=legacy.forest.raw(legacy.features.raw_block(visual,i//2,a['pts_s'],b['pts_s']))
                        encoded[key]=node;pair_count+=1;del inputs,visual
                    node=encoded[key]
                    if node.first!=i//2:raise ValueError('pair index drift')
                    nodes.append(node)
                question=env['prompt'](r['model_visible'])
                if not chunk_checked:
                    check=check_chunk(env,model,builder,nodes,question);save(out/'real_chunk_check.json',check)
                    if not check['passed']:raise ValueError('real chunk check failed; stop pilot')
                    chunk_checked=True
                # All arms read the same immutable CPU feature nodes. No main video
                # language cache is supplied, and every answer starts a fresh cache.
                before=hashlib.sha256(b''.join(n.features.numpy().tobytes() for n in nodes)).hexdigest()
                for arm,selected in [('full_pairwise',nodes),('recent8_pairs',nodes[-8:]),('text_only',[])]:
                    if time.monotonic()-started>1100:raise TimeoutError('pilot inference deadline')
                    start=time.monotonic()
                    try:
                        result=answer(env,model,builder,selected,question)
                        entry=dict(id=r['id'],arm=arm,status='ok',**result)
                    except (ValueError,FloatingPointError) as error:
                        entry=dict(id=r['id'],arm=arm,status='readout_error',error=str(error),text='')
                    entry['elapsed_seconds']=time.monotonic()-start
                    f.write(json.dumps(entry,ensure_ascii=False)+'\n');f.flush();written+=1
                    print(json.dumps(dict(done=written,id=r['id'],arm=arm,text=entry['text'],seconds=entry['elapsed_seconds'])),flush=True)
                    if torch.cuda.memory_allocated()>7.25*2**30:raise MemoryError('GPU allocated budget')
                after=hashlib.sha256(b''.join(n.features.numpy().tobytes() for n in nodes)).hexdigest()
                if before!=after:raise AssertionError('questions modified stored visual features')
                integrity.append(dict(id=r['id'],pairs=len(nodes),frames=len(specs),
                    odd_query_tail_padding=bool(len(specs)%2),feature_bank_unchanged=True))
            del encoded,frames,nodes
            gc.collect();torch.cuda.empty_cache()
    save(out/'completed.json',dict(status='complete' if written==72 else 'partial',records=written,
        questions_requested=24,actual_visual_pair_encodings=pair_count,per_question_integrity=integrity,
        elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),native_whole_video_baseline=False,
        model_revision=profile['model_revision'],model_loads=1))
    del model;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','worker'],required=True);a=p.parse_args()
    out=a.out.resolve()
    if out.parent!=(ROOT/'runs').resolve():raise ValueError('own direct runs directory required')
    (prepare if a.mode=='prepare' else worker)(out)


if __name__=='__main__':main()
