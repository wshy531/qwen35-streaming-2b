"""Fixed native-source reference followed by same-capacity M0/M1/M2.

Uses only existing24 development requests. No annotations in workers, no tuning,
and no claim that the old animal-calibrated event threshold transfers optimally.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from capability_pilot import ROOT,read,save,sha

PREVIOUS=ROOT/'runs/08b_capability_pilot_20261019_01'
REPO=ROOT/'publish/qwen35-streaming'
ACTIVE=REPO/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
DIAG=REPO/'experiments/36tar/streaming-learning/diagnostics'
THRESHOLD=0.047536224126815796


def runtime(stage):
    if stage=='native':
        # Import the untouched original class before any streaming utility can
        # import the identically named package from the active implementation.
        sys.path.insert(0,str(REPO))
        import qwen35_local
    from capability_pilot import imports
    env=imports()
    expected=REPO/'qwen35_local' if stage=='native' else ACTIVE/'qwen35_local'
    actual=Path(sys.modules[env['Qwen3_5ForConditionalGeneration'].__module__].__file__).resolve()
    if actual.parent!=expected.resolve():raise ValueError('wrong original/stream model implementation')
    env['actual_source']=actual
    from first_2b_baseline import complete
    env['complete']=complete
    return env


def detector():
    path=DIAG/'event_memory_v1/memory.py'
    spec=importlib.util.spec_from_file_location('frozen_existing_event_memory',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    cfg=read(DIAG/'forest_runtime_v1/cases.json')
    if cfg['threshold']!=THRESHOLD:raise ValueError('frozen grouping threshold drift')
    return module.distance


def policies(env,nodes):
    from memory_policy import Pair,RecentFIFO,RecentLongHierarchy
    from memory384 import Forest384
    legacy=env['load']();m0=RecentFIFO(8);m1=RecentLongHierarchy(4,4);m2=Forest384()
    distance=detector();previous=None;events=[]
    for node in nodes:
        desc=node.features.mean(0)
        change=None if previous is None else distance(previous,desc)
        starts=previous is None or change>THRESHOLD
        # Candidate ownership is separate from the shared full-reference cache.
        m0.append(Pair(node.first,node.begin,node.end,node.copy()))
        m1.append(Pair(node.first,node.begin,node.end,node.copy()))
        block=legacy.features.raw_block(node.features,node.first,node.begin,node.end)
        m2.append(block,starts_event=starts)
        events.append(dict(pair=node.first,begin=node.begin,end=node.end,adjacent_distance=change,starts_event=starts))
        previous=desc.clone()
    exports=dict(M0=[p.payload.copy() for p in m0.read()],M1=[p.payload.copy() for p in m1.read()],M2=m2.export())
    status={}
    for name,selected in exports.items():
        meta=[n.metadata() for n in selected]
        status[name]=dict(tokens=sum(n['tokens'] for n in meta),nodes=meta,
            candidate_tensor_bytes=sum(n['tensor_bytes'] for n in meta))
        if status[name]['tokens']>384:raise AssertionError('same384Token ceiling')
    status['M1']['representative_span_metadata']=m1.provenance()
    status['M2']['forest']=m2.status()
    status['M2']['detector_state_bytes']=0 if previous is None else previous.numel()*previous.element_size()
    return exports,status,events


def node_digest(nodes):
    h=hashlib.sha256()
    for node in nodes:
        h.update(json.dumps(node.metadata(),sort_keys=True).encode())
        for value in [node.features,node.mass,node.coords]:h.update(value.numpy().tobytes())
    return h.hexdigest()


def decode(env,rows):
    wanted={f['frame_id']:f for r in rows for f in r['model_visible']['frames']}
    if len({r['model_visible']['query_time_s'] for r in rows})!=1:
        raise ValueError('this fixed pilot requires one shared query horizon per video')
    path=ROOT/rows[0]['media_path']
    if sha(path)!=rows[0]['media_sha256']:raise ValueError('media drift')
    packets={};origin=None
    with env['av'].open(str(path)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1
        meta=dict(fps=float(stream.average_rate),total_num_frames=stream.frames)
        for index,frame in enumerate(container.decode(stream)):
            if origin is None:origin=float(frame.time)
            if index not in wanted:continue
            expected=wanted[index];rgb=env['canvas_bytes'](frame);pts=float(frame.time)-origin
            if abs(pts-expected['pts_s'])>1e-7 or hashlib.sha256(rgb).hexdigest()!=expected['rgb_sha256']:
                raise ValueError('shared actual frame identity changed')
            packets[index]=env['FramePacket'](index,pts,env['np'].frombuffer(rgb,dtype='uint8').reshape(192,256,3).copy(),index+1)
            if len(packets)==len(wanted):break
    if len(packets)!=len(wanted):raise ValueError('missing frames')
    if not meta['total_num_frames']:meta['total_num_frames']=max(wanted)+1
    return packets,meta


def encode(env,model,builder,packets,specs):
    torch=env['torch'];legacy=env['load']();nodes=[]
    for i in range(0,len(specs),2):
        a,b=specs[i],specs[min(i+1,len(specs)-1)]
        inputs,_=builder.pair(packets[a['frame_id']],packets[b['frame_id']],0,0,synthetic_padding=i+1==len(specs))
        features=torch.cat(model.model.get_video_features(inputs.pixel_values_videos.to('cuda'),
            inputs.video_grid_thw.to('cuda'),return_dict=True).pooler_output,0)
        nodes.append(legacy.forest.raw(legacy.features.raw_block(features,i//2,a['pts_s'],b['pts_s'])))
        del features,inputs
    return nodes


def native_inputs(env,model,builder,packets,specs,metadata,question,device):
    torch=env['torch'];chosen=[packets[f['frame_id']] for f in specs]
    if len(chosen)%2:chosen.append(chosen[-1])
    meta=dict(metadata,frames_indices=[f.frame_id for f in chosen])
    rendered=builder.template(question)
    native=builder.processor(text=[rendered],videos=[env['np'].stack([f.rgb for f in chosen])],
        video_metadata=[meta],do_resize=False,do_sample_frames=False,return_tensors='pt').to(device)
    if native.input_ids.shape[1]+8>3072:raise ValueError('native full-prefill budget; no truncation')
    oracle=torch.repeat_interleave(native.video_grid_thw,native.video_grid_thw[:,0],0).clone();oracle[:,0]=1
    coords,delta=model.model.get_rope_index(**dict(native,video_grid_thw=oracle))
    native['position_ids']=torch.cat((torch.arange(native.input_ids.shape[1],device=device)[None,None,:],coords),0)
    return native,delta,dict(input_tokens=native.input_ids.shape[1],
        visual_tokens=int((native.input_ids==builder.processor.video_token_id).sum()),
        position_adapter='existing official get_rope_index with per-temporal-group oracle grid; vision grid unchanged',
        actual_RGB_input_unchanged=True,processor_uses_original_fps_and_real_frame_indices=True,
        odd_tail_duplicate=bool(len(specs)%2))


def native_answer(env,model,builder,packets,specs,metadata,question,nodes,kind):
    saved=model.model.rope_deltas
    from readout import context,Limits
    try:
        if kind=='native_original':
            kwargs,delta,meta=native_inputs(env,model,builder,packets,specs,metadata,question,'cuda')
            model.model.rope_deltas=delta
        else:
            embeddings,positions,ids,meta=context(model,builder,nodes,question,limits=Limits(8192,6144,128),max_new_tokens=8)
            if embeddings.shape[1]+8>3072:raise ValueError('original matched full-prefill budget; no truncation')
            kwargs=dict(inputs_embeds=embeddings,position_ids=positions)
        result,first=env['complete'](model,builder.processor,dict(kwargs),8)
        if not env['np'].isfinite(first).all():raise FloatingPointError('nonfinite logits')
        return dict(text=result['answer'],generated_ids=result['generated_ids'],
            input_tokens=meta['input_tokens'],visual_tokens=meta['visual_tokens'],
            first_logits_sha256=hashlib.sha256(first.tobytes()).hexdigest(),
            position_adapter=meta.get('position_adapter','pinned independent pair readout'))
    finally:model.model.rope_deltas=saved


def prepare(stage,out):
    out.mkdir(exist_ok=False)
    requests=read(PREVIOUS/'requests.json')
    if len(requests)!=24:raise ValueError('fixed24 required')
    save(out/'requests.json',requests)
    save(out/'protocol.json',dict(stage=stage,request_sha256=sha(out/'requests.json'),
        previous_request_sha256=sha(PREVIOUS/'requests.json'),model_revision=read(PREVIOUS/'protocol.json')['revision'],
        arms=['native_original','matched_original'] if stage=='native' else ['M0','M1','M2'],
        scope='existing24 development only; not frozen benchmark',
        max_new_tokens=8,model_loads=1,seconds=600,free_RAM_gib=3,free_GPU_gib=5.5,
        visual_cap=384 if stage=='memory' else None,event_threshold=THRESHOLD if stage=='memory' else None,
        candidate_memory_is_separate_from_full_reference_feature_cache=True))


def worker(stage,out):
    import psutil
    if psutil.virtual_memory().available<3*2**30:
        save(out/'blocked.json',dict(reason='preimport3GiB_RAM_guard'));return
    env=runtime(stage);torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    # Verify all pinned legacy modules and weight assets before pretrained load.
    env['load']();profile=env['profile'];modeldir=ROOT/'models/Qwen3.5-0.8B'
    for asset in profile['model_assets']:
        if sha(modeldir/asset['name'])!=asset['sha256']:raise ValueError('asset drift')
    free,_=torch.cuda.mem_get_info();ram=psutil.virtual_memory().available
    guard=dict(free_gpu_bytes=free,free_ram_bytes=ram,model_source=str(env['actual_source']),model_source_sha256=sha(env['actual_source']))
    save(out/'resource_guard.json',guard)
    if free<5.5*2**30 or ram<3*2**30:
        save(out/'blocked.json',dict(reason='preload_GPU_RAM_guard',**guard));return
    started=time.monotonic()
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(modeldir,local_files_only=True,
        dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    save(out/'loaded.json',dict(model_loads=1,source=str(env['actual_source']),source_sha256=sha(env['actual_source'])))
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(modeldir,local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    protocol=read(out/'protocol.json')
    if sha(out/'requests.json')!=protocol['request_sha256']:raise ValueError('request identity drift')
    grouped={}
    for r in read(out/'requests.json'):grouped.setdefault(r['media_path'],[]).append(r)
    written=0;timelines=[];visual_pairs=0
    from capability_pilot import answer
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as f,torch.inference_mode(),env['fp32_reference']():
        for media,rows in grouped.items():
            if time.monotonic()-started>550:raise TimeoutError('fixed stage inference budget')
            packets,metadata=decode(env,rows)
            specs=rows[0]['model_visible']['frames']
            if any(r['model_visible']['frames']!=specs for r in rows):raise ValueError('shared-horizon frames differ')
            nodes=encode(env,model,builder,packets,specs);visual_pairs+=len(nodes)
            if stage=='memory':
                exports,status,events=policies(env,nodes)
                bankhash={name:node_digest(ns) for name,ns in exports.items()}
                timelines.append(dict(media=media,states=status,events=events,
                    full_reference_cache_bytes=sum(n.metadata()['tensor_bytes'] for n in nodes),
                    query_snapshot_tail_padding=bool(len(specs)%2),future_continuation_not_tested_in_this_snapshot_pilot=True))
            for r in rows:
                question=env['prompt'](r['model_visible'])
                for arm in protocol['arms']:
                    begin=time.monotonic()
                    try:
                        result=(native_answer(env,model,builder,packets,specs,metadata,question,nodes,arm)
                            if stage=='native' else answer(env,model,builder,exports[arm],question))
                        entry=dict(id=r['id'],arm=arm,status='ok',**result)
                    except (ValueError,FloatingPointError) as error:
                        entry=dict(id=r['id'],arm=arm,status='readout_error',text='',error=str(error))
                    entry['seconds']=time.monotonic()-begin
                    f.write(json.dumps(entry,ensure_ascii=False)+'\n');f.flush();written+=1
                    if stage=='memory' and any(node_digest(exports[a])!=bankhash[a] for a in protocol['arms']):
                        raise AssertionError('question mutated candidate memory')
                    print(json.dumps(dict(done=written,id=r['id'],arm=arm,text=entry['text'])),flush=True)
                    if torch.cuda.memory_allocated()>7.25*2**30:raise MemoryError('GPU allocated guard')
            if stage=='memory':del exports,status,events
            del nodes,packets;gc.collect();torch.cuda.empty_cache()
    save(out/'timelines.json',timelines)
    save(out/'completed.json',dict(stage=stage,status='complete',records=written,actual_pair_encodings=visual_pairs,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        seconds=time.monotonic()-started,model_loads=1,candidate_query_isolation=stage=='memory',
        frozen_threshold_semantic_transfer_unverified=stage=='memory'))
    del model;gc.collect();torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['native','memory'],required=True)
    p.add_argument('--mode',choices=['prepare','worker'],required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve()
    if out.parent!=(ROOT/'runs').resolve():raise ValueError('own run directory required')
    (prepare if a.mode=='prepare' else worker)(a.stage,out)


if __name__=='__main__':main()
