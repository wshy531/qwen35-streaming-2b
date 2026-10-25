"""Local preparation and an explicitly authorized single Slurm execution."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from ..online_memory_2b.isolation import configure, output_path, write_root
configure()
from ..online_memory_2b.frozen import REPO
from ..online_memory_2b.protocol import read, save, sha, verify_package, verify_model
from ..sember_102_2b.inputs import bundle_path
from ..sember_102_2b.readout import question_prompt
from ..sember_input_diagnostic import runner as old
from ..sember_readout_diagnostic.inference import original_module, native_answer, isolated
from ..sember_readout_diagnostic.plan import digest
from .native import inputs, bounded_answer, watchdog, CONTEXT_NATIVE, QUERY_TILE, WORKER_SECONDS
from .controls import PRESETS, HISTORICAL, geometry, canvas, rotate, schedule, validate_history

HISTORY = 'runs/sember-input-diagnostic-20261024-01'
HISTORY_SHA = {
    'gpu/predictions.jsonl': '314cf842ced3b3f0275c5479f702f80237621ff31ff052e827f5662e3bb91e34',
    'protocol.json': '166664aa1c9f465a8b757c5a05e149349b2b69f28a3a0282cc29a943e29f0d45',
    'cpu/passed.json': '5ed5bb0e1c01fab439b51071ad98a239b7db946d58f90b1c10e806bff33638af',
    'gpu/completed.json': '31b3958e097bbfa4625acad9020be5abe51ca2050dd1733372d13c9599240c20',
}
PROFILE = dict(schema='sember-resolution-v2', presets={k:geometry(k) for k in PRESETS},
    arms=['LS','A06','native','text_only'], fps=1, shifts=list(range(5)), precision='float32',
    scope='same author-evidence frame pairs in all visual arms; oracle diagnostic, not full-stream ranking',
    budget='four recent pairs plus four pair-equivalents of history; grows with resolution',
    resize='historical two-stage letterbox at three 3:4 canvases; no crop/stretch',
    native_attention='FP32 math SDPA, 512-query tiles, full keys/values; no memory compression',
    pretrained_loads=2, simultaneous_models=1, max_new_tokens=8,
    cpus=4, RAM_GiB=48, GPU_GiB_ceiling=72, minimum_free_GPU_GiB=64, slurm_minutes=360,
    context_native=CONTEXT_NATIVE, query_tile=QUERY_TILE, worker_seconds=WORKER_SECONDS,
    context_memory=8192, automatic_retry=False, held_out=False,
    model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc')


def history():
    folder=output_path(write_root()/HISTORY)
    for name, expected in HISTORY_SHA.items():
        if sha(folder/name) != expected: raise ValueError('historical artifact changed: '+name)
    records=old.verify(folder)
    proof=read(folder/'cpu/passed.json'); done=read(folder/'gpu/completed.json')
    for name, expected in proof['hashes'].items():
        if sha(folder/'cpu'/name) != expected: raise ValueError('historical CPU proof changed')
    if (done['status']!='complete' or done['records']!=2040
            or done['CPU_proof_sha256']!=sha(folder/'cpu/passed.json')
            or done['protocol_sha256']!=sha(folder/'protocol.json')):
        raise ValueError('historical completion mismatch')
    rows=[json.loads(s) for s in (folder/'gpu/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    reused=validate_history([r['query'] for r in records],rows)
    return folder, records, reused


def code_lock():
    result=old.code_lock()
    for p in Path(__file__).parent.rglob('*'):
        if p.suffix in ('.py','.sh'): result[p.relative_to(REPO).as_posix()]=sha(p)
    entry=REPO/'scripts/run_2b_sember_resolution.py'
    result[entry.relative_to(REPO).as_posix()]=sha(entry)
    return dict(sorted(result.items()))


def plan(out):
    folder,records,reused=history()
    verify_package()
    out=output_path(out); out.mkdir(parents=True,exist_ok=False)
    save(out/'requests.json',records); save(out/'source_lock.json',code_lock())
    save(out/'historical_reuse.json',reused)
    from .resources import forecast
    save(out/'resources.json',forecast(read(folder/'cpu/inputs.json'),read(folder/'cpu/selections.json')))
    save(out/'protocol.json',dict(PROFILE,schedule=schedule([r['query'] for r in records]),
        historical_sha256=HISTORY_SHA,requests_sha256=sha(out/'requests.json'),
        source_lock_sha256=sha(out/'source_lock.json'),reuse_sha256=sha(out/'historical_reuse.json'),
        resources_sha256=sha(out/'resources.json')))


def verify(out):
    out=output_path(out); p=read(out/'protocol.json')
    folder,records,reused=history()
    from .resources import forecast
    if (any(p.get(k)!=v for k,v in PROFILE.items()) or p['historical_sha256']!=HISTORY_SHA
            or p['schedule']!=schedule([r['query'] for r in records])
            or read(out/'requests.json')!=records or read(out/'historical_reuse.json')!=reused
            or p['requests_sha256']!=sha(out/'requests.json')
            or p['reuse_sha256']!=sha(out/'historical_reuse.json')
            or p['resources_sha256']!=sha(out/'resources.json')
            or read(out/'resources.json')!=forecast(read(folder/'cpu/inputs.json'),read(folder/'cpu/selections.json'))
            or p['source_lock_sha256']!=sha(out/'source_lock.json')
            or read(out/'source_lock.json')!=code_lock()):
        raise ValueError('locked resolution protocol changed')
    return p,records,reused


def decode(env,item,resolution):
    if sha(item['media'])!=item['source']['source_sha256']: raise ValueError('media SHA changed')
    expected={r['frame_id']:r for r in item['frames']}; packets={}
    origin=last=None
    with env['av'].open(str(item['media'])) as container:
        stream=container.streams.video[0]; stream.codec_context.thread_count=1
        if not stream.average_rate or not stream.frames or not stream.duration:
            raise ValueError('missing real video metadata')
        meta=dict(fps=float(stream.average_rate),total_num_frames=stream.frames,
                  duration=float(stream.duration*stream.time_base))
        for i,frame in enumerate(container.decode(stream)):
            if frame.time is None: raise ValueError('missing PTS')
            if origin is None: origin=float(frame.time)
            pts=float(frame.time)-origin
            if not math.isfinite(pts) or (last is not None and pts<=last): raise ValueError('nonmonotonic PTS')
            last=pts
            if pts>item['source']['max_query_time_s']+1e-9: break
            if i not in expected: continue
            low=canvas(frame,'low')
            check=dict(frame_id=i,pts_s=pts,rgb_sha256=hashlib.sha256(low.tobytes()).hexdigest())
            if check!=expected[i]: raise ValueError('historical sampling RGB/PTS drift')
            rgb=env['np'].asarray(low if resolution=='low' else canvas(frame,resolution)).copy()
            packets[i]=env['FramePacket'](i,pts,rgb,i+1)
    if set(packets)!=set(expected): raise ValueError('missing real sampled frames')
    return packets,meta


def selection(item,request,packets):
    chosen=old.previous.get_selection(item,request)
    rows=chosen['evidence']
    if not rows or len(rows)%2: raise ValueError('complete real evidence pairs required')
    result=[packets[r['frame_id']] for r in rows]
    if any(p.timestamp>request['query']['query_time_s'] for p in result): raise ValueError('future frame')
    rows=[dict(frame_id=p.frame_id,pts_s=p.timestamp,
        rgb_sha256=hashlib.sha256(p.rgb.tobytes()).hexdigest()) for p in result]
    return result,rows


def cpu(out,model_path):
    old.previous.allocation()
    out=output_path(out); _,records,_=verify(out)
    model_path=output_path(model_path); verify_package(); verify_model(model_path)
    directory=out/'cpu'; directory.mkdir(exist_ok=False)
    env,builder=old.setup(model_path); torch=env['torch']; watchdog(env,out)
    if torch.cuda.is_available() or torch.cuda.is_initialized(): raise ValueError('CPU GPU visibility')
    with (directory/'contracts.txt').open('x',encoding='utf-8') as log:
        for folder in ('sember_102_2b/tests','sember_readout_diagnostic/tests','sember_input_diagnostic/tests',
                       'sember_resolution_2b/tests','sember_resolution_2b/cluster_tests'):
            subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',
                str(REPO/'experiments/wshy'/folder),'-v'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                check=True,timeout=600)
    from ..online_memory_2b.receipts import successful_counts
    successful_counts((directory/'contracts.txt').read_text(),[11,17,13,20,11])
    torch.manual_seed(123); identity=old.previous.software(env)
    original=original_module()
    model=original.Qwen3_5ForConditionalGeneration(original.Qwen3_5Config(
        **env['tiny_config'](builder.processor),attn_implementation='sdpa')).eval()
    hfolder,_,_=history()
    checks_old={(r['id'],r['arm'],r['shift']):r for r in read(hfolder/'cpu/inputs.json')}
    chosen_old={r['id']:r for r in read(hfolder/'cpu/selections.json')}
    checks=[]; selections=[]; tiny=[]; tiny_examples={}
    projected={(r['id'],r['shift']):r['input_tokens'] for r in read(out/'resources.json')['new_high_inputs']}
    with torch.inference_mode(),env['fp32_reference']():
        for resolution in PRESETS:
            for item in old.previous.sampling():
                packets,meta=decode(env,item,resolution)
                for request in records:
                    q=request['query']
                    if q['source_path']!=item['source']['source_path']: continue
                    selected,rows=selection(item,request,packets)
                    if resolution in ('low','mid') and rows!=chosen_old[q['id']]['B' if resolution=='low' else 'R']:
                        raise ValueError('historical selected frames drift')
                    selections.append(dict(id=q['id'],resolution=resolution,frames=rows))
                    for shift in range(5):
                        rq,_=rotate(q,shift)
                        inp=inputs(env,model,builder,selected,question_prompt(rq),meta)
                        fp=old.fingerprint(inp)
                        if resolution=='high' and fp['input_tokens']!=projected[q['id'],shift]:
                            raise ValueError('high-resolution token forecast differs from actual processor')
                        if resolution!='high':
                            reference=checks_old[q['id'],'B' if resolution=='low' else 'R',shift]
                            if any(fp[k]!=reference[k] for k in fp): raise ValueError('historical processor drift')
                        checks.append(dict(id=q['id'],resolution=resolution,shift=shift,**fp))
                        del inp
                    if len(tiny)<3 and resolution==list(PRESETS)[len(tiny)]:
                        tiny_examples[resolution]=(selected[:2],dict(meta))
                        inp=inputs(env,model,builder,selected[:4],question_prompt(q),meta)
                        from .checks import tiny_native_parity
                        tiny_native_parity(env,model,inp)
                        ans=isolated(env,model,lambda:bounded_answer(env,model,builder,inp))
                        tiny.append(dict(arm='native',resolution=resolution,generated_ids=ans['generated_ids']))
                        del inp
                del packets
            print(json.dumps(dict(stage='cpu',resolution=resolution,inputs=len(checks))),flush=True)
        for request in records:
            q=request['query']
            for shift in range(5):
                rq,_=rotate(q,shift)
                inp=inputs(env,model,builder,[],question_prompt(rq),{})
                fp=old.fingerprint(inp); reference=checks_old[q['id'],'T',shift]
                if any(fp[k]!=reference[k] for k in fp): raise ValueError('historical text input drift')
                checks.append(dict(id=q['id'],resolution='none',shift=shift,**fp))
                del inp
        del model; gc.collect()
        # Separate tiny streaming model; no pretrained weights are loaded.
        model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](
            **env['tiny_config'](builder.processor),attn_implementation='sdpa')).eval()
        from .checks import tiny_memory
        tiny.extend(tiny_memory(env,model,builder,tiny_examples))
        del model
    if len(checks)!=2040 or len(selections)!=306 or len(tiny)!=9: raise ValueError('incomplete CPU checks')
    verify(out)
    save(directory/'inputs.json',checks); save(directory/'selections.json',selections)
    save(directory/'passed.json',dict(status='passed',pretrained_loads=0,environment=identity,
        tiny_answers=tiny,model_path=str(model_path),model_manifest_sha256=sha(model_path/'DOWNLOAD_MANIFEST.json'),
        protocol_sha256=sha(out/'protocol.json'),
        hashes={n:sha(directory/n) for n in ('contracts.txt','inputs.json','selections.json')}))


def gpu(out):
    old.previous.allocation(gpu=True)
    out=output_path(out); protocol,records,reused=verify(out); proof=read(out/'cpu/passed.json')
    if (proof['status']!='passed' or proof['pretrained_loads']!=0
            or proof['protocol_sha256']!=sha(out/'protocol.json')
            or any(sha(out/'cpu'/n)!=h for n,h in proof['hashes'].items())):
        raise ValueError('CPU proof mismatch')
    model_path=output_path(proof['model_path']); verify_model(model_path)
    if proof['model_manifest_sha256']!=sha(model_path/'DOWNLOAD_MANIFEST.json'): raise ValueError('model changed')
    directory=out/'gpu'; directory.mkdir(exist_ok=False)
    env,builder=old.setup(model_path); torch=env['torch']; watchdog(env,out)
    if torch.cuda.device_count()!=1 or old.previous.software(env)!=proof['environment']:
        raise ValueError('GPU/environment identity')
    if torch.cuda.mem_get_info()[0]<PROFILE['minimum_free_GPU_GiB']*2**30: raise MemoryError('64 GiB free GPU required')
    torch.cuda.set_per_process_memory_fraction(min(1.,PROFILE['GPU_GiB_ceiling']*2**30/torch.cuda.get_device_properties(0).total_memory))
    checks={(r['id'],r['resolution'],r['shift']):r for r in read(out/'cpu/inputs.json')}
    selections={(r['id'],r['resolution']):r['frames'] for r in read(out/'cpu/selections.json')}
    references={(r['id'],r['arm'],r['resolution'],r['shift']):r for r in reused}
    replay=set(protocol['schedule']['replay_ids'])
    expected={(q['query']['id'],a,r,s) for q in records for a in ('LS','A06') for r in PRESETS for s in range(5)}
    expected.update((q['query']['id'],'native','high',s) for q in records for s in range(5))
    expected.update((qid,a,r,s) for qid in replay for a,r in
                    (('native','low'),('native','mid'),('text_only','none')) for s in range(5))
    expected_visual_calls=690+sum(len(frames)//2 for frames in selections.values())
    seen=set(); new_tokens=0; count=0; calls=[0]; peaks={}; loads=0; started=time.monotonic()
    def hook(*unused): calls[0]+=1
    def load(cls,name):
        nonlocal loads
        if loads>=2: raise RuntimeError('model load cap')
        loads+=1; torch.manual_seed(123)
        save(directory/f'load_{loads}_started.json',dict(kind=name,implementation=cls.__module__))
        m=cls.from_pretrained(model_path,local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
        if (m.config.text_config.hidden_size!=2048 or m.config.text_config.num_attention_heads!=8
                or any(p.is_floating_point() and p.dtype!=torch.float32 for p in m.parameters())):
            raise ValueError('FP32 2B identity')
        return m
    def write(stream,row):
        nonlocal count,new_tokens
        key=(row['id'],row['arm'],row['resolution'],row['shift'])
        if key not in expected or key in seen or count>=3840 or not 1<=len(row['generated_ids'])<=8:
            raise ValueError('answer cap/duplicate/unplanned key')
        seen.add(key); count+=1; new_tokens+=len(row['generated_ids'])
        if new_tokens>30720 or torch.cuda.max_memory_allocated()>PROFILE['GPU_GiB_ceiling']*2**30: raise RuntimeError('resource/token cap')
        stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
    def row(q,arm,resolution,shift,order,frames,result,seconds,mode,**extra):
        return dict(id=q['id'],source_path=q['source_path'],query_time_s=q['query_time_s'],
            arm=arm,resolution=resolution,shift=shift,display_to_original=order,status='ok',
            frames_sha256=digest(frames),provenance=mode,answer_seconds=seconds,**result,**extra)
    with (directory/'predictions.jsonl').open('x',encoding='utf-8') as stream, \
         (directory/'events.jsonl').open('x',encoding='utf-8') as events,torch.inference_mode(),env['fp32_reference']():
        model=load(original_module().Qwen3_5ForConditionalGeneration,'original'); handle=model.model.visual.register_forward_hook(hook)
        try:
            for resolution in ('low','mid','none','high'):
                torch.cuda.reset_peak_memory_stats()
                items=old.previous.sampling()
                ordered=records
                if resolution=='high':
                    # Expose the largest real native workload first, after reuse gates.
                    size={r['query']['id']:checks[r['query']['id'],'high',0]['input_tokens'] for r in records}
                    ordered=sorted(records,key=lambda r:(-size[r['query']['id']],r['query']['id']))
                    items=sorted(items,key=lambda i:-max(size[r['query']['id']] for r in records
                        if r['query']['source_path']==i['source']['source_path']))
                for item in items:
                    packets,meta=({}, {}) if resolution=='none' else decode(env,item,resolution)
                    for request in ordered:
                        q=request['query']
                        if q['source_path']!=item['source']['source_path'] or (resolution!='high' and q['id'] not in replay): continue
                        selected,frames=([],[]) if resolution=='none' else selection(item,request,packets)
                        if resolution!='none' and frames!=selections[q['id'],resolution]: raise ValueError('CPU/GPU frame drift')
                        for shift in range(5):
                            rq,order=rotate(q,shift); prompt=question_prompt(rq)
                            inp=inputs(env,model,builder,selected,prompt,meta); fp=old.fingerprint(inp)
                            if any(fp[k]!=checks[q['id'],resolution,shift][k] for k in fp): raise ValueError('CPU/GPU input drift')
                            torch.cuda.synchronize(); tick=time.monotonic(); before=calls[0]
                            result=isolated(env,model,lambda:bounded_answer(env,model,builder,inp))
                            torch.cuda.synchronize(); seconds=time.monotonic()-tick; del inp
                            if calls[0]-before!=(0 if resolution=='none' else 1): raise ValueError('native visual calls')
                            arm='text_only' if resolution=='none' else 'native'
                            answer=row(q,arm,resolution,shift,order,frames,result,seconds,
                                'new' if resolution=='high' else 'replay',pixels_sha256=fp['pixels_sha256'])
                            if resolution!='high':
                                ref=references[q['id'],arm,resolution,shift]
                                for field in ('generated_ids','text','input_ids_sha256','position_ids_sha256','frames_sha256','pixels_sha256'):
                                    if answer[field]!=ref[field]: raise ValueError('historical answer replay drift: '+field)
                            write(stream,answer)
                    del packets
                peaks['native_'+resolution]=torch.cuda.max_memory_allocated()
                if resolution=='none':
                    if count!=270: raise ValueError('incomplete historical replay')
                    save(directory/'reuse_enabled.json',dict(status='passed',replayed=270,reused=1530))
                print(json.dumps(dict(stage='gpu-native',resolution=resolution,answers=count)),flush=True)
        finally:
            handle.remove(); del model; gc.collect(); torch.cuda.empty_cache()
        from .memory import Bank,raw,answer as memory_answer
        model=load(env['Qwen3_5ForConditionalGeneration'],'streaming'); handle=model.model.visual.register_forward_hook(hook)
        try:
            for resolution in PRESETS:
                torch.cuda.reset_peak_memory_stats()
                for item in old.previous.sampling():
                    packets,meta=decode(env,item,resolution)
                    for request in records:
                        q=request['query']
                        if q['source_path']!=item['source']['source_path']: continue
                        selected,frames=selection(item,request,packets)
                        if frames!=selections[q['id'],resolution]: raise ValueError('memory frame drift')
                        banks={a:Bank(a,resolution) for a in ('LS','A06')}
                        encode_seconds=0.; updates={a:0. for a in banks}; highwater={a:0 for a in banks}
                        for j in range(0,len(selected),2):
                            inp=inputs(env,model,builder,selected[j:j+2],'Return only A.',meta)
                            torch.cuda.synchronize(); tick=time.monotonic(); before=calls[0]
                            features=torch.cat(model.model.get_video_features(inp.pixel_values_videos.to('cuda'),
                                inp.video_grid_thw.to('cuda'),return_dict=True).pooler_output,0)
                            node=raw(features,j//2,selected[j].timestamp,selected[j+1].timestamp,resolution)
                            del features,inp
                            torch.cuda.synchronize(); encode_seconds+=time.monotonic()-tick
                            if calls[0]-before!=1: raise ValueError('pair encoded other than exactly once')
                            for arm,bank in banks.items():
                                tick=time.monotonic(); bank.append(node); updates[arm]+=time.monotonic()-tick
                                highwater[arm]=max(highwater[arm],bank.status()['resident_bytes'])
                            del node
                        states={a:b.digest() for a,b in banks.items()}
                        for arm,bank in banks.items():
                            for shift in range(5):
                                rq,order=rotate(q,shift); torch.cuda.synchronize(); tick=time.monotonic(); before=calls[0]
                                result=memory_answer(env,model,builder,bank,question_prompt(rq))
                                torch.cuda.synchronize(); seconds=time.monotonic()-tick
                                if calls[0]!=before or any(b.digest()!=states[a] for a,b in banks.items()):
                                    raise ValueError('QA changed memory or encoded video')
                                write(stream,row(q,arm,resolution,shift,order,frames,result,seconds,'new',
                                    encode_seconds=encode_seconds,bank_update_seconds=updates[arm],
                                    bank_peak_bytes=highwater[arm],encoded_pairs=len(selected)//2))
                        # Keep both banks alive until BOTH methods finish all rotations.
                        # Resetting LS earlier would invalidate A06's cross-bank guard.
                        for arm,bank in banks.items():
                            event=dict(id=q['id'],arm=arm,resolution=resolution,state=bank.status(),
                                encode_seconds=encode_seconds,update_seconds=updates[arm],bank_peak_bytes=highwater[arm])
                            bank.reset()
                            if bank.digest()!=Bank(arm,resolution).digest() or bank.status()['resident_bytes']:
                                raise RuntimeError('reset retained state')
                            event['reset_passed']=True; events.write(json.dumps(event)+'\n'); events.flush()
                        del banks,selected
                    del packets
                peaks['memory_'+resolution]=torch.cuda.max_memory_allocated()
                print(json.dumps(dict(stage='gpu-memory',resolution=resolution,answers=count)),flush=True)
            peaks['overall']=max(peaks.values())
        finally:
            handle.remove(); del model; gc.collect(); torch.cuda.empty_cache()
    if count!=3840 or loads!=2 or seen!=expected or calls[0]!=expected_visual_calls:
        raise ValueError('incomplete execution or incorrect visual-forward count')
    verify(out)
    save(directory/'completed.json',dict(status='complete',new_answers=count,new_generated_tokens=new_tokens,
        reused_answers=1530,unique_analysis_answers=5100,pretrained_loads=loads,peak_GPU_bytes=peaks,
        seconds=time.monotonic()-started,visual_forward_calls=calls[0],expected_visual_calls=expected_visual_calls,
        protocol_sha256=sha(out/'protocol.json'),CPU_proof_sha256=sha(out/'cpu/passed.json'),
        predictions_sha256=sha(directory/'predictions.jsonl'),events_sha256=sha(directory/'events.jsonl')))


def score(out):
    from .scoring import build
    out=output_path(out); _,records,reused=verify(out); done=read(out/'gpu/completed.json')
    if (done['status']!='complete' or done['protocol_sha256']!=sha(out/'protocol.json')
            or done['CPU_proof_sha256']!=sha(out/'cpu/passed.json')
            or done['predictions_sha256']!=sha(out/'gpu/predictions.jsonl')
            or done['events_sha256']!=sha(out/'gpu/events.jsonl')): raise ValueError('completion proof mismatch')
    fresh=[json.loads(s) for s in (out/'gpu/predictions.jsonl').read_text().splitlines()]
    summary,details=build([r['query'] for r in records],read(bundle_path()/'labels-offline-only.json'),reused,fresh)
    directory=out/'score'; directory.mkdir(exist_ok=False)
    save(directory/'summary.json',summary); save(directory/'details.json',details)
