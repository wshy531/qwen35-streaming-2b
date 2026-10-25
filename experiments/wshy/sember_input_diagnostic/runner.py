"""Approved 102-question, four-arm, five-position native-model diagnostic."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from ..online_memory_2b.isolation import configure, output_path, write_root
configure()
from ..online_memory_2b.frozen import REPO, runtime
from ..online_memory_2b.protocol import read, save, sha, verify_model, verify_package
from ..sember_102_2b.inputs import base_inputs, bundle_path
from ..sember_102_2b.readout import complete_question_runtime, question_prompt
from ..sember_readout_diagnostic import runner as previous
from ..sember_readout_diagnostic.plan import digest, select
from ..sember_readout_diagnostic.inference import original_module, native_answer, isolated, tensor_sha
from .controls import ARMS, CONTEXT_TOKENS, rotate, matched_dense, canvas

OLD = 'runs/sember-readout-diagnostic-20261024-01'
OLD_SHA = {
    'requests.json': '509b38da065dc5c0fe8d43d23ac6ea75991112936db4a630c1765199501033c6',
    'cpu/selections.json': '36d447984dff797df073630cf930e16254c9a6a8fd176a2127cbad37f73bf4b7',
    'cpu/inputs.json': 'ae1bb40d26766aa20d7c47f7e327f2e46ff6a5b42bd32ca7f2ef466e6dd9800d',
    'gpu/predictions.jsonl': '44a8da303a0111aab288c59dbf9a32cff3baf76cd087b5bcd2a312e62ce7dfcf',
}
PROFILE = dict(schema='sember-input-diagnostic-v1', questions=102, sources=18,
    arms=list(ARMS), shifts=list(range(5)), maximum_answers=2040, max_new_tokens=8,
    maximum_generated_tokens=16320, context_tokens=CONTEXT_TOKENS,
    canvas_B_F=[192,256], canvas_R=[288,384], fps_B_R=1, target_fps_F=2,
    dense_rule='union old anchors and 2FPS candidates inside old endpoints; odd: omit last new frame',
    precision='float32', model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc',
    pretrained_loads=1, cpus=4, RAM_GiB=48, slurm_minutes=240, worker_seconds=14100,
    minimum_free_GPU_GiB=64, maximum_GPU_GiB=72, maximum_RSS_GiB=40,
    automatic_retry=False, held_out=False, compression=False,
    native_scope='original model full forward with explicit temporal-grid position adapter')


def code_lock():
    result = previous.code_lock()
    for p in Path(__file__).parent.rglob('*'):
        if p.suffix in ('.py', '.sh'):
            result[p.relative_to(REPO).as_posix()] = sha(p)
    p = REPO/'scripts/run_2b_sember_input_diagnostic.py'
    result[p.relative_to(REPO).as_posix()] = sha(p)
    return dict(sorted(result.items()))


def old_inputs():
    folder = output_path(write_root()/OLD)
    for name, expected in OLD_SHA.items():
        if sha(folder/name) != expected:
            raise ValueError('historical input changed: '+name)
    records = read(folder/'requests.json')
    if [r['query'] for r in records] != base_inputs()[0]:
        raise ValueError('original 102 requests differ')
    return folder, records


def plan(out):
    out = output_path(out)
    _, records = old_inputs()
    out.mkdir(parents=True, exist_ok=False)
    save(out/'requests.json', records)
    save(out/'source_lock.json', code_lock())
    save(out/'protocol.json', dict(PROFILE, requests_sha256=sha(out/'requests.json'),
        source_lock_sha256=sha(out/'source_lock.json'), old_sha=OLD_SHA))


def verify(out):
    p = read(out/'protocol.json')
    if (any(p.get(k) != v for k,v in PROFILE.items()) or p['old_sha'] != OLD_SHA
            or p['requests_sha256'] != sha(out/'requests.json')
            or p['source_lock_sha256'] != sha(out/'source_lock.json')
            or read(out/'source_lock.json') != code_lock()):
        raise ValueError('locked diagnostic changed')
    _, records = old_inputs()
    if records != read(out/'requests.json'):
        raise ValueError('request identity changed')
    return records


def watchdog(env, out):
    """Hard wall-time and process-tree resident-memory cap, including preflight."""
    deadline = float(os.environ['QWEN_DIAGNOSTIC_STARTED']) + PROFILE['worker_seconds']
    def check():
        while True:
            try:
                proc = env['psutil'].Process()
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except env['psutil'].NoSuchProcess:
                        pass
                if time.time() > deadline or rss > 40*2**30:
                    reason = dict(status='failed', reason='wall-time or process-tree RSS cap',
                                  rss_bytes=rss, time=time.time())
                    with (out/'resource_failure.json').open('x') as f:
                        json.dump(reason, f)
                    os._exit(124)
            except Exception as exc:
                print('resource monitor failure: '+repr(exc), file=sys.stderr, flush=True)
                os._exit(125)
            time.sleep(1)
    threading.Thread(target=check, daemon=True).start()


def decode(env, item):
    """One original decode; retain low dense pixels and high old-anchor pixels."""
    expected = {r['frame_id']:r for r in item['frames']}
    low, high, candidates, verified = {}, {}, [], set()
    due, origin, last = 0., None, None
    horizon = item['source']['max_query_time_s']
    with env['av'].open(str(item['media'])) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if not stream.average_rate or not stream.duration or not stream.frames:
            raise ValueError('missing source timing metadata')
        metadata = dict(fps=float(stream.average_rate), total_num_frames=stream.frames,
                        duration=float(stream.duration*stream.time_base))
        for index, frame in enumerate(container.decode(stream)):
            if frame.time is None:
                raise ValueError('missing real PTS')
            if origin is None:
                origin = float(frame.time)
            pts = float(frame.time)-origin
            if not math.isfinite(pts) or (last is not None and pts <= last):
                raise ValueError('nonmonotonic source PTS')
            last = pts
            if pts > horizon+1e-9:
                break
            dense = pts+1e-9 >= due
            if dense:
                due = (math.floor(pts*2+1e-9)+1)/2
            if not dense and index not in expected:
                continue
            rgb = env['np'].asarray(canvas(frame,192,256)).copy()
            row = dict(frame_id=index, pts_s=pts, rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
            packet = env['FramePacket'](index, pts, rgb, index+1)
            low[index] = packet
            if dense:
                candidates.append(row)
            if index in expected:
                if row != expected[index]:
                    raise ValueError('old 1FPS RGB/PTS replay differs')
                verified.add(index)
                high[index] = env['FramePacket'](index, pts,
                    env['np'].asarray(canvas(frame,288,384)).copy(), index+1)
    if verified != set(expected):
        raise ValueError('missing original frames')
    return low, high, candidates, metadata


def selection(item, request, candidates, high):
    old = select(item['frames'], request['query']['query_time_s'], request['evidence_interval'])
    anchors = old['evidence']
    if not anchors:
        raise ValueError('baseline evidence unexpectedly empty')
    dense, omitted = matched_dense(anchors, candidates)
    r = [dict(row, rgb_sha256=hashlib.sha256(high[row['frame_id']].rgb.tobytes()).hexdigest())
         for row in anchors]
    return dict(id=request['query']['id'], B=anchors, R=r, F=dense, T=[],
        omitted_new_frame=omitted, new_frames=len(dense)-len(anchors),
        support=[anchors[0]['pts_s'],anchors[-1]['pts_s']],
        dense_intervals=[b['pts_s']-a['pts_s'] for a,b in zip(dense,dense[1:])])


def inputs(env, model, builder, packets, prompt, metadata):
    torch = env['torch']
    if not packets:
        rendered = builder.processor.apply_chat_template([dict(role='user',content=prompt)],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        native = builder.processor.tokenizer(rendered, add_special_tokens=False, return_tensors='pt')
        n = native.input_ids.shape[1]
        native['position_ids'] = torch.arange(n)[None,None,:].expand(4,1,-1)
    else:
        if len(packets)%2:
            raise ValueError('unpaired real frame')
        meta = dict(metadata,frames_indices=[p.frame_id for p in packets])
        native = builder.processor(text=[builder.template(prompt)],
            videos=[env['np'].stack([p.rgb for p in packets])], video_metadata=[meta],
            do_resize=False, do_sample_frames=False, return_tensors='pt')
        h,w = packets[0].rgb.shape[:2]
        if native.video_grid_thw.tolist() != [[len(packets)//2,h//16,w//16]]:
            raise ValueError('processor changed declared grid')
        visual = int((native.input_ids==builder.processor.video_token_id).sum())
        if visual != len(packets)//2*(h//32)*(w//32):
            raise ValueError('unexpected visual token count')
        grid = torch.repeat_interleave(native.video_grid_thw,native.video_grid_thw[:,0],0).clone()
        grid[:,0] = 1
        coords,_ = model.model.get_rope_index(**dict(native,video_grid_thw=grid))
        native['position_ids'] = torch.cat((torch.arange(native.input_ids.shape[1])[None,None,:],coords),0)
        timestamps = builder.processor._calculate_timestamps(meta['frames_indices'],meta['fps'],2)
        if (len(timestamps)!=len(packets)//2 or max(abs(t-(a.timestamp+b.timestamp)/2)
                for t,a,b in zip(timestamps,packets[::2],packets[1::2])) > 1/meta['fps']+1e-6):
            raise ValueError('processor timestamps do not match real PTS')
    if native.input_ids.shape[1]+8 > CONTEXT_TOKENS:
        raise ValueError('24K context exceeded; no truncation or fallback')
    return native


def fingerprint(native):
    return dict(input_tokens=int(native.input_ids.shape[1]),
        input_ids_sha256=tensor_sha(native.input_ids),
        position_ids_sha256=tensor_sha(native.position_ids),
        pixels_sha256=tensor_sha(native.pixel_values_videos) if 'pixel_values_videos' in native else None)


def setup(model_path):
    env = complete_question_runtime(runtime())
    env['torch'].set_num_threads(1)
    builder = env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(
        model_path,local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    return env,builder


def cpu(out, model_path):
    previous.allocation()
    out,model_path = output_path(out),output_path(model_path)
    records = verify(out)
    verify_package(); verify_model(model_path)
    directory = out/'cpu'; directory.mkdir(exist_ok=False)
    with (directory/'contracts.txt').open('x',encoding='utf-8') as log:
        for folder in ('sember_102_2b','sember_readout_diagnostic','sember_input_diagnostic'):
            subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',
                str(REPO/'experiments/wshy'/folder/'tests'),'-v'],cwd=REPO,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    env,builder = setup(model_path); torch=env['torch']
    watchdog(env,out)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise ValueError('CPU GPU visibility')
    identity=previous.software(env)
    torch.manual_seed(123)
    original=original_module()
    model=original.Qwen3_5ForConditionalGeneration(original.Qwen3_5Config(
        **env['tiny_config'](builder.processor),attn_implementation='sdpa')).eval()
    olddir,_=old_inputs()
    oldchecks={r['id']:r for r in read(olddir/'cpu/inputs.json')}
    oldselections={r['id']:r for r in read(olddir/'cpu/selections.json')}
    allchecks, selections, tiny=[],[],[]
    with torch.inference_mode(),env['fp32_reference']():
        for item in previous.sampling():
            low,high,candidates,meta=decode(env,item)
            for request in records:
                q=request['query']
                if q['source_path']!=item['source']['source_path']: continue
                chosen=selection(item,request,candidates,high); selections.append(chosen)
                if chosen['B']!=oldselections[q['id']]['evidence']:
                    raise ValueError('old evidence frame set differs')
                for arm in ARMS:
                    bank=high if arm=='R' else low
                    packets=[bank[f['frame_id']] for f in chosen[arm]]
                    for shift in range(5):
                        rq,_=rotate(q,shift); prompt=question_prompt(rq)
                        native=inputs(env,model,builder,packets,prompt,meta)
                        check=fingerprint(native)
                        if arm=='B' and shift==0:
                            old=oldchecks[q['id']]['evidence']
                            if (check['pixels_sha256']!=old['pixels_sha256'] or
                                    check['position_ids_sha256']!=old['native_position_ids_sha256'] or
                                    check['input_tokens']!=old['input_tokens']):
                                raise ValueError('old baseline CPU input replay differs')
                        allchecks.append(dict(id=q['id'],arm=arm,shift=shift,**check))
                        del native
                    if len(tiny)<4 and arm==ARMS[len(tiny)]:
                        native=inputs(env,model,builder,packets[:4],question_prompt(q),meta)
                        answer=isolated(env,model,lambda:native_answer(env,model,builder,native))
                        tiny.append(dict(arm=arm,generated_ids=answer['generated_ids']))
                        del native
            del low,high,candidates
            print(json.dumps(dict(stage='cpu',questions=len(selections),inputs=len(allchecks))),flush=True)
    if len(allchecks)!=2040 or len(selections)!=102 or len(tiny)!=4:
        raise ValueError('incomplete preflight')
    verify(out)
    save(directory/'selections.json',selections); save(directory/'inputs.json',allchecks)
    save(directory/'passed.json',dict(status='passed',pretrained_loads=0,questions=102,sources=18,
        tiny_answers=tiny,environment=identity,model_path=str(model_path),
        model_manifest_sha256=sha(model_path/'DOWNLOAD_MANIFEST.json'),
        protocol_sha256=sha(out/'protocol.json'),
        hashes={name:sha(directory/name) for name in ('selections.json','inputs.json','contracts.txt')}))


def gpu(out):
    previous.allocation(gpu=True)
    out=output_path(out); records=verify(out); proof=read(out/'cpu/passed.json')
    if (proof['status']!='passed' or proof['pretrained_loads']!=0 or
            proof['protocol_sha256']!=sha(out/'protocol.json') or
            any(sha(out/'cpu'/name)!=h for name,h in proof['hashes'].items())):
        raise ValueError('CPU proof mismatch')
    model_path=output_path(proof['model_path']); verify_model(model_path)
    if proof['model_manifest_sha256']!=sha(model_path/'DOWNLOAD_MANIFEST.json'):
        raise ValueError('model changed')
    directory=out/'gpu'; directory.mkdir(exist_ok=False)
    env,builder=setup(model_path); torch=env['torch']; watchdog(env,out)
    if torch.cuda.device_count()!=1 or previous.software(env)!=proof['environment']:
        raise ValueError('environment or GPU count mismatch')
    if torch.cuda.mem_get_info()[0]<64*2**30:
        raise MemoryError('64 GiB free GPU required')
    torch.cuda.set_per_process_memory_fraction(min(1.,72*2**30/torch.cuda.get_device_properties(0).total_memory))
    torch.manual_seed(123)
    checks={(r['id'],r['arm'],r['shift']):r for r in read(out/'cpu/inputs.json')}
    selected={r['id']:r for r in read(out/'cpu/selections.json')}
    olddir,_=old_inputs()
    # Reference is consulted only after each baseline answer, never passed to the model.
    historical={r['id']:r for r in map(json.loads,(olddir/'gpu/predictions.jsonl').read_text().splitlines())
                if r['arm']=='native_evidence'}
    cls=original_module().Qwen3_5ForConditionalGeneration
    save(directory/'load_started.json',dict(attempt=1,class_name=cls.__module__))
    started=time.monotonic()
    model=cls.from_pretrained(model_path,local_files_only=True,dtype=torch.float32,
                             device_map='cuda',attn_implementation='sdpa').eval()
    if model.config.text_config.hidden_size!=2048 or any(
            p.is_floating_point() and p.dtype!=torch.float32 for p in model.parameters()):
        raise ValueError('FP32 2B required')
    calls=[0]
    def hook(*args): calls[0]+=1
    handle=model.model.visual.register_forward_hook(hook)
    seen=set(); generated=0; replayed=0; peak_rss=0
    try:
        with (directory/'predictions.jsonl').open('x',encoding='utf-8') as stream, \
                torch.inference_mode(),env['fp32_reference']():
            # Replay all 102 original-order baseline answers before extending.
            for phase in ('replay','controls'):
                for item in previous.sampling():
                    low,high,candidates,meta=decode(env,item)
                    for request in records:
                        q=request['query']
                        if q['source_path']!=item['source']['source_path']: continue
                        chosen=selection(item,request,candidates,high)
                        if chosen!=selected[q['id']]: raise ValueError('CPU/GPU frames drift')
                        tasks=[('B',0)] if phase=='replay' else [(a,s) for a in ARMS for s in range(5) if (a,s)!=('B',0)]
                        for arm,shift in tasks:
                            bank=high if arm=='R' else low
                            packets=[bank[f['frame_id']] for f in chosen[arm]]
                            rq,order=rotate(q,shift); prompt=question_prompt(rq)
                            native=inputs(env,model,builder,packets,prompt,meta)
                            check=fingerprint(native); key=(q['id'],arm,shift)
                            if check!={k:v for k,v in checks[key].items() if k not in ('id','arm','shift')}:
                                raise ValueError('CPU/GPU input drift')
                            before=calls[0]
                            answer=isolated(env,model,lambda:native_answer(env,model,builder,native))
                            del native
                            if calls[0]-before!=(0 if arm=='T' else 1):
                                raise ValueError('unexpected visual forward count')
                            if key in seen or len(answer['generated_ids'])>8 or len(seen)>=2040:
                                raise ValueError('answer cap or duplicate')
                            seen.add(key); generated+=len(answer['generated_ids'])
                            if generated>16320: raise ValueError('generated token cap')
                            row=dict(id=q['id'],source_path=q['source_path'],query_time_s=q['query_time_s'],
                                arm=arm,shift=shift,display_to_original=order,status='ok',
                                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                frames_sha256=digest(chosen[arm]),pixels_sha256=check['pixels_sha256'],
                                visual_forward_calls=calls[0]-before,**answer)
                            stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
                            if phase=='replay':
                                old=historical[q['id']]
                                for field in ('input_ids_sha256','position_ids_sha256','generated_ids','text','frames_sha256'):
                                    if row[field]!=old[field]: raise ValueError('historical replay differs: '+q['id']+' '+field)
                                replayed+=1
                            peak_rss=max(peak_rss,env['psutil'].Process().memory_info().rss)
                            if torch.cuda.max_memory_allocated()>72*2**30: raise MemoryError('72 GiB ceiling')
                        print(json.dumps(dict(stage='gpu',phase=phase,answers=len(seen),generated_tokens=generated,
                            peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30)),flush=True)
                    del low,high,candidates
                if phase=='replay':
                    if replayed!=102: raise ValueError('incomplete baseline replay')
                    save(directory/'replay_passed.json',dict(questions=102,status='passed'))
    finally:
        handle.remove(); del model; gc.collect(); torch.cuda.empty_cache()
    if seen!={(r['query']['id'],a,s) for r in records for a in ARMS for s in range(5)}:
        raise ValueError('incomplete experiment grid')
    verify(out)
    save(directory/'completed.json',dict(status='complete',records=len(seen),generated_tokens=generated,
        pretrained_loads=1,baseline_replayed=replayed,seconds=time.monotonic()-started,
        peak_GPU_allocated=torch.cuda.max_memory_allocated(),peak_RSS_observed=peak_rss,
        visual_forward_calls=calls[0],predictions_sha256=sha(directory/'predictions.jsonl'),
        CPU_proof_sha256=sha(out/'cpu/passed.json'),protocol_sha256=sha(out/'protocol.json')))


def score(out):
    from .scoring import build
    out=output_path(out); records=verify(out); done=read(out/'gpu/completed.json')
    if (done['status']!='complete' or done['predictions_sha256']!=sha(out/'gpu/predictions.jsonl')
            or done['CPU_proof_sha256']!=sha(out/'cpu/passed.json')
            or done['protocol_sha256']!=sha(out/'protocol.json')):
        raise ValueError('completed output mismatch')
    predictions=[json.loads(s) for s in (out/'gpu/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    summary,details=build([r['query'] for r in records],read(bundle_path()/'labels-offline-only.json'),predictions)
    folder=out/'score'; folder.mkdir(exist_ok=False)
    save(folder/'summary.json',summary); save(folder/'details.json',details)
