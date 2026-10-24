"""One locked matched-coverage comparison. Never submits, downloads or retries a job."""
import argparse
from collections import Counter
import gc
import hashlib
import os
from pathlib import Path
import time
import json
from common import ROOT, ARMS, read, save, sha, runtime


def verify(run):
    assert run.resolve().is_relative_to((ROOT/'runs').resolve())
    protocol = read(run/'protocol.json'); lock = read(run/'execution_lock.json')
    assert lock['protocol_sha256'] == sha(run/'protocol.json')
    for name, digest in lock['input_sha256'].items():
        assert sha(run/name) == digest, name
    for name, digest in lock['runtime_sha256'].items():
        path = ROOT/name
        if name.startswith('repo/') and not (ROOT/'repo').exists():
            path = ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(path) == digest, name
    rows, groups = read(run/'questions.json'), read(run/'groups.json')
    assert len(rows) == len({q['id'] for q in rows}) == 303 and len(groups) == 112
    assert protocol['arms'] == list(ARMS)
    return protocol, rows, groups


def packet_stream(env, path, frames):
    wanted = {f['frame_id']:f for f in frames}; last = max(wanted); origin = None; received = 0
    with env['av'].open(str(path)) as container:
        stream = container.streams.video[0]; stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if origin is None: origin = float(frame.time)
            if index in wanted:
                expected = wanted[index]; pts = float(frame.time)-origin; rgb = env['canvas_bytes'](frame)
                assert abs(pts-expected['pts_s']) < 1e-7
                assert hashlib.sha256(rgb).hexdigest() == expected['rgb_sha256']
                received += 1
                yield env['FramePacket'](index,pts,env['np'].frombuffer(rgb,dtype='uint8').reshape(192,256,3).copy(),index+1)
            if index >= last: break
    assert received == len(frames)


def worker(run, out):
    protocol, rows, groups = verify(run)
    auth = read(run/'GPU_authorization.json')
    assert auth['approved'] and auth['execution_lock_sha256'] == sha(run/'execution_lock.json')
    cpu = read(run.parent/'CPU_preflight.json')
    assert cpu['status'] == 'passed' and cpu['protocol_sha256'] == sha(run/'protocol.json')
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES') not in (None,'','-1')
    env = runtime(); torch = env['torch']; torch.set_num_threads(1); torch.manual_seed(123)
    assert torch.cuda.mem_get_info()[0] >= 24*2**30 and env['psutil'].virtual_memory().available >= 12*2**30
    from session import Session
    from replay import VideoSource, replay
    from score_memory_eval import prompt
    out.mkdir(exist_ok=False)
    started = time.monotonic(); count = generated = encodes = peak = 0; cohorts = Counter()
    model = env['Qwen3_5ForConditionalGeneration'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True, dtype=torch.float32, device_map='cuda', attn_implementation='sdpa').eval()
    assert model.config.text_config.hidden_size == 2048
    save(out/'loaded.json',dict(pretrained_loads=1,seconds=time.monotonic()-started))
    builder = env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True), canvas=(192,256), resize_mode='letterbox')
    with (out/'answers.jsonl').open('x',encoding='utf-8') as answers, \
         (out/'events.jsonl').open('x',encoding='utf-8') as events, \
         (out/'tokens.jsonl').open('x',encoding='utf-8') as tokens, torch.inference_mode(), env['fp32_reference']():
        def guard():
            nonlocal peak
            peak = max(peak,env['psutil'].Process().memory_info().rss)
            assert peak < 20*2**30 and torch.cuda.memory_allocated() < 32*2**30
            assert time.monotonic()-started < 2400
            assert sum(p.stat().st_size for p in out.iterdir() if p.is_file()) < 64*2**20
        def record(q,result):
            nonlocal count, generated
            assert set(result['arms']) == set(ARMS)
            for arm, answer in result['arms'].items():
                assert arm == answer['arm']
                answers.write(json.dumps(dict(id=q['id'],group=q['group'],cohort=q['cohort'],status='ok',**answer))+'\n')
                count += 1; generated += len(answer['generated_ids']); cohorts[q['cohort']] += 1
            answers.flush(); tokens.flush(); guard()
        for number, group in enumerate(groups):
            path = ROOT/group['cluster_media']
            assert path.resolve().is_relative_to(ROOT.resolve()) and sha(path) == group['sha256']
            qs = sorted([q for q in rows if q['group'] == group['id']],key=lambda q:(q['query_time_s'],q['id']))
            session = Session(env,model,builder)
            try:
                if group['cohort'] == 'stream34':
                    spec = dict(source_path=group['source_path'],frames=group['frames'],
                        media_sha256=group['sha256'],duration_s=group['duration_s'])
                    source = VideoSource(path,spec,env['FramePacket'])
                    def on_event(item):
                        if 'state' in item and 'selection_trace' in item['state']:
                            trace=item['state']['selection_trace']
                            item['state']=dict(pairs=item['state']['pairs'],last_pts=item['state']['last_pts'],
                                selection_trace={k:trace[k] for k in ('pair_index','begin_s','end_s','cold_start','selected_indices','temporal_scores','starts_event','adjacent_distance')})
                        if item['event']['kind'] == 'query':
                            q = next(q for q in qs if q['id'] == item['event']['query_id'])
                            record(q,item['answer'])
                        events.write(json.dumps(item)+'\n'); events.flush(); guard()
                    replay(source,qs,session,read(run/'snapshots34.json'),on_event,
                        lambda t:tokens.write(json.dumps(t)+'\n'),read_from='matched_windows')
                else:
                    pending = None
                    for packet in packet_stream(env,path,group['frames']):
                        if pending is None: pending = packet
                        else:
                            state=session.append_pair(pending,packet)
                            trace=state['selection_trace']
                            events.write(json.dumps(dict(kind='selection',group=group['id'],
                                trace={k:trace[k] for k in ('pair_index','begin_s','end_s','cold_start','selected_indices','temporal_scores','starts_event','adjacent_distance')}))+'\n')
                            pending = None; guard()
                    if pending is not None:
                        state=session.snapshot_tail(pending);trace=state['selection_trace']
                        events.write(json.dumps(dict(kind='query_only_selection',group=group['id'],
                            trace={k:trace[k] for k in ('pair_index','begin_s','end_s','cold_start','selected_indices','temporal_scores','starts_event','adjacent_distance')}))+'\n')
                    for q in qs:
                        assert session.pairs == q['pairs']
                        record(q,session.ask(prompt(q),read_from='matched_windows',
                            on_token=lambda t:tokens.write(json.dumps(dict(query_id=q['id'],**t))+'\n')))
                encodes += session.vision_encodings
                events.write(json.dumps(dict(kind='group_end',group=group['id'],state=session.status(),
                    update_seconds=session.bank.update_seconds,encoding_timing=session.timing))+'\n'); events.flush()
            finally: session.close()
            assert not session.bank.banks and session.bank.previous is None
            del session; gc.collect(); torch.cuda.empty_cache(); guard()
            print(dict(groups_complete=number+1,answers=count),flush=True)
    assert count == 1515 and generated <= 12120 and encodes == 3981
    save(out/'completed.json',dict(status='complete',answers=count,generated_tokens=generated,
        cohorts=dict(cohorts),vision_encodings=encodes,pretrained_loads=1,seconds=time.monotonic()-started,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),observed_RSS_bytes=peak,
        bank_visual_cap=384,no_persistent_main_language_state=True,no_full_history_archive=True,
        wallclock_1x=False,shared_process_timings_not_isolated_speed=True,
        execution_lock_sha256=sha(run/'execution_lock.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(); worker(args.run,args.out)
