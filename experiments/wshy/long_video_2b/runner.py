"""Prepare a locked long-video run and CPU proof; no GPU launch implementation."""
import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from ..online_memory_2b.isolation import configure, output_path
configure()
from ..online_memory_2b.frozen import REPO, runtime
from ..online_memory_2b.protocol import (read, save, sha, safe_path, source_lock,
    verify_package, verify_model, deployment_identity, REVISION)
from ..online_memory_2b.receipts import inherited_result, successful_counts
from ..online_memory_2b.runner import isolated_readout, environment
from .protocol import ARMS, validate_queries
from .scheduler import validate_frames, decode_samples, replay_source
from .driver import run_sessions

FIXED = dict(schema='long-video-external-memory-2b-cpu-v1', model_revision=REVISION,
    arms=list(ARMS), sources=6, questions=34, source_sampled_frames=2853,
    last_query_horizon_frames=2726, source_pairs_to_encode=1360,
    planned_GPU_answers=204, planned_GPU_generated_tokens_max=1632,
    max_new_tokens=8, canvas_hw=[192,256], precision='float32',
    visual_tokens_per_policy=384, context_tokens=8192, question_suffix_max=128,
    runtime='bounded external vision; fresh language state each question',
    stop='after last query per source; no query or EOF padding',
    opened_development_not_holdout=True, independent_content_review='pending',
    CPU_pretrained_loads=0, CPU_tiny_queries=2, CPU_tiny_answers=12)
PINNED = dict(requests='f750bebd7ead50d7d75c963e02ab7392cc43d1632bdfb4aa624e09885dfa7f77',
              sources='b0db31454c2b48817d86f88c5796bb76f612ece4a7786f24456eb025617c7f78',
              snapshots='67494d1fb9bb347daf49a19eb4ee58d1c17f4f3541a4f11863475611bafb16e9')
SUITES = {'online_memory_2b/tests':28, 'online_memory_2b/tests_a06':16,
          'online_memory_2b/tests_ls':25, 'long_stream_audit/tests':8, 'long_video_2b/tests':26}


def validate_inputs(folder):
    folder = output_path(folder)
    for name, expected in PINNED.items():
        if sha(folder/(name+'.json')) != expected:
            raise ValueError('canonical long-video input drift: '+name)
    queries, sources, snapshots = (read(folder/(name+'.json')) for name in PINNED)
    groups = validate_queries(queries, canonical=True)
    if (len(sources) != 6 or len({s['source_path'] for s in sources}) != 6
            or {s['source_path'] for s in sources} != set(groups)
            or len(snapshots) != 34 or len({s['query_id'] for s in snapshots}) != 34
            or {s['query_id'] for s in snapshots} != {q['id'] for q in queries}):
        raise ValueError('source or snapshot denominator')
    frames = pairs = total = 0
    for source in sources:
        if set(source) != {'source_path','media_path','media_sha256','duration_s','frames'}:
            raise ValueError('source field allowlist')
        specs = validate_frames(source['frames'])
        horizon = max(q['query_time_s'] for q in groups[source['source_path']])
        if horizon > source['duration_s']:
            raise ValueError('query past declared EOF')
        count = bisect_right([f['pts_s'] for f in specs], horizon)
        frames += count
        pairs += count//2
        total += len(specs)
    if (frames, pairs, total) != (2726,1360,2853):
        raise ValueError('locked long-video budget changed')
    return queries, sources, snapshots


def runtime_lock():
    result = source_lock()
    folder = Path(__file__).parent
    paths = list(folder.rglob('*.py'))+list(folder.rglob('*.sh'))
    paths.append(REPO/'scripts/run_2b_long_video_eval.py')
    result.update({p.relative_to(REPO).as_posix():sha(p) for p in paths})
    return result


def prepare(run, inputs, resource_root, model):
    run, inputs = output_path(run), output_path(inputs)
    if run.exists():
        raise ValueError('new run directory required')
    validate_inputs(inputs)
    verify_package()
    verify_model(model)
    # Media are read-only external resources, resolved against an explicit root.
    for source in read(inputs/'sources.json'):
        if sha(safe_path(resource_root, source['media_path'])) != source['media_sha256']:
            raise ValueError('media file identity drift')
    run.mkdir(parents=True, exist_ok=False)
    for name in ('requests.json','sources.json','snapshots.json','scoring_ONLY.json','receipt.json'):
        with (run/name).open('xb') as stream:
            stream.write((inputs/name).read_bytes())
    save(run/'source_lock.json', runtime_lock())
    save(run/'protocol.json', dict(**FIXED, run_path=str(run), write_root=str(configure()),
        model=str(Path(model).resolve()), resource_root=str(Path(resource_root).resolve()),
        model_manifest_sha256=sha(Path(model)/'DOWNLOAD_MANIFEST.json'),
        deployment=deployment_identity(), created_UTC=datetime.now(timezone.utc).isoformat(),
        input_hashes={name:sha(run/name) for name in ('requests.json','sources.json','snapshots.json',
            'scoring_ONLY.json','receipt.json','source_lock.json')}))


def verify(run):
    run = output_path(run)
    protocol = read(run/'protocol.json')
    if any(protocol.get(k) != v for k,v in FIXED.items()) or protocol['write_root'] != str(configure()):
        raise ValueError('fixed protocol drift')
    if Path(protocol['run_path']) != run:
        raise ValueError('prepared run path changed')
    for name, expected in protocol['input_hashes'].items():
        if sha(run/name) != expected:
            raise ValueError('prepared input changed: '+name)
    lock = read(run/'source_lock.json')
    if lock != runtime_lock():
        raise ValueError('executed source identity changed')
    return protocol, validate_inputs(run)


def cpu(run):
    protocol, (queries, sources, snapshots) = verify(run)
    if (not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.environ.get('SLURM_CPUS_PER_TASK') != '4'
            or int(os.environ.get('SLURM_RESTART_COUNT','0')) != 0):
        raise ValueError('approved CPU allocation, hidden GPU and no restart required')
    verify_model(Path(protocol['model']))
    out = output_path(run/'cpu')
    out.mkdir(exist_ok=False)
    def execute(command, name):
        with (out/name).open('x',encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
        if result.returncode:
            raise RuntimeError('CPU check failed: '+name)
        return (out/name).read_text(encoding='utf-8')
    inherited = inherited_result(execute([sys.executable,'-B',str(REPO/'scripts/run_cpu_checks.py'),
        '--model-dir',protocol['model']], 'inherited.txt'))
    logs = {}
    for folder, count in SUITES.items():
        name = folder.replace('/','_')+'.txt'
        text = execute([sys.executable,'-B','-m','unittest','discover','-s',
            str(REPO/'experiments/wshy'/folder),'-v'], name)
        successful_counts(text,[count])
        logs[folder] = dict(tests=count, sha256=sha(out/name), file=name)
    env = runtime()
    torch = env['torch']
    torch.set_num_threads(1)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError('GPU must stay hidden for CPU validation')
    snap_byid = {s['query_id']:s for s in snapshots}
    groups = validate_queries(queries, canonical=True)
    observed, counters = [], []
    for source in sources:
        path = safe_path(protocol['resource_root'],source['media_path'])
        if sha(path) != source['media_sha256']:
            raise ValueError('media changed before decode')
        def check(query, state):
            snap = snap_byid[query['id']]
            fields = dict(received_frames=state['arrived_frames'],committed_pairs=state['committed_pairs'],
                committed_through_s=state['committed_through_s'],observation_age_s=state['observation_age_s'],
                pending_frame_id=None if state['pending_frame'] is None else state['pending_frame']['frame_id'],
                committed_frames_sha256=state['historical_committed_frames_sha256'])
            if any(snap[k] != v for k,v in fields.items()):
                raise ValueError('actual query snapshot drift: '+query['id'])
            observed.append(dict(id=query['id'], **fields))
        result = replay_source(source['frames'],groups[source['source_path']],
            decode_samples(env,path,source['frames']),lambda *args:None,check)
        if sha(path) != source['media_sha256']:
            raise ValueError('media changed during decode')
        counters.append(result)
    if (sum(c['sampled_frames'] for c in counters),sum(c['encoded_pairs'] for c in counters),len(observed)) != (2726,1360,34):
        raise ValueError('real RGB/PTS coverage denominator')
    processor = env['AutoProcessor'].from_pretrained(protocol['model'],local_files_only=True)
    builder = env['PairInputBuilder'](processor,canvas=(192,256),resize_mode='letterbox')
    maximum = max(builder.question_ids(env['prompt'](q)).shape[1] for q in queries)
    if maximum > 128:
        raise ValueError('full question/options exceed original suffix budget')
    torch.manual_seed(123)
    model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](
        **env['tiny_config'](processor),attn_implementation='sdpa')).eval()
    tiny_frames = [dict(frame_id=i,pts_s=float(i),rgb_sha256='0'*64) for i in range(24)]
    tiny_queries = [dict(id='tiny:'+str(i),source_path='synthetic.mp4',query_time_s=t,
        question='Return only A.',options=['first','second'],cohort='synthetic') for i,t in enumerate((17,23))]
    def tiny_samples():
        for f in tiny_frames:
            yield SimpleNamespace(frame_id=f['frame_id'],timestamp=f['pts_s'])
    def encode(index, first, second):
        tensor = torch.randn(48,32,generator=torch.Generator().manual_seed(800+index))
        return env['legacy'].features.raw_block(tensor,index,first.timestamp,second.timestamp)
    answers = []
    with torch.inference_mode(),env['fp32_reference']():
        tiny = run_sessions(tiny_frames,tiny_queries,tiny_samples(),encode,
            lambda ns,q:isolated_readout(env,model,builder,ns,q),env['prompt'],lambda e:None,
            lambda r:answers.append(dict(id=r['id'],arm=r['arm'],generated=len(r['generated_ids']))))
    if tiny['answers'] != 12 or len(answers) != 12:
        raise RuntimeError('twelve tiny model readouts required')
    save(out/'actual_query_checks.json',observed)
    save(out/'preflight.json',dict(status='passed',pretrained_loads=0,
        protocol_sha256=sha(run/'protocol.json'),source_lock_sha256=sha(run/'source_lock.json'),
        environment=environment(env),inherited=inherited,inherited_log_sha256=sha(out/'inherited.txt'),
        suites=logs, sources_RGB_PTS_checked=6,sampled_frames=2726,pair_callbacks=1360,
        actual_visual_model_forwards=0,queries_checked=34,max_suffix_tokens=maximum,
        actual_query_checks_sha256=sha(out/'actual_query_checks.json'),tiny=tiny,tiny_answers=answers,
        independent_video_content_review='pending',long_video_accuracy_evaluated=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('prepare','preflight'))
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--resource-root',type=Path)
    parser.add_argument('--model-dir',type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if not all((args.inputs,args.resource_root,args.model_dir)):
            parser.error('prepare requires inputs, resource-root and model-dir')
        prepare(args.out,args.inputs,args.resource_root,args.model_dir)
    else:
        cpu(args.out)


if __name__=='__main__':
    main()
