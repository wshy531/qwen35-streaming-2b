"""Freeze exact owned runtime and review policy; never submits a GPU job."""
import argparse
from importlib.metadata import version
from pathlib import Path
import sys
from common import ROOT, HERE, PREP, INPUTS, read, save, sha, dependencies, require_development_review


def runtime_path(name):
    path = ROOT / name
    if name.startswith('repo/') and not (ROOT / 'repo').exists():
        path = ROOT / 'publish/qwen35-streaming-2b' / name[5:]
    return path


def closure():
    sys.path.insert(0, str(ROOT / 'experiments/2b-sember-v1'))
    from sember_common import runtime
    env = runtime()
    base = env['runtime_bridge'].REPO.resolve() if 'runtime_bridge' in env else (
        ROOT / 'repo' if (ROOT / 'repo').exists() else ROOT / 'publish/qwen35-streaming-2b').resolve()
    active = base / 'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    paths = {p.resolve() for p in active.rglob('*.py')}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and (path.is_relative_to((ROOT / 'experiments').resolve()) or path.is_relative_to(base)):
                paths.add(path)
    paths.update((ROOT / p).resolve() for p in dependencies())
    result = {}
    for path in paths:
        name = ('repo/' + path.relative_to(base).as_posix()) if path.is_relative_to(base) else path.relative_to(ROOT.resolve()).as_posix()
        result[name] = sha(path)
    return dict(sorted(result.items()))


def verify_lock(path):
    lock = read(path)
    if lock['schema'] != 'perception-interference-execution-v1':
        raise ValueError('fixed execution lock required')
    for name, digest in lock['runtime_sha256'].items():
        if sha(runtime_path(name)) != digest:
            raise ValueError('runtime changed: ' + name)
    for name, digest in lock['evidence_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('input/review/proof changed: ' + name)
    for name, expected in lock['package_versions'].items():
        if version(name) != expected:
            raise ValueError('execution environment changed: ' + name)
    for name, digest in lock['model_metadata_sha256'].items():
        if sha(ROOT / 'models/Qwen3.5-2B' / name) != digest:
            raise ValueError('model/processor metadata changed: ' + name)
    return lock


def freeze(output, cpu, smoke):
    output, cpu, smoke = (Path(p).resolve() for p in (output, cpu, smoke))
    if not output.is_relative_to((ROOT / 'runs').resolve()):
        raise ValueError('owned project output directory required')
    proof = read(cpu)
    if not proof['status'].startswith('passed') or proof['planned_layouts'] != 90:
        raise ValueError('completed CPU layout proof required')
    if proof['runtime_sha256'] != dependencies():
        raise ValueError('CPU-tested source list changed')
    smoke_proof = read(smoke)
    if smoke_proof['status'] != 'passed_CPU_shared_execution':
        raise ValueError('shared decoder/encoder/QA CPU check required')
    for name, digest in smoke_proof['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('smoke-tested source changed')
    review = read(PREP / 'assistant_review_LOCAL_ONLY.json')
    require_development_review(read(INPUTS / 'manifest_LOCAL_ONLY.json'), review)
    evidence = [p for p in INPUTS.iterdir() if p.is_file()]
    evidence += [PREP / name for name in ('assistant_review_LOCAL_ONLY.json', 'review_scoring_policy_LOCAL_ONLY.json',
        'cluster_media_CPU.json', 'old_reference_predictions_LOCAL_ONLY.json', 'assistant_review_CPU_check.json')]
    evidence += [cpu, smoke]
    for task in review['tasks']:
        evidence.extend(ROOT / item['relative_path'] for item in task['evidence_images'])
    for donor in review['donor_reviews']:
        evidence.extend(ROOT / item['relative_path'] for item in donor['evidence_images'])
    result = dict(schema='perception-interference-execution-v1', runtime_sha256=closure(),
        evidence_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(evidence))},
        package_versions={name: version(name) for name in ('torch', 'transformers', 'numpy', 'av', 'Pillow', 'safetensors', 'psutil')},
        model_metadata_sha256={p.name: sha(p) for p in (ROOT / 'models/Qwen3.5-2B').iterdir()
            if p.suffix in ('.json', '.jinja') and p.stat().st_size < 40 * 2**20},
        budget=dict(GPUs=1, CPUs=4, RAM_GiB=24, minutes=20, formal_answers=90, warmup_answers=3,
            maximum_generated_tokens=744, visual_calls=367, pretrained_loads=1, training_steps=0,
            downloads=0, automatic_retry=False, requeue=False), GPU_authorization_not_created=True,
        review_scope=review['scope'])
    save(output, result)
    verify_lock(output)
    print(dict(status='execution_frozen_not_authorized', runtime_files=len(result['runtime_sha256']),
               evidence_files=len(result['evidence_sha256']), GPU_jobs=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    args = parser.parse_args()
    freeze(args.output, args.cpu, args.smoke)
