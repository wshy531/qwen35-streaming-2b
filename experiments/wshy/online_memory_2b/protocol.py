"""Standard-library preparation and identity locks. Labels stay in scoring files."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import secrets
import subprocess
from datetime import datetime, timezone

from .frozen import ACTIVE, ARMS, PACKAGE, REPO
from .isolation import output_path, write_root

REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
FIXED_PROTOCOL = dict(
    schema='online-memory-2b-v2', model_revision=REVISION, arms=list(ARMS),
    questions=171, sources=67, main_answers=684, maximum_answers=756,
    maximum_generated_tokens=6048, max_new_tokens=8, pretrained_loads=1,
    gpu_seconds=2550, slurm_minutes=45, cpus=4, memory_GiB=24,
    min_free_GPU_GiB=16, min_free_RAM_GiB=12, max_allocated_GPU_GiB=24, max_RSS_GiB=20,
    canvas_hw=[192, 256], threshold=0.047536224126815796, precision='float32',
    context_tokens=8192, visual_tokens_per_policy=384, opened_development_not_holdout=True,
    runtime='external visual bank; fresh language cache per question')

LONG_SHORT_ARMS = ('M0', 'M2', 'M2_A06', 'LS_shared384_v1', 'reservoir4', 'text_only')
LONG_SHORT_PROTOCOL = dict(FIXED_PROTOCOL, schema='online-memory-2b-long-short-v1',
    arms=list(LONG_SHORT_ARMS), main_answers=1026, maximum_answers=1146,
    maximum_generated_tokens=9168)
# These are separate suites so the original 28-test replay contract stays intact.
COMPARISON_SUITES = {'tests_a06': 16, 'tests_ls': 25}


def protocol_settings(schema):
    if schema == FIXED_PROTOCOL['schema']:
        return FIXED_PROTOCOL
    if schema == LONG_SHORT_PROTOCOL['schema']:
        return LONG_SHORT_PROTOCOL
    raise ValueError('unknown fixed protocol schema')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with output_path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def safe_path(root, relative):
    # Dataset catalogs are POSIX, independent of host filesystem conventions.
    if not isinstance(relative, str) or '\\' in relative or ':' in relative:
        raise ValueError('relative POSIX path required')
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or Path(relative).is_absolute():
        raise ValueError('path escapes root')
    return path


def validate_requests(requests):
    seen, groups = set(), {}
    for row in requests:
        if set(row) != {'id', 'cohort', 'media_path', 'media_sha256', 'model_visible'}:
            raise ValueError('request field allowlist violated')
        if row['id'] in seen:
            raise ValueError('duplicate request')
        seen.add(row['id'])
        visible = row['model_visible']
        if set(visible) != {'question', 'options', 'query_time_s', 'frames'}:
            raise ValueError('model-visible allowlist violated')
        if (not isinstance(visible['question'], str) or not visible['question'].strip()
                or not 2 <= len(visible['options']) <= 5
                or any(not isinstance(x, str) or not x.strip() for x in visible['options'])):
            raise ValueError('complete question and options required')
        query = visible['query_time_s']
        frames = visible['frames']
        if not isinstance(query, (int, float)) or not math.isfinite(query) or query < 0 or not frames:
            raise ValueError('finite query and sampled prefix required')
        last_id, last_pts = -1, -1.
        for frame in frames:
            if set(frame) != {'frame_id', 'pts_s', 'rgb_sha256'}:
                raise ValueError('frame field allowlist violated')
            if (type(frame['frame_id']) is not int or frame['frame_id'] <= last_id
                    or not math.isfinite(frame['pts_s']) or frame['pts_s'] <= last_pts
                    or frame['pts_s'] > query + 1e-9):
                raise ValueError('future or non-chronological frame')
            last_id, last_pts = frame['frame_id'], frame['pts_s']
        group = groups.setdefault(row['media_sha256'], [])
        if group and group[0]['model_visible']['frames'] != frames:
            raise ValueError('same source must use identical sampled prefix')
        if group and group[0]['model_visible']['query_time_s'] != query:
            raise ValueError('same source must use identical query horizon')
        group.append(row)
    if not seen:
        raise ValueError('nonempty requests required')
    return groups


def verify_package():
    for item in read(PACKAGE / 'PACKAGE_MANIFEST.json')['files']:
        path = safe_path(PACKAGE, item['path'])
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('frozen package identity changed: ' + item['path'])


def verify_model(model):
    path = ACTIVE / 'scripts/model_assets.py'
    spec = importlib.util.spec_from_file_location('_online2b_assets', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.verify_assets(model)
    if manifest['model'] != 'Qwen/Qwen3.5-2B' or manifest['revision'] != REVISION:
        raise ValueError('pinned 2B model required')
    old = read(PACKAGE / 'bundle/runs/2b-natural-eval-inputs-20261020-01/cpu_memory/preflight.json')['identity']
    identities = {
        REPO / 'qwen35_local/modeling_qwen3_5.py': 'original_model_sha256',
        ACTIVE / 'qwen35_local/modeling_qwen3_5.py': 'stream_model_sha256',
        ACTIVE / 'streaming/online.py': 'online_sha256',
        ACTIVE / 'streaming/hybrid.py': 'hybrid_sha256',
        Path(model) / 'DOWNLOAD_MANIFEST.json': 'manifest_sha256',
    }
    for file, field in identities.items():
        if sha(file) != old[field]:
            raise ValueError('historical model/runtime identity changed: ' + field)
    return manifest


def source_lock():
    paths = [REPO / 'scripts/run_2b_memory_eval.py', REPO / 'scripts/run_cpu_checks.py',
             REPO / 'scripts/audit_2b_long_stream.py']
    paths.extend(Path(__file__).parent.rglob('*.sh'))
    for folder in (Path(__file__).parent, REPO / 'experiments/wshy/long_stream_audit',
                   ACTIVE, REPO / 'tests',
                   REPO / 'experiments/36tar/streaming-learning/tools',
                   REPO / 'experiments/36tar/streaming-learning/code/qwen35-streaming'):
        paths.extend(folder.rglob('*.py'))
    paths.extend([REPO / 'requirements.txt', REPO / 'requirements-cluster.txt',
                  PACKAGE / 'PACKAGE_MANIFEST.json'])
    return {p.relative_to(REPO).as_posix(): sha(p) for p in sorted(set(paths))}


def checkpoint_grid(groups):
    return {h: sorted({n for n in (4, 8, (len(groups[h][0]['model_visible']['frames']) + 1)//2)
                       if n <= (len(groups[h][0]['model_visible']['frames']) + 1)//2})
            for h in sorted(groups)[:8]}


def deployment_identity():
    def git(*args):
        return subprocess.check_output(['git', '--no-optional-locks', '-c', 'core.fsmonitor=false',
                                        *args], cwd=REPO, text=True).strip()
    return dict(commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                worktree_status=git('status', '--short'), repo=str(REPO.resolve()))


def prepare(run, model, dataset, manifest=None, *, comparison=False):
    settings = LONG_SHORT_PROTOCOL if comparison else FIXED_PROTOCOL
    run = output_path(run)
    if run.exists():
        raise ValueError('new output directory required')
    verify_package()
    verify_model(model)
    canonical = PACKAGE / 'bundle/runs/2b-short-event-inputs-20261020-01/requests.json'
    requests = read(manifest or canonical)
    # This first runner intentionally replays the exact preregistered set.
    if requests != read(canonical):
        raise ValueError('this replay requires the canonical 171 request objects')
    groups = validate_requests(requests)
    if len(requests) != 171 or len(groups) != 67:
        raise ValueError('171 questions / 67 sources required')
    media = {r['sha256']: r for r in read(PACKAGE / 'catalog/media_manifest124.json')}
    for h in groups:
        entry = media[h]
        file = safe_path(dataset, entry['relative_path'])
        if file.stat().st_size != entry['bytes'] or sha(file) != h:
            raise ValueError('dataset identity mismatch: ' + entry['relative_path'])
    checkpoints = checkpoint_grid(groups)
    run.mkdir(parents=True, exist_ok=False)
    save(run / 'requests.json', requests)
    save(run / 'media.json', {h: media[h] for h in groups})
    save(run / 'scoring_ONLY.json', read(PACKAGE / 'catalog/evaluated171_catalog.json'))
    # Model predictions are regression references, never official labels.
    expected = [r for r in rows(PACKAGE / 'evidence/short-event-v1/result/main/predictions.jsonl')
                if r['arm'] in ARMS]
    save(run / 'expected_predictions.json', expected)
    save(run / 'source_lock.json', source_lock())
    save(run / 'protocol.json', dict(
        **settings, run_nonce=secrets.token_hex(16), deployment=deployment_identity(),
        write_root=str(write_root()),
        created_UTC=datetime.now(timezone.utc).isoformat(), run_path=str(run.resolve()),
        model=str(Path(model).resolve()),
        dataset=str(Path(dataset).resolve()),
        model_manifest_sha256=sha(Path(model) / 'DOWNLOAD_MANIFEST.json'),
        checkpoints=checkpoints, probe_answers=(len(settings['arms'])-1) * sum(map(len, checkpoints.values())),
        input_hashes={name: sha(run/name) for name in (
            'requests.json', 'media.json', 'scoring_ONLY.json', 'expected_predictions.json', 'source_lock.json')},
    ))


def verify_run(run):
    run = output_path(run)
    protocol = read(run / 'protocol.json')
    settings = protocol_settings(protocol.get('schema'))
    if any(protocol.get(key) != value for key, value in settings.items()):
        raise ValueError('fixed protocol fields changed')
    if protocol['write_root'] != str(write_root()):
        raise ValueError('prepared write boundary changed')
    if protocol['run_path'] != str(run.resolve()):
        raise ValueError('prepared run moved; prepare a new protocol')
    for name, expected in protocol['input_hashes'].items():
        if sha(safe_path(run, name)) != expected:
            raise ValueError('run input drift: ' + name)
    for relative, expected in read(run / 'source_lock.json').items():
        if sha(safe_path(REPO, relative)) != expected:
            raise ValueError('source drift: ' + relative)
    if source_lock() != read(run/'source_lock.json'):
        raise ValueError('source inventory drift')
    verify_package()
    if sha(Path(protocol['model']) / 'DOWNLOAD_MANIFEST.json') != protocol['model_manifest_sha256']:
        raise ValueError('model manifest changed')
    requests = read(run / 'requests.json')
    groups = validate_requests(requests)
    if (len(requests) != 171 or len(groups) != 67 or protocol['checkpoints'] != checkpoint_grid(groups)
            or protocol['probe_answers'] != (len(settings['arms'])-1) * sum(map(len, checkpoint_grid(groups).values()))):
        raise ValueError('fixed request/checkpoint denominator changed')
    return protocol, requests
