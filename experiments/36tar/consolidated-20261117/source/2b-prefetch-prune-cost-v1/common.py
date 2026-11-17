"""One bounded composition experiment. No model or scheduler is started on import."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT/'runs/prefetch_prune_prepare_20261109_01'
sys.path.insert(0, str(ROOT/'experiments/2b-stream-prune-cost-v1'))
sys.path.insert(0, str(ROOT/'experiments/2b-bounded-prefetch-v1'))
import worker
from prefetch import PrefetchPairs
from v0_source import V0Pairs

CASES = {
    'serial_full': ('serial', 'full'),
    'prefetch_full': ('prefetch', 'full'),
    'prefetch_fixed': ('prefetch', 'fixed12'),
    'prefetch_content': ('prefetch', 'heuristic11'),
}
MATCH = ('generated_ids', 'input_ids_sha256', 'position_ids_sha256',
         'visual_tokens', 'visible_pairs', 'source_state')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def resolve(relative):
    path = ROOT/relative
    if relative.startswith('repo/') and not path.exists():
        path = ROOT/'publish/qwen35-streaming-2b'/relative[5:]
    elif relative.startswith('publish/qwen35-streaming-2b/') and not path.exists():
        path = ROOT/'repo'/relative[len('publish/qwen35-streaming-2b/'):]
    return path

def verify(media=False):
    for relative, digest in read(PREP/'execution_lock.json').items():
        assert sha(resolve(relative)) == digest, relative
    if media:
        for q in read(PREP/'panel.json'):
            assert sha(ROOT/q['media_path']) == q['media_sha256'], q['id']

def run_case(env, model, builder, question, case, guard):
    schedule, arm = CASES[case]
    previous = worker.V0Pairs
    worker.V0Pairs = V0Pairs if schedule == 'serial' else PrefetchPairs
    try:
        result = worker.run_source(env, model, builder, question, arm, guard)
    finally:
        worker.V0Pairs = previous
    result['wall_not_in_callback_seconds'] = result.pop('decode_sampling_driver_seconds')
    result.update(case=case, schedule=schedule,
                  residual_scope='Overlap residual, NOT independent decoder service time')
    return result
