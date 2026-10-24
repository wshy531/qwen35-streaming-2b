"""Fixed V1 workload and explicit dependencies. No submission side effects."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/2b_vision_cache_prepare_20261024_01'
ARMS = ('full48_8', 'cache25_8')
CONFIGS = {'slice1':dict(thread_count=1, thread_type='SLICE'),
           'auto4':dict(thread_count=4, thread_type='AUTO')}
for name in ('2b-compressed-banks-v1','2b-natural-eval-v1','2b-long-stream-v1',
             '2b-short-event-v1','2b-closed-balance-v1','2b-ls-depth-v1'):
    sys.path.append(str(ROOT/'experiments'/name))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def runtime():
    import runtime_bridge
    runtime_bridge.REPO = ROOT/'repo' if (ROOT/'repo').exists() else ROOT/'publish/qwen35-streaming-2b'
    runtime_bridge.ACTIVE = runtime_bridge.REPO/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    return runtime_bridge.runtime('memory')
