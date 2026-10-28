"""Isolated QProbe research. Imports never submit jobs or load real weights."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'runs/2b_question_probe_prepare_20261024_01'
ARMS = ('LS_full48', 'LS_read24', 'LS_probe24')
CONFIGS = {'slice1': dict(thread_count=1, thread_type='SLICE'),
           'auto4': dict(thread_count=4, thread_type='AUTO')}
for name in ('2b-natural-eval-v1', '2b-long-stream-v1', '2b-short-event-v1',
             '2b-closed-balance-v1', '2b-ls-depth-v1', '2b-compressed-banks-v1',
             '2b-readout-compression-v1'):
    sys.path.append(str(ROOT/'experiments'/name))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def runtime():
    import runtime_bridge
    repo = ROOT/'repo' if (ROOT/'repo').exists() else ROOT/'publish/qwen35-streaming-2b'
    runtime_bridge.REPO = repo
    runtime_bridge.ACTIVE = repo/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    return runtime_bridge.runtime('memory')
