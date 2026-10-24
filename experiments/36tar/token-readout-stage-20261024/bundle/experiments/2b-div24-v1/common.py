"""Isolated projected-token comparison; existing model and dataset inputs are read-only."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/2b_div24_prepare_20261023_01'
BASE = ROOT / 'experiments/2b-natural-eval-v1'
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / 'experiments/2b-long-stream-v1'))
ARMS = ('full48', 'odd24', 'grid24', 'tome24', 'div24')


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
    repo = ROOT / 'repo' if (ROOT / 'repo').exists() else ROOT / 'publish/qwen35-streaming-2b'
    runtime_bridge.REPO = repo
    runtime_bridge.ACTIVE = repo / 'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    return runtime_bridge.runtime('memory')
