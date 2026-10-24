"""Standalone, fixed-coverage cost comparison. No submission on import."""
import hashlib,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/2b_standalone_cost_prepare_20261023_01'
ARMS=('full48_8','odd24_8','change24_8')
for relative in ('2b-compressed-banks-v1','2b-natural-eval-v1','2b-long-stream-v1',
                 '2b-short-event-v1','2b-closed-balance-v1','2b-ls-depth-v1'):
    sys.path.append(str(ROOT/'experiments'/relative))


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False);stream.write('\n')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def runtime():
    import runtime_bridge
    repo=ROOT/'repo' if (ROOT/'repo').exists() else ROOT/'publish/qwen35-streaming-2b'
    runtime_bridge.REPO=repo
    runtime_bridge.ACTIVE=repo/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    return runtime_bridge.runtime('memory')
