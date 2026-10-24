"""Independent, fixed-budget LS x depth comparison. No submission on import."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/2b_readout_compression_prepare_20261024_01'
POLICIES=('M2','A06','LS')
ARMS=tuple(p+'_'+mode for p in POLICIES for mode in ('full48','read24'))
for relative in ('2b-natural-eval-v1', '2b-long-stream-v1',
                 '2b-short-event-v1', '2b-closed-balance-v1'):
    sys.path.append(str(ROOT / 'experiments' / relative))


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

sys.path.append(str(ROOT/'experiments/2b-ls-depth-v1'))
sys.path.append(str(ROOT/'experiments/2b-compressed-banks-v1'))

CONFIGS={'slice1':dict(thread_count=1,thread_type='SLICE'),'auto4':dict(thread_count=4,thread_type='AUTO')}
