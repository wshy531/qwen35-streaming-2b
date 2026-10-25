"""Fixed S-EMBER protocol. Importing never loads weights or submits a job."""
from pathlib import Path
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/sember_adapter_prepare_20261025_01'
PREP=ROOT/'runs/sember_prepare_20261025_01'
REV='897886cc967b2d0e6e45f1200605f5d3b0ffbb23'
DEFAULT_DATA=ROOT/'datasets/S-EMBER'/REV
PARTNER_DATA=Path('/home/u2025010462/share/czy/wsy/data/S-EMBER')/REV
DATA=Path(os.environ.get('SEMBER_MEDIA_ROOT',str(DEFAULT_DATA)))
if DATA.resolve() not in (DEFAULT_DATA.resolve(),PARTNER_DATA.resolve()):
    raise ValueError('media root must be project-local or the exact user-authorized read-only partner snapshot')
ARMS=('M0_full108_8','M0_odd54_16','M0_change54_16','M2_full108','M2_change54',
      'A06_full108','A06_change54','LS_full108','LS_change54','M0_odd54_8','M0_change54_8')
THRESHOLD=0.047536224126815796


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def runtime(stage='memory'):
    # Select project paths through the existing bridge, without altering model kernels.
    import sys
    sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
    import runtime_bridge
    repo=ROOT/'repo' if (ROOT/'repo').exists() else ROOT/'publish/qwen35-streaming-2b'
    runtime_bridge.REPO=repo
    runtime_bridge.ACTIVE=repo/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    return runtime_bridge.runtime(stage)
