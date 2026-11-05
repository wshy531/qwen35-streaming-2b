"""Owned paths and append-only receipts; importing does not load a model."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREP = ROOT / 'runs/recent_evidence_prepare_20261028_01'
INPUTS = PREP / 'inputs'
DEFAULT_MEDIA = ROOT / 'datasets/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos'
PARTNER_MEDIA = Path('/home/u2025010462/share/czy/wsy/data/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos')
MEDIA = Path(os.environ.get('RECENT_EVIDENCE_MEDIA_ROOT', str(DEFAULT_MEDIA)))
if MEDIA.resolve() not in (DEFAULT_MEDIA.resolve(), PARTNER_MEDIA.resolve()):
    raise ValueError('only own media or explicitly authorized read-only partner snapshot allowed')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def shared():
    import sys
    sys.path.insert(0, str(ROOT / 'experiments/2b-sember-v1'))
    from sember_common import runtime
    return runtime()
