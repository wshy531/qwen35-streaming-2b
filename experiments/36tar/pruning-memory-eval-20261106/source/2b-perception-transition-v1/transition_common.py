"""Owned preparation paths; the completed first diagnostic stays unchanged."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREP = ROOT / 'runs/perception_transition_prepare_20261027_01'
INPUTS = PREP / 'inputs'
SOURCE = ROOT / 'runs/count_action_prepare_20261026_01'
REFERENCE = ROOT / 'runs/memory_collab_prepare_20261026_01'
ARMS = ('recent', 'full', 'history_value_swap')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()
