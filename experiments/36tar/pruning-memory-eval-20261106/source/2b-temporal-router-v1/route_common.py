"""Owned paths and UTF-8 snapshots; no model import or credential access."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PREP=ROOT/'runs/temporal_router_prepare_20261027_01'
INPUTS=PREP/'inputs'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def question_key(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()
