"""Frozen five-arm quality and independent cost protocol, no resources on import."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/content_main_prepare_20261101_01'
for name in ['2b-content-quality-v1','2b-midprune-quality-v1']:
    sys.path.insert(0,str(ROOT/'experiments'/name))
from session import Session,MODES
from sember_common import runtime,DATA
from resolution_runtime import make_builder
from resolution_source import CausalSource,PairDriver
from sember_protocol import prompt
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def verify():
    lock=read(PREP/'execution_lock.json')
    for rel,h in lock.items():
        p=ROOT/rel
        if rel.startswith('repo/') and not p.exists():p=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        if sha(p)!=h:raise ValueError('changed source/input: '+rel)
    return lock
def pack(q,prefix,result):
    return dict(result,question_id=q['question_id'],video_id=q['video_id'],query_prefix=prefix)
