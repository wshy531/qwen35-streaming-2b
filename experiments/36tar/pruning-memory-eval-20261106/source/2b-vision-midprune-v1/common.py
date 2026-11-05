"""Independent cost-only candidate. No real weight loading on import."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/vision_midprune_prepare_20261025_01'
sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
from sember_common import runtime,read,sha


def save(path,value):
    def convert(v):
        if isinstance(v,(set,frozenset)):return sorted(v)
        raise TypeError(type(v).__name__)
    value=json.dumps(value,ensure_ascii=False,indent=2,default=convert)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as stream:stream.write(value)


def verify():
    lock=read(OUT/'inputs/execution_lock.json')
    for rel,digest in lock['runtime_sha256'].items():
        path=ROOT/rel
        if rel.startswith('repo/') and not (ROOT/'repo').exists():path=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(path)==digest,rel
    for name,digest in lock['input_sha256'].items():assert sha(OUT/'inputs'/name)==digest,name
    return read(OUT/'inputs/protocol.json')
