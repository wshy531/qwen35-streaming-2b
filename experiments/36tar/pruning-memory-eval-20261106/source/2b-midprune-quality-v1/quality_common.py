"""Fixed follow-up to R1. Importing never loads real weights or submits work."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/midprune_quality_prepare_20261026_01'
MAIN=ROOT/'runs/sember_main_prepare_20261025_01'
RESOLUTION=ROOT/'runs/resolution_cost_prepare_20261025_01'
for rel in ('experiments/2b-sember-v1','experiments/2b-resolution-cost-v1','experiments/2b-vision-midprune-v1'):
    sys.path.insert(0,str(ROOT/rel))
from sember_common import runtime,read,sha,DATA

ARMS=('full108_8','post_odd54_8','post_change54_8','mid_odd54_8')
OLD_ARMS=dict(zip(ARMS[:3],('M0_full108_8','M0_odd54_8','M0_change54_8')))
KEEP={a:108 if a==ARMS[0] else 54 for a in ARMS}


def lines(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s]


def golden_path():
    local=MAIN/'GPU_results_218526/bounded/answers.jsonl'
    return local if local.exists() else ROOT/'runs/2b-sember-main-218526/bounded/answers.jsonl'


def save(path,value):
    def convert(v):
        if isinstance(v,(set,frozenset)):return sorted(v)
        raise TypeError(type(v).__name__)
    value=json.dumps(value,ensure_ascii=False,indent=2,default=convert)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(value)


def verify():
    lock=read(OUT/'inputs/execution_lock.json')
    for rel,digest in lock['runtime_sha256'].items():
        p=ROOT/rel
        if rel.startswith('repo/') and not (ROOT/'repo').exists():p=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(p)==digest,rel
    for name,digest in lock['input_sha256'].items():assert sha(OUT/'inputs'/name)==digest,name
    return read(OUT/'inputs/protocol.json')
