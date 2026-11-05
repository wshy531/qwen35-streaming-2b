"""Formal orchestration only; reuse the unchanged, GPU-validated S-EMBER core."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
from sember_common import ARMS,DATA,PREP,read,save,sha,runtime
OUT=ROOT/'runs/sember_main_prepare_20261025_01'


def verify():
    lock=read(OUT/'inputs/execution_lock.json')
    for rel,digest in lock['runtime_sha256'].items():
        path=ROOT/rel if not rel.startswith('repo/') or (ROOT/'repo').exists() else ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(path)==digest,rel
    for name,digest in lock['input_sha256'].items():assert sha(OUT/'inputs'/name)==digest,name
    return read(OUT/'inputs/protocol.json')
