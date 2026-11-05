import sys, json, hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/2b-isolated-recent-v1'))
from experiment import runtime,make_builder,context,Session,split,compact,protected_readout,ARMS,read,save,sha,parse_choice
sys.path.insert(0,str(R/'experiments/streamingbench-router-v1'))
from v0_source import V0Pairs
P=R/'runs/protection_streamingbench_prepare_20261103_01'

def prompt(q):
    return q['question']+'\n'+'\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(q['options']))+'\nAnswer with the option letter only.'

def verify():
    for n,h in read(P/'execution_lock.json').items():
        p=R/n
        if n.startswith('repo/') and not p.exists():p=R/'publish/qwen35-streaming-2b'/n[5:]
        assert sha(p)==h,n

def pair_digest(d,a,b,padding):
    d.update(json.dumps([a.frame_id,a.timestamp,b.frame_id,b.timestamp,bool(padding)],separators=(',',':')).encode())
    d.update(a.rgb.tobytes());d.update(b.rgb.tobytes())
