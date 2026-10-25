"""One fixed resolution/cost experiment; no real model load on import."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
from sember_common import read,save,sha,runtime,DATA,THRESHOLD
OUT=ROOT/'runs/resolution_cost_prepare_20261025_01'
MAIN=ROOT/'runs/sember_main_prepare_20261025_01'
POLICIES=('M0','M2','A06','LS')
SPECS={}
for policy in POLICIES:
    for view,canvas,keep in [('mid108',(288,384),108),('high192',(384,512),192),('highchange108',(384,512),108)]:
        SPECS[policy+'_'+view]=dict(policy=policy,view=view,canvas=canvas,keep=keep,
            budget=8*keep,recent=4*keep,closed=3*keep)
SPECS['M0_highuniform108']=dict(policy='M0',view='highuniform108',canvas=(384,512),keep=108,
    budget=864,recent=432,closed=324)
ARMS=tuple(SPECS)


def arms_for(canvas):return tuple(a for a,s in SPECS.items() if s['canvas']==tuple(canvas))


def verify():
    lock=read(OUT/'inputs/execution_lock.json')
    for rel,digest in lock['runtime_sha256'].items():
        path=ROOT/rel
        if rel.startswith('repo/') and not (ROOT/'repo').exists():path=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(path)==digest,rel
    for name,digest in lock['input_sha256'].items():assert sha(OUT/'inputs'/name)==digest,name
    return read(OUT/'inputs/protocol.json')
