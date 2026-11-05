"""Frozen recent-state isolation experiment; importing does not load weights."""
import json,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[2]
P=R/'runs/isolated_recent_prepare_20261103_01'
for n in ('2b-resolution-cost-v1','2b-perception-transition-v1'):
    sys.path.insert(0,str(R/'experiments'/n))
from resolution_common import runtime,DATA
from resolution_runtime import make_builder,context
from resolution_session import Session
from resolution_source import CausalSource,PairDriver
from resolution_nodes import digest
from sember_protocol import prompt,parse_choice
from canonical_prompt import question_prompt,rotate
import protected_readout
ARMS=('recent','joint','isolated')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def split(bank):
    h=[e.node.copy() for e in bank.history];r=[e.node.copy() for e in bank.recent]
    assert digest(h+r)==digest(bank.export())
    assert len(r)<=4 and sum(len(n.features) for n in h+r)<=864
    return h,r
def compact(row):
    row=dict(row);ids=row.pop('input_ids');pos=row.pop('position_ids');row.pop('first_logits')
    row['bounded_context_ids_sha256']=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
    row['position_sha256']=hashlib.sha256(pos.cpu().numpy().tobytes()).hexdigest()
    return row
def verify():
    for name,h in read(P/'execution_lock.json').items():
        p=R/name
        if name.startswith('repo/') and not p.exists():p=R/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(p)==h,name

