"""Frozen selector deployment helpers, without labels or evidence intervals."""
from pathlib import Path
import sys,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/selector_qa_prepare_20261030_01'
TRAIN=ROOT/'runs/evidence_selector_train_prepare_20261029_01'
HEADS=ROOT/'runs/selector-fit-219523'
ARMS=['recent','full','lexical','matched_latest','content_visible','history_blind']

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)

def media_path(source):
    if 'authorized_shared_readonly_path' in source:
        allowed=Path('/home/u2025010462/share/czy/wsy/data/S-EMBER/897886cc967b2d0e6e45f1200605f5d3b0ffbb23/videos').resolve()
        p=Path(source['authorized_shared_readonly_path']).resolve()
        if source['evaluation_group']!='regression64' or p.parent!=allowed or p.name!=source['name']:
            raise ValueError('Only previously authorized exact shared regression media')
        return p
    p=(ROOT/source['relative_path']).resolve()
    if not p.is_relative_to((ROOT/'datasets').resolve()):raise ValueError('Only own dataset media otherwise')
    return p

def shared():
    sys.path.insert(0,str(ROOT/'experiments/2b-evidence-selector-train-v1'))
    from selector_export import shared as old
    return old()

def load_heads(torch):
    from selector_model import EvidenceSelector
    result={}
    for name in ('content_visible','history_blind'):
        checkpoint=torch.load(HEADS/(name+'.pt'),map_location='cpu',weights_only=True)
        model=EvidenceSelector();model.load_state_dict(checkpoint['state_dict']);model.eval();model.requires_grad_(False)
        if sum(p.numel() for p in model.parameters())!=414465:raise ValueError('head capacity changed')
        if checkpoint['history_blind']!=(name=='history_blind'):raise ValueError('condition mismatch')
        result[name]=model
    return result

def descriptors(torch,nodes,query_time):
    from selector_model import metadata
    if len(nodes)!=8 or any(len(n.features)!=108 or n.features.shape[1]!=2048 for n in nodes):
        raise ValueError('fixed LS4_4 actual bank required')
    if any(n.end>query_time for n in nodes):raise ValueError('future input')
    recent=nodes[-4:];history=nodes[:-4]
    def desc(values):
        x=torch.cat([n.features for n in values]);return torch.cat((x.mean(0),x.max(0).values))
    return dict(recent=desc(recent)[None],history=torch.stack([desc([n]) for n in history])[None],
        metadata=torch.tensor([metadata(n.metadata(),query_time,recent[0].begin) for n in history],dtype=torch.float32)[None],
        mask=torch.ones((1,4),dtype=torch.bool))

def choose(torch,heads,name,q,values):
    from selector_model import select
    scoring_name='content_visible' if name=='matched_latest' else name
    logits,route=heads[scoring_name](q[None],values['recent'],values['history'],values['metadata'],values['mask'],
        blind_history=scoring_name=='history_blind')
    decision=select(logits[0],route[0],[108]*4)
    selected=decision['history_indices']
    if name=='matched_latest':selected=list(range(4-len(selected),4))
    return dict(selected_history_indices=selected,route_logit=float(route[0]),
        reading='history' if selected else 'recent',evidence_verified=False,
        matched_count_uses_visible_head=name=='matched_latest')

def allowed_selections():
    import itertools
    return [list(x) for n in range(3) for x in itertools.combinations(range(4),n)]+[[0,1,2,3]]
