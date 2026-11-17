"""Frozen 7B pruning and utility-head composition; no new model training."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/7b-prune-selector-transfer-v1'))
from common import *
from selector import Head,descriptor_features,choices_mask,choose,forward,prompt,option_ids
from fit_selector import FIELDS
sys.path.insert(0,str(ROOT/'experiments/7b-prune-state-repair-v3'))
from state_contract import compare_state,NEW_CANVAS
from run_prune import check_pair
HERE=ROOT/'runs/7b_prune_head_prepare_20261111_01'

def verify_all():
    verify()
    for rel,h in read(HERE/'lock.json').items():assert sha(ROOT/rel)==h,rel

def load_head():
    plan=read(HERE/'plan.json');ck=ROOT/plan['frozen_head'];assert sha(ck)==plan['head_sha']
    value=torch.load(ck,map_location='cpu',weights_only=True)
    assert value['width']==3584 and value['epoch']==20
    head=Head();head.load_state_dict(value['state_dict']);return head.eval().requires_grad_(False)

@torch.no_grad()
def select_nodes(head,query,recent,history):
    choices,mask=choices_mask(len(history));f=descriptor_features(query,recent,history)
    assert all(torch.isfinite(f[k]).all() for k in FIELDS)
    values=head(*(f[k][None] for k in FIELDS),mask)[0]
    chosen=choices[choose(values,choices)]
    norms={k:float(f[k].float().norm()) for k in FIELDS if k!='node_mask'}
    return [history[i] for i in chosen]+recent,chosen,dict(predicted_gains=values.tolist(),norms=norms)

def check_reference(actual,expected):
    for k in ('selected_original','visual_tokens'):assert actual[k]==expected[k],(actual['id'],k)
    torch.testing.assert_close(torch.tensor(actual['option_logits']),torch.tensor(expected['option_logits']),rtol=1e-5,atol=1e-5)
