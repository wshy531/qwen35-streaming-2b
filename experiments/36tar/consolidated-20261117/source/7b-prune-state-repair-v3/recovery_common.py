import sys,json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
REC=ROOT/'runs/7b_prune_state_repair_20261111_01'

def specification(mode):
    return json.loads((REC/'recovery.json').read_text(encoding='utf8'))[mode]

def check_control(row,references,mode):
    fields=('id','subset','shift') if mode=='selector' else ('id','mode','route','shift')
    refs=[r for r in references if all(r[k]==row[k] for k in fields)]
    assert len(refs)==1
    ref=refs[0]
    for k in ('selected_original','visual_tokens'):assert row[k]==ref[k],(fields,k)
    torch.testing.assert_close(torch.tensor(row['option_logits']),torch.tensor(ref['option_logits']),rtol=1e-5,atol=1e-5)

def verify_recovery():
    lock=json.loads((REC/'lock.json').read_text(encoding='utf8'))
    for rel,h in lock.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
