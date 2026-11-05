"""Fixed six-case mechanism diagnostic; no new method or training."""
import json, hashlib, sys, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/prune_timing_prepare_20261103_01'
for name in ['2b-content-main-v1','2b-content-quality-v1','streamingbench-router-v1','2b-text-control-v1','2b-perception-transition-v1']:
    sys.path.insert(0,str(ROOT/'experiments'/name))
from session import Session as OriginalSession
import session as original_module
from v0_source import V0Pairs
from canonical_prompt import question_prompt,rotate
from sember_common import runtime
from resolution_runtime import make_builder
MODES=['full','post_fixed','mid_fixed','post_content','mid_content']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify():
    for rel,h in read(PREP/'execution_lock.json').items():
        p=ROOT/rel
        if rel.startswith('repo/') and not p.exists():p=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(p)==h,rel
def choice(text,shift):
    m=re.search(r'\b([A-D])\b',text.strip().upper())
    return None if m is None else (ord(m[1])-65+shift)%4

class AuditedSession(OriginalSession):
    """Instrument existing encoder unchanged; release per-pair tensors immediately."""
    def append_pair(self,*args,**kwargs):
        import torch
        records={};original=original_module.encode
        def traced(*a,**k):
            k['return_trace']=True
            value,ids,info=original(*a,**k)
            assert k['mode'] not in records
            records[k['mode']]=(value,ids,info)
            return value,ids,info
        original_module.encode=traced
        try:result=super().append_pair(*args,**kwargs)
        finally:original_module.encode=original
        checks={}
        try:
            assert set(records)==set(MODES)-{'full'}
            for kind in ['fixed','content']:
                p,pi,pt=records['post_'+kind];m,mi,mt=records['mid_'+kind]
                assert torch.equal(pi,mi),'different selected IDs: '+kind
                assert pi.shape==(54,) and p.shape==m.shape
                cut=pt['cut_after_layers']
                assert pt['layer_tokens']==[432]*(cut*2)
                assert mt['layer_tokens']==[432]*cut+[216]*cut
                delta=(m-p).float()
                checks[kind]=dict(ids=pi.cpu().tolist(),same_IDs=True,
                    relative_feature_L2=float(delta.norm()/p.float().norm().clamp_min(1e-12)),
                    cut_after_layers=cut,post_layer_tokens=pt['layer_tokens'],mid_layer_tokens=mt['layer_tokens'])
            assert torch.equal(self.bank.scorers['post_content'].previous,self.bank.scorers['mid_content'].previous)
            self.last_pair_audit=dict(pair_index=self.pairs-1,checks=checks,previous_middle_features_equal=True)
        finally:records.clear()
        return result
