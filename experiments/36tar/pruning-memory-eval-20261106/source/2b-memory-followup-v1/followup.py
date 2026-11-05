"""Frozen API33 + LS and A06 adapters for the existing StreamingBench150 protocol."""
import sys, dataclasses, time
from pathlib import Path
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'experiments/2b-protection-streamingbench-v1'))
from fair_common import runtime,make_builder,context,Session,split,compact,protected_readout,read,save,sha,prompt,pair_digest,V0Pairs
P=R/'runs/protection_streamingbench_prepare_20261103_01'
OUT=R/'runs/memory_followup_prepare_20261103_01'
sys.path.insert(0,str(R/'experiments/2b-read-benefit-v1'))
from feature_runtime import extract,inputs
sys.path.insert(0,str(R/'scripts'))
from router_head_predict import predict
HEAD_PATH=R/'runs/router_head_training_delivery_20261029_01/heads_all.json'
HEAD='round1_balanced/API/recent_video'

def legacy_nodes(nodes):
    # Explicit type conversion: old extractor validates the original dataclass.
    from sember_nodes import Node,validate
    result=[]
    for n in nodes:
        x=Node(**{f.name:getattr(n,f.name) for f in dataclasses.fields(Node)}).copy()
        validate(x);assert x.metadata()==n.metadata();result.append(x)
    return result

def choose(env,model,builder,h,r,question,bundle):
    if not h:return dict(route='recent',score=None,score_is_probability=False,head_called=False,seconds=0.)
    assert len(r)==4
    device=model.get_input_embeddings().weight.device
    if device.type=='cuda':env['torch'].cuda.synchronize()
    tick=time.monotonic()
    hidden,meta=extract(env,model,builder,legacy_nodes(r),question,True)
    score,route=predict(bundle,HEAD,hidden.numpy()[None])
    if device.type=='cuda':env['torch'].cuda.synchronize()
    return dict(route=str(route[0]),score=float(score[0]),score_is_probability=False,head_called=True,
        seconds=time.monotonic()-tick,feature_meta=meta)

def answer(env,model,builder,session,q,bundle):
    h,r=split(session.bank.banks['LS_mid108']);before=session.bank.fingerprint()
    decision=choose(env,model,builder,h,r,q['question'],bundle)
    if decision['route']=='full':z=session.ask('LS_mid108',prompt(q),lambda _:None,max_new_tokens=16)
    else:z=compact(protected_readout.answer(env,model,builder,[],r,prompt(q),isolated=False,limit=16))
    assert session.bank.fingerprint()==before
    return dict(z,decision=decision)
