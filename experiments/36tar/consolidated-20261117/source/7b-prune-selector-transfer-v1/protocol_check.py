"""All question/option layouts, split integrity, actual tiny scorer/QA path."""
from common import *
from selector import prompt,option_ids,choices_mask,forward
from transformers import AutoProcessor
from native_cpu import make_tiny

def main(out):
    torch.set_num_threads(1);torch.manual_seed(7110)
    p=AutoProcessor.from_pretrained((ROOT/'runs/7b_transfer_prepare_20261107_01/processor_assets' if (ROOT/'runs/7b_transfer_prepare_20261107_01/processor_assets').exists() else ROOT/'models/Qwen2.5-VL-7B-Instruct'),local_files_only=True)
    groups=read(PREP/'fit_sources.json');evaluation=read(PREP/'evaluation.json')
    train=[q for g in groups if g['role']=='train' for q in g['questions']]
    dev=[q for g in groups if g['role']=='development' for q in g['questions']]
    assert (len(train),len(dev),len(evaluation))==(144,32,288)
    for key in ('media_sha256','source_id'):
        a,b,c=[{q[key] for q in rows} for rows in (train,dev,evaluation)]
        assert not(a&b or a&c or b&c),key
    assert {q['suite'] for q in evaluation}=={'D18','SB150','S-EMBER-early-candidate'}
    bank=Memory()
    for i in range(16):bank.append(torch.zeros(140,64),torch.arange(140),i,2.*i,2.*i+1)
    recent,history=bank.parts();choices,_=choices_mask(len(history));h=hashlib.sha256();layouts=0;maximum=0
    for q in train+dev+evaluation:
        option_ids(p,len(q['options']))
        for c in choices:
            nodes=[history[i] for i in c]+recent
            for shift in (0,2):
                ids,pos,_=segmented.layout(p,nodes,prompt(q,shift),'segmented')
                assert len(ids.shape)==2 and pos.shape==(3,1,ids.numel())
                h.update(ids.numpy().tobytes());h.update(pos.numpy().tobytes());layouts+=1;maximum=max(maximum,ids.numel())
    model=make_tiny(real_tokens=True).float().eval().requires_grad_(False)
    micro=[]
    for q in (train[0],evaluation[0]):
        for shift in (0,2):
            values=forward(model,p,recent,prompt(q,shift))[option_ids(p,len(q['options']))]
            assert torch.isfinite(values).all();micro.append(dict(id=q['id'],shift=shift,selected=int(values.argmax())))
    r=dict(passed=True,layouts=layouts,layout_sha=h.hexdigest(),max_tokens=maximum,source_sha_disjoint=True,
        train=144,development=32,evaluation=288,main=270,diagnostic=18,micro=micro,GPU=0)
    save(out,r);print(json.dumps(r))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);main(ap.parse_args().out)
