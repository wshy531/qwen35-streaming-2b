"""All frozen question suffixes against adversarial bounded bank layouts; no LM."""
import os
import torch
from sember_common import ROOT,OUT,PREP,ARMS,read,save,runtime
from sember_nodes import pair_node,merge
from sember_banks import Banks
from sember_runtime import make_builder,context
from sember_protocol import prompt


class EmbeddingsOnly:
    def __init__(self):self.embedding=torch.nn.Embedding(248320,32)
    def get_input_embeddings(self):return self.embedding


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1);torch.manual_seed(975)
    env=runtime();processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor);model=EmbeddingsOnly();banks=Banks()
    for i in range(160):banks.append(torch.randn(108,32),i,i*4.,i*4.+2)
    rows=[]
    with torch.inference_mode():
        for q in read(PREP/'selection/requests.json'):
            text=prompt(q)
            for arm in ARMS:
                bank=banks.banks[arm]
                emb,pos,ids,meta=context(model,builder,bank.export(),text,visual_limit=bank.budget)
                assert torch.equal(pos[0,0],torch.arange(ids.shape[1])) and emb.shape[1]==ids.shape[1]
                assert pos.shape==(4,1,ids.shape[1]) and pos.max()<262144
                rows.append(dict(question_id=q['question_id'],arm=arm,input_tokens=meta['input_tokens'],
                    visual_tokens=meta['visual_tokens'],suffix_tokens=meta['suffix_tokens']))
        from forest import Node as OldNode
        from readout import context as old_context,Limits
        builder48=make_builder(env,processor,(192,256))
        a=pair_node(torch.randn(48,32),0,0.,2.,(6,8));b=pair_node(torch.randn(48,32),1,4.,6.,(6,8))
        nodes=[merge(a,b),pair_node(torch.randn(48,32),2,8.,10.,(6,8))]
        fields=('features','mass','coords','first','last','begin','end','center','groups','depth')
        old_nodes=[OldNode(**{k:getattr(n,k) for k in fields}) for n in nodes]
        new=context(model,builder48,nodes,'What is visible?',visual_limit=864)
        old=old_context(model,builder48,old_nodes,'What is visible?',limits=Limits(4096,864,32,512))
        assert all(torch.equal(a,b) for a,b in zip(new[:3],old[:3]))
    result=dict(status='passed',question_count=580,layout_count=len(rows),rows=rows,
        max_input_tokens=max(r['input_tokens'] for r in rows),max_suffix_tokens=max(r['suffix_tokens'] for r in rows),
        layout_source='fixed synthetic 160-pair budget-stress bank states; not real visual features or per-query model states',
        old_48_readout_embeddings_ids_positions_exact=True,LM_forward_calls=0,pretrained_loads=0,GPU_jobs=0)
    save(OUT/'layout_preflight.json',result);print({k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':main()
