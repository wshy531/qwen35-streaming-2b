import os,sys,argparse,json
from pathlib import Path
from types import SimpleNamespace
from audit_common import *
def main(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261103)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor,(288,384));layouts=[]
    for q in read(PREP/'requests.json'):
        for shift in range(4):
            text=question_prompt(rotate(q,shift)[0]);assert builder.question_ids(text).shape[1]<512
            for mode in MODES:layouts.append(dict(id=q['id'],shift=shift,mode=mode,question_tokens=builder.question_ids(text).shape[1]))
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=4
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
    from quality_banks import selected_node
    from resolution_runtime import context
    contracts={x['id']:x for x in read(PREP/'sample_contract.json')};actual_layouts=[]
    for q in read(PREP/'requests.json'):
        frames=contracts[q['id']]['frames']
        for mode in MODES:
            # Synthetic values, actual pair times/counts. Content IDs are a layout proxy.
            ids=t.arange(108) if mode=='full' else t.arange(1,108,2)
            nodes=[selected_node(t.zeros((len(ids),model.get_input_embeddings().embedding_dim)),ids,
                frames[i]['pair_index'],frames[i]['pts_s'],frames[i+1]['pts_s']) for i in range(0,len(frames),2)]
            for shift in range(4):
                text=question_prompt(rotate(q,shift)[0])
                emb,pos,token_ids,meta=context(model,builder,nodes,text,max_new_tokens=8)
                assert meta['visual_tokens']==len(nodes)*len(ids) and meta['input_tokens']+8<=4096
                actual_layouts.append(dict(id=q['id'],mode=mode,shift=shift,input_tokens=meta['input_tokens'],
                    visual_tokens=meta['visual_tokens'],ids_sha256=__import__('hashlib').sha256(token_ids.numpy().tobytes()).hexdigest()))
    import numpy as np
    rng=np.random.default_rng(20261103);tested=AuditedSession(env,model,builder);reference=OriginalSession(env,model,builder)
    with t.inference_mode():
        for i in range(9):
            a=SimpleNamespace(frame_id=2*i,timestamp=float(2*i),rgb=rng.integers(0,256,(288,384,3),np.uint8))
            b=SimpleNamespace(frame_id=2*i+1,timestamp=float(2*i+1),rgb=rng.integers(0,256,(288,384,3),np.uint8))
            tested.append_pair(a,b);reference.append_pair(a,b)
            assert tested.bank.selector_digest()==reference.bank.selector_digest()
            # Compare actual exported tensors, not only bookkeeping.
            for mode in MODES:
                left=tested.bank.banks[mode].nodes();right=reference.bank.banks[mode].nodes()
                assert len(left)==len(right)==min(i+1,8)
                for x,y in zip(left,right):
                    assert x.metadata()==y.metadata()
                    for key in ('features','mass','coords'):
                        assert t.equal(getattr(x,key),getattr(y,key)),key
        before=tested.status();digest=tested.bank.selector_digest();rows=[]
        text=question_prompt(rotate(read(PREP/'requests.json')[0],0)[0])
        for mode in MODES:
            a=tested.ask(mode,text,lambda _:None,max_new_tokens=2)
            b=reference.ask(mode,text,lambda _:None,max_new_tokens=2)
            assert a['generated_ids']==b['generated_ids'] and a['text']==b['text']
            assert a['branch_released'] and a['read_state_unchanged']
            rows.append(dict(mode=mode,generated_ids=a['generated_ids'],visual_tokens=a['visual_tokens']))
        assert tested.status()==before and tested.bank.selector_digest()==digest
        tested.close();reference.close()
    deps=[]
    for mod in list(sys.modules.values()):
        value=getattr(mod,'__file__',None)
        if value:
            p=Path(value).resolve()
            if p.suffix=='.py' and p.is_relative_to(ROOT.resolve()):
                rel=p.relative_to(ROOT.resolve()).as_posix()
                if rel.startswith(('experiments/','scripts/','repo/','publish/')):deps.append(rel)
    save(out,dict(status='passed',layouts=actual_layouts,question_layouts=layouts,rows=rows,identical_ID_checks=18,old_session_exact_pairs=9,
        tiny_QA=10,GPU=0,pretrained_loads=0,loaded_source_files=sorted(set(deps))))
    print('120 layouts, 18 selection-ID checks, 9 exact session continuations, 10 tiny QA passed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
