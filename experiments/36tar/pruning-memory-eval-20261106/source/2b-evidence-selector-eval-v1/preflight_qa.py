"""Real tokenizer layouts, restored heads, and six tiny QA calls; no real model weights."""
import argparse,os,hashlib
from pathlib import Path
import numpy as np
from selector_qa_common import ROOT,PREP,HEADS,read,save,shared,load_heads,choose,allowed_selections

def main(out):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1' or out.exists():raise RuntimeError('CPU/new output required')
    base,env,make_builder,_,_,pair_node,digest,extract=shared()
    from canonical_prompt import question_prompt,rotate
    torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(20261030)
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    plan=read(PREP/'plan.json');tasks={t['id']:t for t in read(PREP/'requests.json')}
    refs={(r['task_id'],r['arm'],r['shift']):r for r in read(PREP/'references_LOCAL_ONLY.json')}
    rows=[];reference_matches=0
    for item in plan['checkpoints']:
        chunks=[]
        for n in item['nodes']:
            label=f"<{n['center']:.1f} seconds>" if n['first']==n['last'] else f"<{n['begin']:.1f}-{n['end']:.1f} seconds, compressed memory>"
            chunks.append(builder.tokenize(label+builder.processor.vision_start_token+
                builder.processor.video_token*n['tokens']+builder.processor.vision_end_token))
        for shift in range(5):
            suffix=builder.question_ids(question_prompt(rotate(tasks[item['task_id']],shift)[0]))
            for sel in allowed_selections():
                ids=torch.cat([builder.prefix_ids,*[chunks[i] for i in sel+[4,5,6,7]],suffix],1)
                h=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                assert ids.shape[1]+8<=8192
                arm='recent' if not sel else 'full' if len(sel)==4 else None
                ref=refs.get((item['task_id'],arm,shift))
                if ref:assert ref['input_ids_sha256']==h;reference_matches+=1
                rows.append(dict(task_id=item['task_id'],shift=shift,selection=sel,input_ids_sha256=h,
                    input_tokens=ids.shape[1],visual_tokens=432+108*len(sel)))
    assert len(rows)==6000 and reference_matches==640
    heads=load_heads(torch);decisions=[]
    ids=set(tasks)
    with np.load(ROOT/'runs/selector-features-219523/features.npz',allow_pickle=False) as f:
        indexes=[i for i,tid in enumerate(f['ids'].tolist()) if tid in ids]
        assert len(indexes)==36
        expected={name:{r['id']:r['history_indices'] for r in read(HEADS/(name+'_predictions.json'))}
                  for name in heads}
        with torch.inference_mode():
            for i in indexes:
                tid=f['ids'][i].item() if hasattr(f['ids'][i],'item') else str(f['ids'][i])
                q=torch.from_numpy(f['query'][i].copy())
                values={k:torch.from_numpy(f[k][i:i+1].copy()) for k in ('recent','history','metadata','mask')}
                picked={}
                for name in ('content_visible','history_blind','matched_latest'):
                    d=choose(torch,heads,name,q,values);picked[name]=d['selected_history_indices']
                    if name in heads:assert picked[name]==expected[name][tid]
                    assert len(picked[name])<=2
                    decisions.append(dict(task_id=tid,condition=name,selection=picked[name]))
                assert len(picked['matched_latest'])==len(picked['content_visible'])
    # CPU tiny random QA checks retained coordinates and repeat branch isolation, not model capability.
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](builder.processor),attn_implementation='sdpa')).float().eval()
    width=model.get_input_embeddings().embedding_dim
    nodes=[pair_node(torch.randn(108,width),i,2.*i,2.*i+1,(9,12)) for i in range(8)]
    before=digest(nodes);answer={};q=question_prompt(rotate(next(iter(tasks.values())),0)[0])
    with torch.inference_mode():
        for sel in ([],[0],[2,3],[2,3],[0],[]):
            result=base.infer(env,model,builder,nodes,q,sel)
            key=str(sel)
            if key in answer:assert answer[key]==result['generated_ids']
            answer[key]=result['generated_ids'];assert result['branch_released']
    assert before==digest(nodes) and model.model.rope_deltas is None
    import selector_qa_eval
    from unittest.mock import patch
    with patch.dict(os.environ,{'SLURM_JOB_ID':''}):
        try:selector_qa_eval.main(ROOT/'runs/forbidden_selector_qa_CPU')
        except RuntimeError as e:assert 'Slurm' in str(e)
        else:raise AssertionError('GPU entry accepted missing allocation')
    save(out,dict(status='passed',layout_rows=rows,reference_input_matches=640,
        restored_head_and_matched_decisions=decisions,tiny_random_QA_calls=6,
        restored_long_head_decisions=72,layout_positions_real_GPU_check_pending=True,
        no_pretrained_base_loads=True,no_real_QA=True,GPU_jobs=0,
        maximum_input_tokens=max(r['input_tokens'] for r in rows)))
    print(dict(layouts=6000,reference_inputs=640,restored_heads=72,matched=36,tiny_QA=6,GPU=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
