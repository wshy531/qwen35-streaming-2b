"""Exercise new inference, stream boundary, and decision gates without real weights."""
import os,sys,json
from pathlib import Path
from prepare import ROOT,PREP,read,save
import worker

def main(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1' or output.exists():raise RuntimeError('CPU and new receipt required')
    env,make_builder,*_=worker.setup();torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(20261029)
    from sember_nodes import pair_node,digest
    from canonical_prompt import question_prompt,rotate
    from analyze import score
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor)
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](processor),attn_implementation='sdpa')).float().eval()
    width=model.get_input_embeddings().embedding_dim
    nodes=[pair_node(torch.randn(108,width),i,2.*i,2.*i+1,(9,12)) for i in range(8)];before=digest(nodes)
    tasks=read(PREP/'inputs/requests.json');labels=read(PREP/'audit/labels_LOCAL_ONLY.json');q=question_prompt(rotate(tasks[0],0)[0])
    answers={}
    with torch.inference_mode():
        for selected in ([],[0],[0,1],[0,1],[0],[]):
            r=worker.infer(env,model,builder,nodes,q,selected)
            key=str(selected)
            assert r['branch_released'] and 1<=len(r['generated_ids'])<=8
            if key in answers:assert answers[key]==r['generated_ids']
            answers[key]=r['generated_ids']
    assert digest(nodes)==before and model.model.rope_deltas is None
    # Synthetic counterexamples ensure current-only gains cannot open training.
    arms=read(PREP/'protocol.json')['arms'];fake=[]
    for t in tasks:
        for s in range(5):
            for arm in arms:
                correct=arm=='time_matched_nodes' or (t['role']=='current' and arm in ('recent','lexical'))
                idx=labels[t['id']] if correct else (labels[t['id']]+1)%5
                fake.append(dict(task_id=t['id'],arm=arm,shift=s,available=True,text=chr(65+(idx-s)%5)))
    positive=score(fake,tasks,labels);assert positive['passed']
    only_current=[dict(r) for r in fake]
    old={t['id'] for t in tasks if t['role']=='old'}
    for r in only_current:
        if r['task_id'] in old and r['arm']=='time_matched_nodes':
            r['text']=chr(65+((labels[r['task_id']]+1-r['shift'])%5))
    negative=score(only_current,tasks,labels);assert not negative['passed'] and not negative['gates']['old_correction_sources_at_least_two']
    prose=[dict(r,text=r['text']+'. Explanation omitted.') for r in fake]
    prose_score=score(prose,tasks,labels)
    assert all(prose_score['groups'][k]['correct']==v['correct'] for k,v in positive['groups'].items())
    # Real CPU decoder input, bounded synthetic fixture, no project media modification.
    import av,numpy as np
    fixture=PREP/'synthetic_decoder_fixture.mp4'
    if fixture.exists():raise FileExistsError(fixture)
    with av.open(str(fixture),'w') as c:
        v=c.add_stream('mpeg4',rate=2);v.width=64;v.height=48;v.pix_fmt='yuv420p'
        for i in range(16):
            image=np.full((48,64,3),i*10,dtype=np.uint8)
            f=av.VideoFrame.from_ndarray(image,format='rgb24')
            for packet in v.encode(f):c.mux(packet)
        for packet in v.encode():c.mux(packet)
    spec=dict(check_frames=[],cutoff_s=7.,max_pairs=4)
    decoded=list(worker.sampled_pairs(fixture,spec,lambda:None))
    assert len(decoded)==4 and decoded[0][0].timestamp==0 and decoded[-1][1].timestamp==7
    save(output,dict(status='passed_new_worker_CPU_smoke',tiny_random_QA_calls=6,generated_length_limit=8,
        real_decoder_fixture_frames=16,sampled_real_CPU_fixture_pairs=4,bank_unchanged=True,
        repeat_branch_generation_ids_equal=True,current_only_improvement_gate_rejected=True,
        positive_gate_control_passed=True,format_changes_not_penalized_in_content=True,
        real_pretrained_loads=0,GPU_jobs=0,API_calls=0,real_benchmark_answers=0))
    print('Passed: 6 tiny QA calls, CPU video sampling, repeat isolation and decision gate controls; real models/GPU0.')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
