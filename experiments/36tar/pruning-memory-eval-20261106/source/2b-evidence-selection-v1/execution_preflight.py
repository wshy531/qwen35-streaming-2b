"""Verify real recorded token layouts without pretrained weights or video IO."""
import os,sys,itertools,hashlib
from prepare import ROOT,PREP,read,save
from selection import subset_columns

def main(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1' or output.exists():raise RuntimeError('CPU and new output required')
    sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
    sys.path.insert(0,str(ROOT/'experiments/2b-perception-transition-v1'))
    sys.path.insert(0,str(ROOT/'experiments/2b-temporal-router-v1'))
    from sember_common import runtime
    from sember_runtime import make_builder
    from canonical_prompt import question_prompt,rotate
    from temporal_policy import lexical_route
    env=runtime();torch=env['torch'];torch.set_num_threads(1)
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    tasks={r['id']:r for r in read(PREP/'inputs/requests.json')};plan=read(PREP/'runtime_plan.json')
    rows=[];refs=0;unavailable=0
    for item in plan['tasks']:
        t=tasks[item['task_id']];nodes=item['nodes'];chunks=[]
        for n in nodes:
            label=f"<{n['center']:.1f} seconds>" if n['first']==n['last'] else f"<{n['begin']:.1f}-{n['end']:.1f} seconds, compressed memory>"
            chunks.append(builder.tokenize(label+builder.processor.vision_start_token+
                builder.processor.video_token*n['tokens']+builder.processor.vision_end_token))
        for shift in range(5):
            text=question_prompt(rotate(t,shift)[0]);suffix=builder.question_ids(text)
            decisions=dict(recent=[],full=[0,1,2,3],
                lexical=[] if lexical_route(t['question'])['route']=='recent' else [0,1,2,3],
                time_matched_nodes=item['selection']['selected_history_indices'],
                nonmatching_nodes=item['selection']['control_history_indices'])
            for arm,sel in decisions.items():
                if sel is None:
                    unavailable+=1;rows.append(dict(task_id=t['id'],shift=shift,arm=arm,available=False));continue
                ids=torch.cat([builder.prefix_ids,*[chunks[i] for i in sel+[4,5,6,7]],suffix],1)
                digest=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                if arm in ('recent','full'):
                    ref=next(r for r in item['references'] if r['arm']==arm and r['shift']==shift)
                    assert digest==ref['input_ids_sha256'],(t['id'],arm,shift)
                    refs+=1
                assert ids.shape[1]+8<=8192
                rows.append(dict(task_id=t['id'],shift=shift,arm=arm,available=True,
                    input_tokens=ids.shape[1],input_ids_sha256=digest,
                    selected_history_indices=sel,visual_tokens=432+108*len(sel)))
    assert refs==640 and len(rows)==1600 and unavailable==55
    import worker
    from unittest.mock import patch
    with patch.dict(os.environ,{'SLURM_JOB_ID':'','SLURM_CPUS_PER_TASK':'4'}):
        try:worker.main(ROOT/'runs/evidence_selection_forbidden_CPU_output',None)
        except RuntimeError as e:assert 'Slurm' in str(e)
        else:raise AssertionError('GPU entry accepted without allocation')
    save(output,dict(status='passed_real_recorded_token_layouts_CPU',rows=rows,reference_inputs_exact=refs,
        planned_slots=len(rows),available_layouts=len(rows)-unavailable,unavailable_control_slots=unavailable,
        pretrained_loads=0,GPU_jobs=0,real_node_feature_values_not_reconstructed=True))
    print(dict(reference_inputs_exact=refs,available_layouts=len(rows)-unavailable,unavailable_controls=unavailable))
if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
