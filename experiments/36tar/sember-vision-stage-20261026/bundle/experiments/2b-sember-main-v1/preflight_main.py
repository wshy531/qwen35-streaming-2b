"""Formal roster/orchestration/guard checks, not another model capability run."""
import ast
import os
import subprocess
import sys
from pathlib import Path
from main_common import ROOT,OUT,PREP,ARMS,read,save,sha
from evaluate_main import group_queries,Session,CausalSource,PairDriver,prompt


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    inputs=OUT/'inputs';protocol=read(inputs/'protocol.json');requests=read(inputs/'requests.json')
    videos=read(inputs/'videos.json');expected=read(inputs/'expected_prefixes.json')
    source_lock=read(PREP/'selection/selection_lock.json')
    assert len(requests)==478 and len(videos)==128 and len(group_queries(requests))==128
    assert {q['question_id'] for q in requests}==set(source_lock['formal_question_ids'])
    assert set(expected)=={q['question_id'] for q in requests}
    forbidden={'correct_index','correct_letter','ground_truths','answer_start_time','answer_end_time','official_memory_recency','end_gap_proxy_s'}
    assert all(not forbidden.intersection(q) for q in requests)
    assert protocol['measured_answers']==478*(len(ARMS)+1)==5736
    assert protocol['generation_tokens_max']==(5736+12)*16==91968
    pairs=0
    for vid,qs in group_queries(requests):
        assert all(a['question_time']<=b['question_time'] for a,b in zip(qs,qs[1:]))
        pairs+=expected[qs[-1]['question_id']]['pairs']
        for q in qs:
            p=expected[q['question_id']]
            assert p['last_committed_pts']<=q['question_time']+1e-9
            assert p['real_frames']==p['pairs']*2+p['pending_real_frames']-int(p['true_EOF_tail_padded'])
    assert pairs==14678 and 2*pairs+2==protocol['visual_encodings_including_warmup']
    import evaluate_profile
    for name,original in (('Session',Session),('CausalSource',CausalSource),('PairDriver',PairDriver),('prompt',prompt)):
        assert getattr(evaluate_profile,name) is original,name
    # Actual command entry points must refuse a no-allocation launch before weight loading.
    environment=os.environ.copy();environment.pop('SLURM_JOB_ID',None)
    worker=subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_main.py')),'--mode','bounded',
        '--out',str(OUT/'SHOULD_NOT_EXIST'),'--deadline','0'],env=environment,capture_output=True,text=True)
    assert worker.returncode!=0 and 'explicitly approved Slurm GPU allocation required' in worker.stderr
    assert not (OUT/'SHOULD_NOT_EXIST').exists()
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    script=Path(__file__).with_name('run_main.sh');assert b'\r' not in script.read_bytes()
    result=dict(status='passed',questions=478,sources=128,measured_answers=5736,warmup_answers=12,
        actual_pairs_per_process=pairs,vision_encodings_including_warmup=29358,pretrained_loads_planned=2,
        runtime_classes_identical_to_GPU_profile=True,roster_unchanged=True,labels_absent=True,
        all_query_prefixes_causal=True,unallocated_worker_refuses=True,
        actual_pretrained_loads=0,GPU_jobs=0,no_capability_claim=True)
    save(OUT/'main_CPU_preflight.json',result);print(result,flush=True)


if __name__=='__main__':main()
