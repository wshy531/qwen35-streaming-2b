"""Freeze the existing six-source cost panel; do not choose examples by accuracy."""
import argparse
import collections
import shutil
from common import ROOT, PREP, CASES, read, write, sha, verify

def prepare():
    PREP.mkdir(exist_ok=False)
    old = ROOT/'runs/stream_prune_cost_prepare_20261102_01'
    panel = read(ROOT/'runs/bounded_prefetch_prepare_20261106_01/panel.json')
    assert len(panel) == len({q['source_id'] for q in panel}) == 6
    refs = {(r['id'], r['arm']): r for r in read(old/'reference.json')}
    selected = [refs[q['id'], arm] for q in panel for arm in ('full','fixed12','heuristic11')]
    # Four-period Williams crossover: equal position counts and balanced carryover.
    orders = ((0,1,3,2),(1,2,0,3),(2,3,1,0),(3,0,2,1))
    names = list(CASES)
    jobs = []
    for repeat in range(4):
        for i in (range(6) if repeat%2 == 0 else reversed(range(6))):
            for position, c in enumerate(orders[(repeat+i)%4]):
                jobs.append(dict(repeat=repeat, source_index=i, id=panel[i]['id'],
                                 case=names[c], order_position=position))
    assert len(jobs) == len({(j['repeat'],j['id'],j['case']) for j in jobs}) == 96
    for q in panel:
        for case in names:
            assert collections.Counter(j['order_position'] for j in jobs if j['id']==q['id'] and j['case']==case)=={0:1,1:1,2:1,3:1}
    warm = read(old/'warmup.json')
    assert warm['media_path'] in {q['media_path'] for q in panel}
    pairs = sum(refs[q['id'],'full']['source_state']['complete_pairs'] for q in panel)
    # Existing driver commits floor(query seconds / 2) pairs on this warm clip;
    # verify during CPU/real execution; the budget deliberately allows 8.
    plan = dict(sources=6, questions=6, repeats=4, cases=list(CASES), formal_answers=96,
                warmup_answers=4, maximum_answers=100, maximum_generated_tokens=12800,
                formal_visual_calls=pairs*16, warmup_visual_call_cap=32,
                maximum_visual_calls=pairs*16+32, FP32_model_loads=1, updates=0,
                scheduled_minutes=75, internal_minutes=70, GPU=1, CPU=4, memory_GiB=24,
                result_limit_MiB=64, primary_comparison='prefetch_fixed/content versus prefetch_full',
                input='1FPS 288x384, recent8pairs, max128 answer+explanation',
                primary_cost='sum of per-source median pipeline wall seconds over four repeats',
                accuracy='All 96 formal outputs must match their frozen method endpoints; not a new quality benchmark',
                decision='>=5% incremental wall-time reduction and faster on >=5/6 sources supports a deployment candidate only; else freeze pruning cost branch',
                stop='Any source/input/output mismatch or internal deadline: preserve partial rows and stop, no automatic retry',
                limitations=['A800 FP32 fast causal replay, not real-time or Jetson',
                             'six opened sources, not independent capability or broad noninferiority',
                             'one loaded model; separate fresh decoder and state for every run, no shared decoded frames/features',
                             'OS page cache not flushed; balanced order and repeated comparison, not cold-start I/O',
                             'no timer arithmetic that adds overlapping CPU and GPU work'])
    for name, value in [('panel',panel),('reference',selected),('warmup',warm),('jobs',jobs),('plan',plan)]:
        write(PREP/f'{name}.json',value)
    shutil.copyfile(ROOT/'runs/frontend_closure_20261109_01/CPU/synthetic_timing_fixture.mp4',PREP/'fixture.mp4')
    print(plan)

def freeze():
    cpu=read(PREP/'CPU_local.json')
    assert cpu['status']=='passed_random_weights_only'
    lock=dict(cpu['loaded_project_source_SHA'])
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.sh','.md'):
            lock[path.relative_to(ROOT).as_posix()]=sha(path)
    for name in ['panel.json','reference.json','warmup.json','jobs.json','plan.json','fixture.mp4']:
        path=PREP/name;lock[path.relative_to(ROOT).as_posix()]=sha(path)
    write(PREP/'execution_lock.json',lock)
    print('LOCKED',len(lock))

if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze','verify']);p.add_argument('--media',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='freeze':freeze()
    else:verify(a.media);print('VERIFIED',len(read(PREP/'execution_lock.json')))
