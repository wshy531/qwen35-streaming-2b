"""Lock isolated sources, reused validated runtime and answer-free tasks."""
import ast
from pathlib import Path
from resolution_common import ROOT,OUT,MAIN,SPECS,read,save,sha,verify


def main():
    assert read(OUT/'local_CPU_preflight.json')['status']=='passed'
    p=read(OUT/'inputs/protocol.json')
    tasks=[]
    for name,canvas in [('quality_mid',(288,384)),('quality_high',(384,512))]:
        tasks.append(dict(name=name,kind='quality',canvas=canvas,arms=[a for a,s in SPECS.items() if s['canvas']==canvas],videos=None))
    for block,vids in enumerate(p['cost_blocks']):
        views=p['cost_views'];order=views[block:]+views[:block]
        for arm in order:tasks.append(dict(name='cost_b'+str(block)+'_'+arm,kind='cost',canvas=SPECS[arm]['canvas'],arms=[arm],videos=vids))
    assert len(tasks)==11
    save(OUT/'inputs/tasks.json',tasks)
    original=read(MAIN/'inputs/execution_lock.json');deps=original['runtime_sha256'].copy()
    # Check every dependency of the active job without changing any of them.
    for rel,h in deps.items():
        path=ROOT/rel
        if rel.startswith('repo/') and not (ROOT/'repo').exists():path=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(path)==h,rel
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.sh','.md'):
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
            deps[path.relative_to(ROOT).as_posix()]=sha(path)
    save(OUT/'inputs/execution_lock.json',dict(runtime_sha256=deps,
        input_sha256={v.name:sha(v) for v in (OUT/'inputs').iterdir() if v.is_file()},
        local_preflight_SHA=sha(OUT/'local_CPU_preflight.json'),prior_active_job_sources_unchanged=True))
    save(OUT/'local_analysis_identity.json',dict(labels_SHA=sha(ROOT/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json'),
        scorer_SHA=sha(Path(__file__).with_name('analyze.py')),labels_not_deployed=True))
    verify();print(dict(status='frozen',runtime_files=len(deps),tasks=len(tasks),execution_lock_SHA=sha(OUT/'inputs/execution_lock.json')))


if __name__=='__main__':main()
