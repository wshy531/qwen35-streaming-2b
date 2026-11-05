"""Preserve active experiment identities; freeze only this independent control."""
import ast
from pathlib import Path
from text_common import ROOT,OUT,MAIN,read,save,sha,verify


def main():
    assert read(OUT/'local_CPU_preflight.json')['status']=='passed'
    deps=read(MAIN/'inputs/execution_lock.json')['runtime_sha256'].copy()
    for rel,h in deps.items():
        path=ROOT/rel
        if rel.startswith('repo/') and not (ROOT/'repo').exists():path=ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(path)==h,rel
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.sh','.md'):
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
            deps[path.relative_to(ROOT).as_posix()]=sha(path)
    deps['scripts/analyze_sember_main.py']=sha(ROOT/'scripts/analyze_sember_main.py')
    save(OUT/'inputs/execution_lock.json',dict(runtime_sha256=deps,
        input_sha256={p.name:sha(p) for p in (OUT/'inputs').iterdir() if p.is_file()},
        local_CPU_SHA=sha(OUT/'local_CPU_preflight.json'),explicit_new_GPU_approval_required=True))
    save(OUT/'local_analysis_identity.json',dict(labels_SHA=sha(ROOT/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json'),
        baseline_scored_SHA=sha(MAIN/'scored_rows_LOCAL_ONLY.json'),source_interval_code_SHA=deps['scripts/analyze_sember_main.py'],
        labels_and_baseline_scores_not_deployed=True))
    verify();print(dict(status='frozen',runtime_files=len(deps),execution_lock_SHA=sha(OUT/'inputs/execution_lock.json')))


if __name__=='__main__':main()
