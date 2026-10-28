"""Freeze one candidate and its original implementation dependencies, not a parameter sweep."""
import ast
from pathlib import Path
from common import ROOT,OUT,read,save,sha,verify


def main():
    assert read(OUT/'local_CPU_preflight.json')['status']=='passed'
    old=read(ROOT/'runs/sember_main_prepare_20261025_01/inputs/execution_lock.json')
    deps=old['runtime_sha256'].copy()
    for name,digest in deps.items():
        path=ROOT/name
        if name.startswith('repo/') and not (ROOT/'repo').exists():path=ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(path)==digest,name
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.sh','.md'):
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
            if path.suffix=='.sh':assert 13 not in path.read_bytes()
            deps[path.relative_to(ROOT).as_posix()]=sha(path)
    save(OUT/'inputs/execution_lock.json',dict(runtime_sha256=deps,
        input_sha256={p.name:sha(p) for p in (OUT/'inputs').iterdir() if p.is_file()},
        local_CPU_sha256=sha(OUT/'local_CPU_preflight.json'),new_GPU_approval_required=True))
    verify();print(dict(status='frozen',runtime_files=len(deps),lock_sha256=sha(OUT/'inputs/execution_lock.json')),flush=True)


if __name__=='__main__':main()
