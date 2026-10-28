"""Pin old runtime, unchanged R1 encoder, four-arm follow-up and CPU/input receipts."""
import ast
from pathlib import Path
from quality_common import ROOT,OUT,read,save,sha,verify


def main():
    assert read(OUT/'local_CPU_preflight.json')['status']=='passed'
    deps=read(ROOT/'runs/resolution_cost_prepare_20261025_01/inputs/execution_lock.json')['runtime_sha256'].copy()
    # Ensure the measured R1 algorithm is reused byte for byte.
    old=read(ROOT/'runs/vision_midprune_prepare_20261025_01/inputs/execution_lock.json')
    key='experiments/2b-vision-midprune-v1/midprune.py';assert sha(ROOT/key)==old['runtime_sha256'][key]
    deps[key]=old['runtime_sha256'][key]
    for name,digest in deps.items():
        p=ROOT/name
        if name.startswith('repo/') and not (ROOT/'repo').exists():p=ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(p)==digest,name
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.sh','.md'):
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
            if p.suffix=='.sh':assert 13 not in p.read_bytes()
            deps[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'inputs/execution_lock.json',dict(runtime_sha256=deps,
        input_sha256={p.name:sha(p) for p in (OUT/'inputs').iterdir() if p.is_file()},
        local_CPU_sha256=sha(OUT/'local_CPU_preflight.json'),new_specific_GPU_approval_required=True))
    verify();print(dict(status='frozen',runtime_files=len(deps),execution_lock_SHA=sha(OUT/'inputs/execution_lock.json')),flush=True)


if __name__=='__main__':main()
