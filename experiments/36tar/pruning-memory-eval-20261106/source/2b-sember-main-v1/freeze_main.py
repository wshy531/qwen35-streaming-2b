"""Add formal orchestration/input locks without touching the validated core."""
import ast
from pathlib import Path
from main_common import ROOT,OUT,read,save,sha


def main():
    assert read(OUT/'main_CPU_preflight.json')['status']=='passed'
    original=ROOT/'runs/sember_adapter_prepare_20261025_01/inputs/execution_lock.json'
    dependencies=read(original)['runtime_sha256'].copy()
    for rel,digest in dependencies.items():
        path=ROOT/rel if not rel.startswith('repo/') or (ROOT/'repo').exists() else ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(path)==digest,rel
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ('.py','.sh','.md'):
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
            dependencies[p.relative_to(ROOT).as_posix()]=sha(p)
    inputs=OUT/'inputs'
    save(inputs/'execution_lock.json',dict(runtime_sha256=dependencies,
        input_sha256={p.name:sha(p) for p in inputs.iterdir() if p.is_file()},
        local_CPU_preflight_SHA=sha(OUT/'main_CPU_preflight.json'),
        validated_development_execution_SHA=sha(original),source_core_changed=False,
        explicit_GPU_authorization_recorded_separately=True))
    print(dict(status='formal_execution_frozen',runtime_files=len(dependencies),input_files=5,
        execution_lock_SHA=sha(inputs/'execution_lock.json')),flush=True)


if __name__=='__main__':main()
