"""Seal passing CPU checks, prior runtime, and the development profile inputs."""
import ast
from pathlib import Path
from sember_common import ROOT,OUT,read,save,sha


def main():
    names=('core_preflight.json','tiny_session_check.json','native_numeric_comparison.json',
        'boundary_preflight.json','layout_preflight.json','input_preflight_all.json')
    for name in names:assert read(OUT/name)['status']=='passed',name
    old=ROOT/'runs/2b_question_probe_prepare_20261024_01/inputs/execution_lock.json'
    dependencies=read(old)['runtime_sha256'].copy()
    for rel,digest in dependencies.items():
        path=ROOT/rel if not rel.startswith('repo/') or (ROOT/'repo').exists() else ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        assert sha(path)==digest,rel
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.sh','.md'):
            if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
            dependencies[path.relative_to(ROOT).as_posix()]=sha(path)
    inputs=OUT/'inputs';protocol=read(inputs/'profile_protocol.json')
    assert not protocol['GPU_authorized'] and protocol['answers_upper']==108
    save(inputs/'execution_lock.json',dict(
        runtime_sha256=dependencies,input_sha256={p.name:sha(p) for p in inputs.iterdir() if p.is_file()},
        local_CPU_receipts_SHA={name:sha(OUT/name) for name in names},
        previous_runtime_lock_SHA=sha(old),GPU_authorized=False,
        authorization_basis='NONE: separate explicit GPU approval required',
        original_native_control='random CPU tiny only, real full reference uses validated continuation'))
    print(dict(status='frozen',runtime_files=len(dependencies),input_files=len(read(inputs/'execution_lock.json')['input_sha256']),
        execution_lock_SHA=sha(inputs/'execution_lock.json'),GPU_authorized=False),flush=True)


if __name__=='__main__':main()
