"""Freeze tested source and inputs. This file cannot confer GPU authorization."""
import ast
from pathlib import Path
from common import ROOT, OUT, read, save, sha


def main():
    cpu = read(OUT/'CPU_preflight.json');tiny = read(OUT/'tiny_session_check.json')
    assert cpu['status'] == tiny['status'] == 'passed' and len(cpu['layout_rows']) == 909
    assert tiny['answers'] == 5 and tiny['probes'] == 1
    old_lock = ROOT/'runs/2b_readout_compression_prepare_20261024_01/inputs/execution_lock.json'
    dependencies = read(old_lock)['runtime_sha256'].copy()
    for rel, digest in dependencies.items():
        path = ROOT/rel if not rel.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/rel[5:]
        if (ROOT/'repo').exists():path = ROOT/rel
        assert sha(path) == digest, rel
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py', '.sh', '.md'):
            if path.suffix == '.py':ast.parse(path.read_text(encoding='utf-8'))
            dependencies[path.relative_to(ROOT).as_posix()] = sha(path)
    inputs = OUT/'inputs'
    assert read(inputs/'protocol.json')['approval']['state'] == 'not_requested_yet'
    save(inputs/'execution_lock.json', dict(protocol_sha256=sha(inputs/'protocol.json'),
        input_sha256={p.name: sha(p) for p in inputs.iterdir() if p.is_file()},
        runtime_sha256=dependencies, local_CPU_sha256=sha(OUT/'CPU_preflight.json'),
        local_tiny_sha256=sha(OUT/'tiny_session_check.json'),
        old_execution_lock_sha256=sha(old_lock), authorization_basis='NONE; separate explicit GPU approval required'))
    save(OUT/'analysis_lock.json', dict(scorer=sha(Path(__file__).with_name('analyze.py')),
        audit_scorer=sha(Path(__file__).with_name('audit_and_decide.py')),
        labels=sha(OUT/'labels_LOCAL_ONLY.json'), references=sha(OUT/'historical_references_LOCAL_ONLY.json')))
    print(dict(status='frozen', runtime_files=len(dependencies),
               execution_lock_sha256=sha(inputs/'execution_lock.json'), GPU_authorized=False))


if __name__ == '__main__':main()
