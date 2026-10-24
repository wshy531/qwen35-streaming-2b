"""Lock checked code and inputs; this grants no GPU authorization."""
import ast
from pathlib import Path
from common import ROOT, OUT, read, save, sha


def main():
    run = OUT/'inputs'; cpu = read(OUT/'CPU_preflight.json')
    assert cpu['status'] == 'passed' and cpu['protocol_sha256'] == sha(run/'protocol.json')
    previous = ROOT/'runs/2b_token_compression_prepare_20261021_01/inputs/execution_lock.json'
    lock = read(previous)['runtime_sha256']
    for name, digest in lock.items():
        path = ROOT/name if not name.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(path) == digest, name
    # Additional same-budget baseline and untouched source snapshot.
    for name in ('experiments/2b-short-event-v1/policies.py','experiments/2b-closed-balance-v1/policy.py'):
        lock[name] = sha(ROOT/name)
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.sh'):
            if path.suffix == '.py': ast.parse(path.read_text(encoding='utf-8'))
            lock[path.relative_to(ROOT).as_posix()] = sha(path)
    save(run/'execution_lock.json',dict(protocol_sha256=sha(run/'protocol.json'),
        input_sha256={p.name:sha(p) for p in run.iterdir() if p.is_file()},runtime_sha256=lock,
        CPU_receipt_sha256=sha(OUT/'CPU_preflight.json'),GPU_authorized=False,
        resources=read(run/'protocol.json')['resources']))
    save(OUT/'analysis_lock.json',dict(labels_sha256=sha(OUT/'labels_LOCAL_ONLY.json'),
        scorer_sha256=sha(Path(__file__).parent/'analyze.py'),
        historical_references_sha256=sha(OUT/'historical_references_LOCAL_ONLY.json'),
        scorer_contract_sha256=sha(OUT/'scorer_contract.json')))
    print(dict(runtime_files=len(lock),execution_lock_sha256=sha(run/'execution_lock.json'),GPU_authorized=False))


if __name__ == '__main__': main()
