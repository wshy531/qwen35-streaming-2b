"""Reuse the original input contract and freeze the four prior LS endpoints."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'runs/midprune_memory_prepare_20261102_01'
OUT = ROOT / 'runs/ls_prune_coverage_prepare_20261103_01'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(name, data):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    spec = read(P / 'protocol.json')
    original = ROOT / 'runs/midprune-memory-220237/answers.jsonl'
    assert sha(original) == read(OUT / 'pilot_coverage_audit.json')['sources_sha256']['answers.jsonl']
    rows = [json.loads(x) for x in original.read_text(encoding='utf-8').splitlines()]
    replay = {q['id']: {} for q in read(P / 'questions.json')}
    for r in rows:
        if r['policy'] == 'LS':
            arm = 'full' if r['view'] == 'full' else r['view'] + '_auto864'
            assert arm not in replay[r['id']]
            replay[r['id']][arm] = {k:r[k] for k in ('generated_ids', 'input_ids_sha256', 'position_ids_sha256', 'nodes')}
    assert len(replay) == 18 and all(len(x) == 4 for x in replay.values())
    write('replay.json', replay)
    lock = read(P / 'execution_lock.json')
    for name, expected in lock.items():
        p = ROOT / name
        if name.startswith('repo/') and not p.exists():
            p = ROOT / 'publish/qwen35-streaming-2b' / name[5:]
        assert sha(p) == expected, name
    ck = ROOT / spec['checkpoint']
    assert sha(ck) == spec['checkpoint_sha256']
    paths = [ck, OUT / 'replay.json', OUT / 'CPU_local.json']
    paths.extend(p for p in Path(__file__).parent.iterdir() if p.suffix in ('.py', '.sh'))
    lock.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    write('execution_lock.json', lock)
    print(len(lock), sha(OUT / 'execution_lock.json'))


if __name__ == '__main__':
    main()
