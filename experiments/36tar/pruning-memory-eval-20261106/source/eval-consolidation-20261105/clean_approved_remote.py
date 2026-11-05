"""Delete only the two user-approved frozen remote media lists."""
import hashlib
import json
from pathlib import Path
import stat
import sys

ROOT = Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar')
FOLDER = ROOT / 'runs/dataset_cleanup_20261105_01'
RETIRED_SHA = 'c5da5abc97fd3f9696a01c41c34575c87b8a86752d755050f9efb8bb364328de'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    unused_path = FOLDER / 'unused_short_cleanup_plan.json'
    assert digest(unused_path) == sys.argv[1]
    retired_path = FOLDER / 'retired_media_plan.json'
    assert digest(retired_path) == RETIRED_SHA
    retired = json.loads(retired_path.read_text())
    unused = json.loads(unused_path.read_text())
    assert retired['files_count'] == len(retired['files']) == 2460
    targets = [dict(path=q['path'], bytes=q['bytes'], sha256=q['sha256'], group='retired')
               for q in retired['files']]
    targets += [dict(path=q['cluster_path'], bytes=q['bytes_per_copy'], sha256=q['sha256'], group='unused_short')
                for q in unused['files'] if q['cluster_path']]
    assert len(targets) == len({q['path'] for q in targets}) == 2512
    roots = [ROOT / p for p in ('datasets/EPIC-event-expert-v1', 'datasets/VideoCon-support-v1',
        'datasets/VideoCon-teacher-screen-v1', 'datasets/eval-consolidation-short-validation-v1')]
    receipt = FOLDER / 'deletion_receipt.json'
    assert not receipt.exists()
    for q in targets:
        path = Path(q['path'])
        resolved = path.resolve(strict=True)
        assert any(resolved.is_relative_to(p.resolve(strict=True)) for p in roots)
        assert resolved.is_relative_to(ROOT.resolve())
        assert not path.is_symlink()
        for parent in path.parents:
            if parent == ROOT:
                break
            assert not parent.is_symlink()
        s = path.stat()
        assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_size == q['bytes']
        assert digest(path) == q['sha256'], path
    # Source lists were inspected before approval. No glob or recursive deletion.
    result = dict(status='deleting', authorization='User approved the three frozen cleanup groups',
                  deleted_files=0, deleted_bytes=0, groups={}, total_targets=len(targets))
    def checkpoint():
        tmp = receipt.with_suffix('.writing')
        tmp.write_text(json.dumps(result, indent=2))
        tmp.replace(receipt)
    checkpoint()
    try:
        for q in targets:
            path = Path(q['path'])
            assert path.stat().st_size == q['bytes'] and path.stat().st_nlink == 1 and not path.is_symlink()
            path.unlink()
            result['deleted_files'] += 1
            result['deleted_bytes'] += q['bytes']
            group = result['groups'].setdefault(q['group'], dict(files=0, bytes=0))
            group['files'] += 1
            group['bytes'] += q['bytes']
            if result['deleted_files'] % 100 == 0:
                checkpoint()
        assert all(not Path(q['path']).exists() for q in targets)
        result['status'] = 'complete'
    finally:
        checkpoint()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
