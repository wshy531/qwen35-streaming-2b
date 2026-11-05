"""Install a manifest-verified project bundle; never overwrite different files."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''): h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True); args = p.parse_args()
    root = args.root.resolve()
    assert root == Path('/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar').resolve()
    prefixes = ['experiments/2b-delayed-readiness-v1/', 'runs/delayed_training_candidates_20261104_01/',
                'datasets/delayed-training-nextqa-v1/']
    with tarfile.open(args.archive, 'r:') as tar:
        manifest = json.load(tar.extractfile('bundle_manifest.json'))
        members = {m.name: m for m in tar.getmembers()}
        assert len(members) == len(tar.getmembers()) == len(manifest) + 1
        assert set(members) == set(manifest) | {'bundle_manifest.json'}
        # Validate EVERY path and existing target before any payload writes.
        for name, entry in manifest.items():
            rel = PurePosixPath(name); member = members[name]
            assert not rel.is_absolute() and '..' not in rel.parts and ':' not in name
            assert member.isfile() and any(name.startswith(v) for v in prefixes)
            target = root / name; assert target.resolve().is_relative_to(root)
            assert member.size == entry['bytes']
            if target.exists(): assert sha(target) == entry['sha256'], 'different existing target: ' + name
        for name, entry in manifest.items():
            target = root / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(members[name]) as src, target.open('xb') as dst:
                    for chunk in iter(lambda: src.read(1024**2), b''): dst.write(chunk)
            assert sha(target) == entry['sha256'], 'installed SHA mismatch: ' + name
    receipt = dict(status='all_SHA_verified', files=len(manifest), bytes=sum(r['bytes'] for r in manifest.values()),
                   archive_sha256=sha(args.archive), entries=manifest)
    out = root / 'runs/delayed_training_candidates_20261104_01/verified_cluster.json'
    assert not out.exists(), 'existing receipt; no implicit reinstall'
    out.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='entries'}))


if __name__ == '__main__': main()
