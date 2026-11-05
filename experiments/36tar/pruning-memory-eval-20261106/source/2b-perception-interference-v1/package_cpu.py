"""Small owned-source/evidence transfer only; no models, videos or credentials."""
import tarfile
from common import ROOT, HERE, PREP, INPUTS, OLD, read, save, sha


def main():
    manifest = read(INPUTS / 'manifest_LOCAL_ONLY.json')
    paths = [p for p in HERE.iterdir() if p.suffix in ('.py', '.md')]
    paths += [p for p in INPUTS.iterdir() if p.is_file()]
    paths += [PREP / 'independent_review_LOCAL_ONLY.json']
    paths += [ROOT / name for name in manifest['input_origin_sha256']]
    paths += [OLD / 'partner_readonly/policy.py', OLD / 'source_execution_binding.json']
    paths += [ROOT / p['image_relative_path'] for p in manifest['pairs']]
    paths = sorted(set(paths))
    archive = PREP / 'CPU_prepare_bundle_v1.tar.gz'
    if archive.exists():
        raise FileExistsError(archive)
    hashes = {}
    with tarfile.open(archive, 'x:gz') as tar:
        for path in paths:
            assert path.is_file() and path.resolve().is_relative_to(ROOT.resolve())
            assert path.stat().st_size < 2 * 2**20
            name = path.relative_to(ROOT).as_posix()
            hashes[name] = sha(path)
            tar.add(path, arcname=name, recursive=False)
    assert archive.stat().st_size < 3 * 2**20
    save(PREP / 'CPU_prepare_bundle_v1.json', dict(files=hashes,
        archive_sha256=sha(archive), bytes=archive.stat().st_size, pretrained_models=0, videos=0))
    print(dict(archive=str(archive), bytes=archive.stat().st_size, files=len(hashes)))


if __name__ == '__main__':
    main()
