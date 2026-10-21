"""Restore archived text into a NEW offline workspace; never execute it."""
import argparse
import gzip
from pathlib import Path
import shutil
from verify_delivery import HERE, Package


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists() or not destination.parent.is_dir():
        raise ValueError('new destination under existing parent required')
    package = Package()
    package.check_integrity()
    previous = HERE.parent / 'memory-eval-2b/bundle'
    shutil.copytree(previous, destination, ignore=shutil.ignore_patterns('GPU_authorization.json', '__pycache__'))
    for relative, item in package.manifest['files'].items():
        if not relative.startswith('bundle/'):
            continue
        target = destination / relative[len('bundle/'):]
        if not target.resolve().is_relative_to(destination):
            raise ValueError('unexpected archive path')
        target.parent.mkdir(parents=True, exist_ok=True)
        source = HERE / item['stored_path']
        opener = gzip.open if item['gzip'] else open
        with opener(source, 'rb') as src, target.open('wb') as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
    print('Restored offline text workspace. No media/models/authorization installed; no experiments started.')


if __name__ == '__main__':
    main()
