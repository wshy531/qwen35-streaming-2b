"""Pin one unmodified PyPI release in a separate environment, sharing heavy libraries."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import venv
from audit_common import ROOT, OUT, save

VERSION = '5.4.0'


def main():
    metadata_url = f'https://pypi.org/pypi/transformers/{VERSION}/json'
    with urllib.request.urlopen(metadata_url, timeout=30) as response:
        meta = json.load(response)
    files = [v for v in meta['urls'] if v['filename'].endswith('-py3-none-any.whl')]
    assert len(files) == 1
    item = files[0]
    assert item['size'] <= 20*2**20 and item['url'].startswith('https://files.pythonhosted.org/')
    deps = OUT / 'official_package'
    deps.mkdir(parents=True, exist_ok=True)
    wheel = deps / item['filename']
    if not wheel.exists():
        with urllib.request.urlopen(item['url'], timeout=60) as response, wheel.open('xb') as stream:
            total = 0
            while block := response.read(1024*1024):
                total += len(block)
                assert total <= item['size']
                stream.write(block)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert wheel.stat().st_size == item['size'] and digest == item['digests']['sha256']
    environment = ROOT / 'envs/native-audit-transformers-5.4.0'
    assert not environment.exists(), 'never overwrite an existing environment'
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    executable = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    site = environment / ('Lib/site-packages' if sys.platform == 'win32' else f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
    # Explicitly share fixed heavy Torch/etc, while the pinned package takes priority.
    original_site = Path(sys.prefix) / ('Lib/site-packages' if sys.platform == 'win32' else f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
    (site / 'shared_fixed_dependencies.pth').write_text(str(original_site)+'\n', encoding='utf-8')
    subprocess.run([str(executable), '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir', str(wheel)], check=True)
    identity = subprocess.check_output([str(executable), '-c',
        'import transformers,json; print(json.dumps(dict(version=transformers.__version__,path=transformers.__file__)))'], text=True)
    identity = json.loads(identity)
    assert identity['version'] == VERSION and Path(identity['path']).resolve().is_relative_to(environment.resolve())
    result = dict(status='isolated_official_release_installed', version=VERSION,
        metadata_url=metadata_url, wheel_url=item['url'], wheel_bytes=item['size'], wheel_sha256=digest,
        environment=str(environment), identity=identity, heavy_dependencies_shared_from=str(original_site),
        original_environment_modified=False, real_weights_downloaded=False, GPU_jobs=0)
    save(OUT/'official_environment.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
