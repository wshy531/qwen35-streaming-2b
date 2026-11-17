"""Install a hash-verified official wheel in an isolated environment, CPU only."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import venv

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT/'runs/7b_transfer_prepare_20261107_01'
def main(version):
    VERSION = version
    environment = ROOT/f'envs/qwen25-vl-7b-native-{VERSION}'
    assert not environment.exists(), 'Do not replace an existing environment'
    deps = PREP/'official_package'
    deps.mkdir(exist_ok=True)
    with urllib.request.urlopen(f'https://pypi.org/pypi/transformers/{VERSION}/json', timeout=30) as response:
        meta = json.load(response)
    candidates = [r for r in meta['urls'] if r['filename'].endswith('-py3-none-any.whl')]
    assert len(candidates) == 1
    item = candidates[0]
    assert item['size'] <= 20*2**20 and item['url'].startswith('https://files.pythonhosted.org/')
    wheel = deps/item['filename']
    with urllib.request.urlopen(item['url'], timeout=60) as response, wheel.open('xb') as f:
        total=0
        while data := response.read(1024*1024):
            total += len(data)
            assert total <= item['size']
            f.write(data)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert total == item['size'] and digest == item['digests']['sha256']
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    suffix = 'Lib/site-packages' if sys.platform == 'win32' else f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    site = environment/suffix
    shared = Path(sys.prefix)/suffix
    (site/'shared_fixed_dependencies.pth').write_text(str(shared)+'\n',encoding='utf-8')
    executable = environment/('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    subprocess.run([str(executable),'-m','pip','install','--no-index','--no-deps','--no-cache-dir',str(wheel)],check=True)
    identity = json.loads(subprocess.check_output([str(executable),'-c',
        'import transformers,json; print(json.dumps(dict(version=transformers.__version__,path=transformers.__file__)))'],text=True))
    assert identity['version']==VERSION and Path(identity['path']).resolve().is_relative_to(environment.resolve())
    (PREP/f'official_environment_{VERSION}.json').write_text(json.dumps(dict(identity=identity,
        wheel_sha256=digest,wheel_bytes=total,wheel_url=item['url'],
        shared_dependencies=str(shared),original_environment_changed=False),indent=2)+'\n')
    print(json.dumps(identity),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--version',choices=['5.2.0','5.5.4'],required=True)
    main(parser.parse_args().version)
