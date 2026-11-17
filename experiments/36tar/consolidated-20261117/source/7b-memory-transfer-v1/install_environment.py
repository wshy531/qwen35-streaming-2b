"""Offline cluster install of the already verified official 5.2.0 wheel."""
import hashlib,json,subprocess,sys,venv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_transfer_prepare_20261107_01'
def main():
    assert sys.platform=='linux'
    env=ROOT/'envs/qwen25-vl-7b-native-5.2.0'
    assert not env.exists(), 'never replace an environment'
    wheel=PREP/'official_package/transformers-5.2.0-py3-none-any.whl'
    assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='9ecaf243dc45bee11a7d93f8caf03746accc0cb069181bbf4ad8566c53e854b4'
    venv.EnvBuilder(with_pip=True,system_site_packages=True).create(env)
    suffix=f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    (env/suffix/'shared_fixed_dependencies.pth').write_text(str(Path(sys.prefix)/suffix)+'\n')
    python=env/'bin/python'
    subprocess.run([str(python),'-m','pip','install','--no-index','--no-deps','--no-cache-dir',str(wheel)],check=True)
    identity=json.loads(subprocess.check_output([str(python),'-c','import transformers,json; print(json.dumps(dict(version=transformers.__version__,path=transformers.__file__)))'],text=True))
    assert identity['version']=='5.2.0' and Path(identity['path']).resolve().is_relative_to(env.resolve())
    (PREP/'official_environment_cluster.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(identity),flush=True)
if __name__=='__main__':main()
