"""Create this experiment's venv using existing, read-only dependencies."""
from pathlib import Path
import os,sysconfig,venv
root=Path(__file__).resolve().parents[2]
target=root/'envs/evidence-selection-v1'
if target.exists():raise FileExistsError('do not overwrite an environment')
source=Path(sysconfig.get_paths()['purelib']).resolve()
if not source.is_relative_to(root.resolve()):raise ValueError('project dependency environment only')
venv.EnvBuilder(with_pip=False,symlinks=os.name!='nt').create(target)
dest=target/'Lib/site-packages' if os.name=='nt' else target/'lib'/f'python{sysconfig.get_python_version()}'/'site-packages'
dest.mkdir(parents=True,exist_ok=True)
(dest/'project_runtime_dependencies.pth').write_text(str(source)+'\n',encoding='utf-8')
print('Isolated environment created; no install/download or old environment modification.')
