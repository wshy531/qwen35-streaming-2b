"""New environment referring only to existing owned runtime packages; no install."""
import importlib.util,os,sysconfig,venv
from pathlib import Path
root=Path(__file__).resolve().parents[2];target=root/'envs/evidence-selector-eval-v1'
if target.exists():raise FileExistsError('Do not overwrite environment')
deps=Path(importlib.util.find_spec('torch').origin).resolve().parents[1]
if not deps.is_relative_to(root.resolve()):raise ValueError('Only owned existing dependencies')
venv.EnvBuilder(with_pip=False,symlinks=os.name!='nt').create(target)
dest=target/'Lib/site-packages' if os.name=='nt' else target/'lib'/f'python{sysconfig.get_python_version()}'/'site-packages'
dest.mkdir(parents=True,exist_ok=True);(dest/'project_runtime_dependencies.pth').write_text(str(deps)+'\n',encoding='utf-8')
print('Created independent environment; installed/downloaded packages: 0')
