"""Independent environment referencing project dependencies, no installation."""
from pathlib import Path
import os,sysconfig,venv
root=Path(__file__).resolve().parents[2];target=root/'envs/evidence-selector-train-v1'
if target.exists():raise FileExistsError('do not overwrite environment')
deps=Path(sysconfig.get_paths()['purelib']).resolve()
if not deps.is_relative_to(root.resolve()):raise ValueError('project dependency path required')
venv.EnvBuilder(with_pip=False,symlinks=os.name!='nt').create(target)
dest=target/'Lib/site-packages' if os.name=='nt' else target/'lib'/f'python{sysconfig.get_python_version()}'/'site-packages'
dest.mkdir(parents=True,exist_ok=True);(dest/'project_runtime_dependencies.pth').write_text(str(deps)+'\n',encoding='utf-8')
print('Independent selector environment created, dependencies unchanged.')
