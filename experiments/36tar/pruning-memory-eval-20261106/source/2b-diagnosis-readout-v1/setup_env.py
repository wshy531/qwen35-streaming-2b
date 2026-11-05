"""Create a separate directory sharing project-owned installed packages."""
import importlib.util
import os
from pathlib import Path
import sysconfig
import venv
root=Path(__file__).resolve().parents[2]
target=root/'envs/diagnosis-readout-v1'
assert not target.exists()
deps=Path(importlib.util.find_spec('torch').origin).resolve().parents[1]
assert deps.is_relative_to(root.resolve())
venv.EnvBuilder(with_pip=False,symlinks=os.name!='nt').create(target)
site=target/('Lib/site-packages' if os.name=='nt' else f'lib/python{sysconfig.get_python_version()}/site-packages')
site.mkdir(exist_ok=True,parents=True)
(site/'project_runtime.pth').write_text(str(deps)+'\n')
print('Independent directory; no package download')
