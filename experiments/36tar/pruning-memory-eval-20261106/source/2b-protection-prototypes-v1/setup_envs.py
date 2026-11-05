import importlib.util,os,sysconfig,venv
from pathlib import Path
root=Path(__file__).resolve().parents[2]
deps=Path(importlib.util.find_spec('torch').origin).resolve().parents[1]
assert deps.is_relative_to(root.resolve())
for name in ('anchor-merge-v1','isolated-recent-v1'):
    dest=root/'envs'/name
    assert not dest.exists(),dest
    venv.EnvBuilder(with_pip=False,symlinks=os.name!='nt').create(dest)
    site=dest/('Lib/site-packages' if os.name=='nt' else f'lib/python{sysconfig.get_python_version()}/site-packages')
    site.mkdir(parents=True,exist_ok=True);(site/'project_runtime.pth').write_text(str(deps)+'\n')
