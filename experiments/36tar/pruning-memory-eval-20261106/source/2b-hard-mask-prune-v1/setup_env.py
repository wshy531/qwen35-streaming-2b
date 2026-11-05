"""Separate interpreter; reuse already installed project packages without downloads."""
import importlib.util, os, sysconfig, venv
from pathlib import Path
root = Path(__file__).resolve().parents[2]
dest = root / 'envs/hard-mask-prune-v1'
assert not dest.exists(), 'preserve existing environment'
deps = Path(importlib.util.find_spec('torch').origin).resolve().parents[1]
assert deps.is_relative_to(root.resolve())
venv.EnvBuilder(with_pip=False, symlinks=os.name != 'nt').create(dest)
site = dest / ('Lib/site-packages' if os.name == 'nt' else f'lib/python{sysconfig.get_python_version()}/site-packages')
site.mkdir(parents=True, exist_ok=True)
(site / 'project_runtime.pth').write_text(str(deps) + '\n')
