"""Create an independent environment with the verified local dependency set."""
import importlib.util
import os
from pathlib import Path
import sysconfig
import venv

root = Path(__file__).resolve().parents[2]
dest = root / 'envs/memory-prune-panel150-v1'
assert not dest.exists()
deps = Path(importlib.util.find_spec('torch').origin).resolve().parents[1]
assert deps.is_relative_to(root.resolve())
venv.EnvBuilder(with_pip=False, symlinks=os.name != 'nt').create(dest)
site = dest / ('Lib/site-packages' if os.name == 'nt' else f'lib/python{sysconfig.get_python_version()}/site-packages')
site.mkdir(parents=True, exist_ok=True)
(site/'project_runtime.pth').write_text(str(deps)+'\n')
