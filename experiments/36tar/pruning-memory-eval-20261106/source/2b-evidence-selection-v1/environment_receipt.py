"""Read-only package inventory, no torch import or installation."""
from importlib.metadata import version
import json,sys
from prepare import save
from pathlib import Path
packages={k:version(k) for k in ('torch','transformers','numpy','av','Pillow','safetensors','psutil')}
if __name__=='__main__':
    result=dict(packages=packages,python=sys.version.split()[0],platform=sys.platform)
    if len(sys.argv)>1:save(Path(sys.argv[1]),result)
    print(json.dumps(result))
