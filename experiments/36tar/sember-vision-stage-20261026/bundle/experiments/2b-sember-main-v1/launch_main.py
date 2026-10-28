"""Two sequential workers under one approved allocation, with a shared deadline."""
import os
from pathlib import Path
import subprocess
import sys
import time
import json


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('approved Slurm allocation required')
    root=Path(__file__).resolve().parents[2]
    protocol=json.loads((root/'runs/sember_main_prepare_20261025_01/inputs/protocol.json').read_text())
    out=root/'runs'/('2b-sember-main-'+os.environ['SLURM_JOB_ID']);out.mkdir(exist_ok=False)
    deadline=time.time()+protocol['resource_budget']['minutes']*60-protocol['stop_before_allocation_end_seconds']
    for mode in ('bounded','reference'):
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_main.py')),'--mode',mode,
            '--out',str(out/mode),'--deadline',str(deadline)],cwd=root,check=True)


if __name__=='__main__':main()
