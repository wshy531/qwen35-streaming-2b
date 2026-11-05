"""Sequential independent processes; failure stops the fixed experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT/'runs/stream_prune_cost_prepare_20261102_01'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and args.out.resolve().is_relative_to((ROOT/'runs').resolve())
    args.out.mkdir(exist_ok=False)
    started = time.monotonic()
    jobs = json.loads((PREP/'jobs.json').read_text(encoding='utf-8'))
    try:
        for index, job in enumerate(jobs):
            remaining = 88*60-(time.monotonic()-started)
            assert remaining > 60, 'Insufficient remaining allocation'
            out = args.out/f'block{job["block"]}_{job["arm"]}'
            command = [sys.executable, str(Path(__file__).parent/'worker.py'), '--job', str(index), '--out', str(out), '--seconds', str(remaining-15)]
            with (args.out/f'worker{index}.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=remaining, check=True)
            assert (out/'completed.json').exists()
            assert sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file()) < 64*2**20
            print(json.dumps(dict(workers_completed=index+1, seconds=time.monotonic()-started)), flush=True)
        (args.out/'completed.json').write_text(json.dumps(dict(status='complete', workers=len(jobs), seconds=time.monotonic()-started)), encoding='utf-8')
    except Exception as error:
        (args.out/'failed.json').write_text(json.dumps(dict(error=repr(error), seconds=time.monotonic()-started)), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
