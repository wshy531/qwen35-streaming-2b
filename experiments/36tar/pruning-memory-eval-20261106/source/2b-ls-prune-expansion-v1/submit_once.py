"""Verify both preflights and submit the already authorized budget once."""
from pathlib import Path
import json
import os
import subprocess
from datetime import datetime, timezone

from worker import MINUTES, OUT, ROOT, read, sha, verify_lock

SCRIPT = 'experiments/2b-ls-prune-expansion-v1/run_eval.sh'


def write(name, value):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(value, f, indent=2)


def main():
    assert os.name == 'posix' and not os.environ.get('SLURM_JOB_ID')
    verify_lock()
    local = read(OUT/'CPU_local.json')
    assert local == read(OUT/'CPU_cluster.json')
    assert local['passed'] and local['synthetic_layouts']==1416 and local['main_pairs']==26898
    assert read(OUT/'media_cluster.json')['files']==227
    authorization = read(OUT/'GPU_authorization.json')
    assert authorization['approved'] and authorization['lock_sha256']==sha(OUT/'execution_lock.json')
    assert authorization['minutes']==MINUTES and authorization['gpus']==1
    expected = {'--partition=g078t', '--comment=gx_llm', '--gres=gpu:1', '--cpus-per-task=4',
                '--mem=24G', '--time=08:00:00', '--no-requeue',
                '--job-name=q35-ls-prune-expansion',
                '--output=/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/runs/ls-prune-expansion-%j.out'}
    actual = {line.removeprefix('#SBATCH ').strip() for line in (ROOT/SCRIPT).read_text().splitlines()
              if line.startswith('#SBATCH ')}
    assert actual==expected
    subprocess.run(['bash','-n',SCRIPT],cwd=ROOT,check=True)
    env = {k:v for k,v in os.environ.items() if not k.startswith('SBATCH_') and k not in ('SLURM_ACCOUNT','SLURM_QOS')}
    cmd = ['sbatch','--parsable',SCRIPT]
    write('submit_intent.json',dict(cmd=cmd,authorization=authorization,time=datetime.now(timezone.utc).isoformat()))
    result = subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
    write('submission.json',dict(returncode=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip()))
    result.check_returncode()
    print(result.stdout.strip())


if __name__ == '__main__':
    main()
