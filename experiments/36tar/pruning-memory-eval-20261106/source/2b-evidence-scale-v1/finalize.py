"""Bounded one-off CPU collection. Never submits jobs or makes API calls."""
import argparse,json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[2];E=R/'experiments/2b-evidence-scale-v1'
RUNS=[R/'runs/evidence-scale-nextqa-v3-220098',R/'runs/evidence-scale-stream-v5-220117']
OUT=R/'runs/evidence-scale-analysis-20261102-01'
def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-minutes',type=int,default=0);a=p.parse_args();assert 0<=a.wait_minutes<=95
    start=time.monotonic();last=0
    while not all((r/'completed.json').exists() for r in RUNS):
        assert not any((r/'failed.json').exists() for r in RUNS),'worker failure; do not retry'
        assert time.monotonic()-start<a.wait_minutes*60,'bounded wait ended; no new work'
        if time.monotonic()-last>=60:
            print(json.dumps(dict(stage='awaiting_generation',completed=[r.name for r in RUNS if (r/'completed.json').exists()])),flush=True);last=time.monotonic()
        time.sleep(15)
    OUT.mkdir(exist_ok=True)
    def run(script,args):
        print(json.dumps(dict(stage=script)),flush=True)
        subprocess.run([sys.executable,str(E/script),*map(str,args)],cwd=R,check=True)
    run('analyze_v2.py',['--runs',*RUNS,'--out',OUT])
    run('validate_results.py',['--runs',*RUNS,'--out',OUT/'integrity.json'])
    run('report.py',['--analysis',OUT,'--runs',*RUNS])
    # Audit real payloads only. An independent paid authorization is still required.
    run('judge_batch.py',['--inputs',OUT,'--audit'])
    receipt=dict(status='CPU_collection_complete',seconds=time.monotonic()-start,GPU=0,API=0,
        original_runs=[r.relative_to(R).as_posix() for r in RUNS],output=OUT.relative_to(R).as_posix())
    (OUT/'collection_complete.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
