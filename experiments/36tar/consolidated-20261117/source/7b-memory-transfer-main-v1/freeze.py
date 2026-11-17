from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/7b_formal_execution_20261107_01'
PLAN=ROOT/'runs/7b_formal_plan_20261107_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a,b=[read(RUN/n) for n in ('CPU_local.json','CPU_cluster.json')]
    assert a['passed'] and b['passed'] and b['media_verified']==201
    for k in ('part_checks','layout_count','run_main_sha256'):assert a[k]==b[k]
    paths=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))+list(Path(__file__).parent.glob('*.json'))
    paths+=list(PLAN.glob('part-*.json'))+[PLAN/'plan.json',PLAN/'input_lock.json']
    paths+=[RUN/n for n in ('CPU_local.json','CPU_cluster.json','media_verified.json','authorization.json')]
    paths+=[ROOT/'runs/7b_transfer_prepare_20261107_01/execution_lock.json',ROOT/'runs/7b-memory-profile-221557/answers.jsonl',ROOT/'runs/7b-memory-profile-221557/completed.json',ROOT/'runs/7b-memory-profile-221557/qualification.json']
    with (RUN/'execution_lock.json').open('x') as f:json.dump({p.relative_to(ROOT).as_posix():sha(p) for p in paths},f,indent=2)
    print(json.dumps(dict(files=len(paths),sha256=sha(RUN/'execution_lock.json'))))
if __name__=='__main__':main()
