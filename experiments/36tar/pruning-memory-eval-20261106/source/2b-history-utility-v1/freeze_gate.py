"""Build a new additive lock, preserving every inherited runtime digest."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'runs/history_utility_prepare_20261104_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    inherited=json.loads((ROOT/'runs/training_order_main_prepare_20261104_02/execution_lock.json').read_text(encoding='utf-8'))
    absent=[]
    for p,h in inherited.items():
        if (ROOT/p).exists():assert sha(ROOT/p)==h,p
        else:absent.append(p) # Cluster-only inherited sources remain mandatory there.
    extra=[p for p in (ROOT/'experiments/2b-history-utility-v1').iterdir() if p.is_file() and p.suffix in ('.py','.md','.sh')]
    extra += [P/'train_manifest_LOCAL_ONLY.json',P/'development_manifest_LOCAL_ONLY.json',P/'data_contract.json']
    # Imports not necessarily used by the earlier pruned replay path.
    extra += [ROOT/'experiments/2b-resolution-cost-v1'/n for n in ('resolution_banks_base.py','resolution_common.py','resolution_nodes.py','resolution_runtime.py')]
    lock=dict(inherited)
    for p in extra:lock[p.relative_to(ROOT).as_posix()]=sha(p)
    (P/'gate_lock.json').write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    with tarfile.open(P/'gate-preparation.tar','w') as tar:
        for p in extra+[P/'gate_lock.json']:tar.add(p,arcname=p.relative_to(ROOT).as_posix())
    print(json.dumps(dict(files=len(lock),cluster_only_inherited=len(absent),package_bytes=(P/'gate-preparation.tar').stat().st_size)))
if __name__=='__main__':main()
