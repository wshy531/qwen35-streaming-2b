"""Freeze this exact implementation after both CPU contracts pass."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_transfer_prepare_20261107_01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=json.loads((PREP/'CPU_local.json').read_text());b=json.loads((PREP/'CPU_cluster.json').read_text())
    assert a['passed'] and b['passed'] and a['layout_count']==b['layout_count']==1140
    assert a['layout_sha256']==b['layout_sha256']
    assert len(a['micro'])==len(b['micro'])==3
    for left,right in zip(a['micro'],b['micro']):
        for key in ('key','pixel_exact','position_exact','pixel_sha256','ids','arms'):
            assert left[key]==right[key],key
        assert max(left['native_stream_max_abs'],right['native_stream_max_abs'])<=1e-6
    files=[p for p in (ROOT/'experiments/7b-memory-transfer-v1').iterdir() if p.suffix in ('.py','.md','.json','.sh')]
    files += [PREP/n for n in ['config.json','profile.json','questions.json','traces.json',
        'CPU_local.json','CPU_cluster.json','asset_manifest.json','decoded_cpu.json']]
    files += list(PREP.glob('cpu_frames_*.npz'))
    files += [ROOT/'experiments/2b-temporal-router-v1/temporal_policy.py',ROOT/'experiments/2b-evidence-eval-v1/response.py']
    values={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    with (PREP/'execution_lock.json').open('x') as f:json.dump(values,f,indent=2)
    print(json.dumps(dict(files=len(values),lock_sha256=sha(PREP/'execution_lock.json'))))
if __name__=='__main__':main()
