"""Bind new schedule/worker to previously verified mixed-state runtime."""
from pathlib import Path
import argparse,importlib.metadata,sys
from prepare import ROOT,read,save,digest
HERE=Path(__file__).resolve().parent
EXEC=ROOT/'runs/read_benefit_execution_prepare_20261028_01'
BASE=ROOT/'runs/recent_evidence_prepare_20261028_01/execution_lock.json'
def verify(path):
    lock=read(path)
    assert lock['schema']=='read-benefit-features-v1'
    sys.path.insert(0,str(ROOT/'experiments/2b-recent-evidence-v1'))
    from freeze_evidence import verify as base_verify
    base_verify(BASE)
    for name,sha in lock['sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('changed file '+name)
    for name,version in lock['package_versions'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('dependency drift '+name)
    return lock
def main(output):
    check=read(EXEC/'CPU_schedule_cluster.json');tiny=read(EXEC/'CPU_features_cluster.json');media=read(EXEC/'CPU_media_identity.json')
    assert check['status']=='passed_CPU_feature_schedule' and check['visual_pairs']==453 and check['queries']==120
    assert tiny['status']=='passed_CPU_layout_and_tiny_representation' and tiny['layout_count']==240 and tiny['tiny_extractions']==6
    assert media['status']=='passed_CPU_recent_media_identity' and media['media_files']==18 and media['actual_pixel_smoke_frames']==2
    for name,sha in tiny['source_sha256'].items():
        if digest(HERE/name)!=sha:raise ValueError('tiny-tested source drift '+name)
    files={f for f in HERE.iterdir() if f.suffix in ('.py','.sh','.md')}
    files.update(EXEC/n for n in ('feature_requests.json','media_resolution.json','sources.json','CPU_schedule_cluster.json','CPU_features_cluster.json','CPU_media_identity.json'))
    files.add(BASE)
    lock=dict(schema='read-benefit-features-v1',sha256={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(files)},
        package_versions={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','av','psutil')},
        budget=dict(GPU=1,CPU=4,RAM_GiB=24,minutes=20,formal_feature_prefills=240,warmup_feature_prefills=2,visual_calls=453,loads=1,generated_tokens=0),
        scientific_configuration='original fixed 33-parameter ridge head; no seed/threshold sweep',GPU_authorization_not_created=True)
    save(output,lock);verify(output);print(dict(status='frozen_CPU_verified_not_submitted',files=len(files),budget=lock['budget']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--verify',type=Path);a=p.parse_args()
    if a.verify:verify(a.verify);print('execution lock verified')
    else:main(a.output)
