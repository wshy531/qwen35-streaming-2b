"""Freeze own code, complete CPU proof and prior verified mixed-state runtime."""
import argparse
import importlib.metadata
from pathlib import Path
import sys
from gate_common import ROOT,HERE,INPUTS,read,save,sha

SHARED='runs/temporal_router_prepare_20261027_01/execution_lock.json'
def verify(path):
    lock=read(path)
    if lock['schema']!='recent-evidence-execution-v1':raise ValueError('fixed evidence lock required')
    sys.path.insert(0,str(ROOT/'experiments/2b-temporal-router-v1'))
    from freeze_router import verify as old_verify
    old_verify(ROOT/SHARED)
    for name,value in lock['sha256'].items():
        if sha(ROOT/name)!=value:raise ValueError('changed dependency: '+name)
    for name,value in lock['package_versions'].items():
        if importlib.metadata.version(name)!=value:raise ValueError('changed package: '+name)
    return lock

def main(output,cpu,tiny,media,identity):
    proof=read(cpu);smoke=read(tiny);audits=[read(p) for p in media]
    if (proof['status']!='passed_CPU_recent_evidence' or proof['layout_count']!=2400
        or smoke['status']!='passed_CPU_recent_evidence' or smoke['tiny_prefills']!=6):
        raise ValueError('full layouts and six shared tiny prefills required')
    sources=[s for a in audits for s in a['sources']]
    if (len(audits)!=3 or {a['batch'] for a in audits}!={0,1,2}
        or any(a['status']!='passed_CPU_recent_media_batch' for a in audits)
        or sum(a['pairs'] for a in audits)!=453 or len({s['name'] for s in sources})!=18):
        raise ValueError('all fixed media pixels required')
    remote=read(identity)
    if (remote['status']!='passed_CPU_recent_media_identity' or remote['media_files']!=18
        or remote['actual_pixel_smoke_frames']!=2 or remote['complete_remote_redecode_claim'] is not False
        or {s['name']:s['sha256'] for s in remote['sources']}!={s['name']:s['sha256'] for s in sources}):
        raise ValueError('remote exact media identity and pixel smoke required')
    for name,value in proof['source_sha256'].items():
        if sha(ROOT/name)!=value:raise ValueError('CPU-tested source changed: '+name)
    for name,value in proof['input_sha256'].items():
        if sha(INPUTS/name)!=value:raise ValueError('CPU-tested input changed')
    for name,value in smoke['source_sha256'].items():
        if Path(name).name in ('evidence_runtime.py','evidence_policy.py','preflight_evidence.py') and sha(ROOT/name)!=value:
            raise ValueError('tiny-tested forward dependency changed: '+name)
    base=Path(sys.prefix).resolve()
    if base.name!='recent-evidence-v1' or not base.is_relative_to((ROOT/'envs').resolve()):
        raise ValueError('isolated own environment required')
    files={p for p in HERE.iterdir() if p.suffix in ('.py','.md','.sh')}
    files.update(p for p in INPUTS.iterdir() if p.is_file())
    files.update((Path(cpu),Path(tiny),Path(identity),ROOT/SHARED,base/'pyvenv.cfg',base/'bin/python'))
    files.update(Path(p) for p in media)
    files.update(base.glob('lib/python*/site-packages/project_runtime_dependencies.pth'))
    lock=dict(schema='recent-evidence-execution-v1',sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},
        package_versions={p:importlib.metadata.version(p) for p in ('torch','transformers','numpy','psutil','av')},
        budget=read(INPUTS/'protocol.json')['budget'],GPU_authorization_not_created=True)
    save(output,lock);verify(output)
    print(dict(status='frozen_not_authorized',files=len(files),budget=lock['budget']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu',type=Path,required=True);p.add_argument('--media',type=Path,nargs=3,required=True)
    p.add_argument('--tiny-proof',type=Path,required=True)
    p.add_argument('--media-identity',type=Path,required=True)
    a=p.parse_args();main(a.output,a.cpu,a.tiny_proof,a.media,a.media_identity)
