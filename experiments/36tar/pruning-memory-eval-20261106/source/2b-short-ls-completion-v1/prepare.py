"""Freeze existing weights, short questions and all three old answer endpoints."""
import hashlib
import json
import py_compile
import sys
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import run_eval as r
import torch

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    prep=r.PREP
    prep.mkdir(exist_ok=True)
    groups=json.loads((ROOT/'runs/history_utility_short_prepare_20261104_01/short_manifest.json').read_text())
    ids={q['id'] for g in groups for q in g['questions']}
    assert len(ids)==345
    references={}
    sources={'initial':('training-order-main-220483','initial'),
             'old_CE':('training-order-reference-220485','frozen_reference'),
             'guarded':('repair-guard-220499','final')}
    for arm,(run,stage) in sources.items():
        rows=[json.loads(l) for l in (ROOT/'runs'/run/'answers.jsonl').read_text().splitlines()]
        selected=[x for x in rows if x['stage']==stage and x['view']=='full_short']
        assert len(selected)==690 and {x['id'] for x in selected}==ids
        ref={qid:{} for qid in ids}
        for x in selected:
            assert str(x['shift']) not in ref[x['id']]
            ref[x['id']][str(x['shift'])]=dict(choice_original=x['choice_original'],ce=x['ce'])
        assert all(set(v)=={'0','2'} for v in ref.values())
        references[arm]=ref
    r.m.save(prep/'short_manifest.json',groups)
    r.m.save(prep/'references.json',references)
    inherited=json.loads((ROOT/'runs/history_utility_main_prepare_20261104_01/execution_lock.json').read_text())
    lock={}
    for path,digest in inherited.items():
        if not path.startswith('runs/'):
            if (ROOT/path).exists():
                assert sha(ROOT/path)==digest,path
            else:
                assert path.startswith('repo/'),path
            lock[path]=digest
    paths=list(Path(__file__).parent.glob('*'))+[prep/'short_manifest.json',prep/'references.json']
    paths += [ROOT/p for p in r.CHECKPOINTS.values()]
    for path in paths:
        if not path.is_file():continue
        if path.suffix=='.py':py_compile.compile(str(path),doraise=True)
        lock[path.relative_to(ROOT).as_posix()]=sha(path)
    for arm,path in r.CHECKPOINTS.items():
        state=torch.load(ROOT/path,map_location='cpu',weights_only=True)
        assert sum(x.numel() for x in state.values())==69697
        assert all(torch.isfinite(x).all() for x in state.values())
    r.m.save(prep/'execution_lock.json',lock)
    r.m.save(prep/'CPU_local.json',dict(status='passed',reference_rows=2070,questions=345,
        checkpoints=3,parameters_each=69697,locked_files=len(lock),compiled=True))
    send=[p for p in paths if p.is_file() and (p.parent==Path(__file__).parent or p.parent==prep)]
    send += [prep/'execution_lock.json',prep/'CPU_local.json']
    with tarfile.open(prep/'preparation.tar','w') as archive:
        for path in send:archive.add(path,arcname=path.relative_to(ROOT).as_posix())
    print(json.dumps(dict(files=len(lock),references=2070,archive_bytes=(prep/'preparation.tar').stat().st_size)))

if __name__=='__main__':main()
