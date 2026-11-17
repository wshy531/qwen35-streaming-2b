import json,hashlib,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/7b_content_residual_prepare_20261113_01';OUT.mkdir(exist_ok=False)
read=lambda p:json.loads((ROOT/p).read_text(encoding='utf8'))
panel=read('runs/training_order_main_prepare_20261104_02/panel_LOCAL_ONLY.json')
g=next(panel['groups'][v] for v in panel['train'] if panel['groups'][v]['role']=='long_train')
plan=dict(purpose='Exact content start, real QA gradient/recompute and initialized optimizer restoration qualification',
    source_id=g['video_id'],media_path=g['path'],media_sha256=g['sha256'],question_id=g['questions'][0]['id'],
    pairs=9,vision=45,language=5,generated_tokens=0,exported_updates=0,local_optimizer_steps=2,
    model_loads=1,GPU='1 A800',CPU=4,memory_GiB=64,minutes=15,
    limitation='Truncated training-side pixels and a training question exercise execution, not answerability or ability; no trained checkpoint exported')
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
files=set(read('runs/7b_trained_prune_completion_prepare_20261113_01/lock.json'))
files.update(read('runs/7b_prune_training_audit_cpu_20261113_02/completed.json')['imported_project_files'])
files={f for f in files if '/7b-prune-training-audit-v1/' not in f}
files.update(str(p.relative_to(ROOT)).replace('\\','/') for p in Path(__file__).parent.iterdir() if p.is_file() and p.suffix in ('.py','.sh'))
files.add(str((OUT/'plan.json').relative_to(ROOT)).replace('\\','/'))
lock={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(files)}
(OUT/'lock.json').write_text(json.dumps(lock,indent=2)+'\n')
bundle=ROOT/'runs/7b_content_residual_upload_20261113_01.tar';assert not bundle.exists()
with tarfile.open(bundle,'w') as t:
    for folder in (Path(__file__).parent,OUT):
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix in ('.py','.sh','.json'):t.add(p,arcname=str(p.relative_to(ROOT)).replace('\\','/'))
print(json.dumps(dict(files=len(files),plan=plan,bundle_sha=hashlib.sha256(bundle.read_bytes()).hexdigest())))
