from audit_common import *
lock=read(ROOT/'runs/content_confirm_prepare_20261101_01/execution_lock.json')
for n,h in lock.items():
    p=ROOT/n
    if not p.exists() and n.startswith('repo/'):p=ROOT/'publish/qwen35-streaming-2b'/n[5:]
    assert sha(p)==h,n
for p in [*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.glob('*.sh'),*Path(__file__).parent.glob('*.md'),
    *[PREP/n for n in ('requests.json','media.json','reference_choices.json','sample_contract.json','protocol.json','labels_LOCAL_ONLY.json')]]:
    lock[p.relative_to(ROOT).as_posix()]=sha(p)
for n in read(PREP/'CPU_local_v2.json')['loaded_source_files']:
    canonical='repo/'+n[len('publish/qwen35-streaming-2b/'):] if n.startswith('publish/qwen35-streaming-2b/') else n
    lock[canonical]=sha(ROOT/n)
save(PREP/'execution_lock.json',lock)
print(json.dumps(dict(files=len(lock),sha256=sha(PREP/'execution_lock.json'))))
