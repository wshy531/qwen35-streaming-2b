from adapt import *
verify_adapt()
groups=read(HERE/'fit_sources.json');tasks=read(HERE/'questions.json')
qs=[q for g in groups for q in g['questions']]+tasks
paths={str(path(q)):q['media_sha256'] for q in qs}
for value in paths:
    p=Path(value);assert p.is_file() and p.stat().st_size>0
    assert str(p.resolve()).startswith('/home/share/czy/swt/my_pape/qwen35-streaming-2b-36tar/')
assert (ROOT/'envs/qwen25-vl-7b-native-5.2.0/bin/python').is_file()
head=load_head();assert sum(p.numel() for p in head.parameters())==713537
save(HERE/'remote_check.json',dict(passed=True,media_paths=len(paths),
    media_hashes='frozen existing manifests; runtime will hash every source before use',
    lock_sha=sha(HERE/'lock.json')))
print(json.dumps(dict(passed=True,media_paths=len(paths))))
