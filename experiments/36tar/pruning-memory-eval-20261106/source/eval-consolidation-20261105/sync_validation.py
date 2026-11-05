"""Install the verified validation pilot in the two approved cluster paths."""
import hashlib
from pathlib import PurePosixPath
import re
import shlex
import subprocess

from fetch_validation import ROOT, OUT, SHORT, STREAM, read, save

REMOTE = '/home/u2025010462/share/czy/swt/my_pape/qwen35-streaming-2b-36tar'


def ssh(command):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                           'ChaoSuan', command], check=True, capture_output=True, text=True, timeout=180).stdout


def transfer(target, files):
    for start in range(0, len(files), 24):
        sources = [str(ROOT / path) for path in files[start:start + 24]]
        subprocess.run(['scp', '-o', 'BatchMode=yes', *sources, f'ChaoSuan:{REMOTE}/{target}/'],
                       check=True, timeout=300)


def main():
    audit = read(OUT / 'audit_summary.json')
    receipt = read(OUT / 'acquisition.json')
    assert receipt['status'] != 'running' and len(receipt['complete']) == audit['verified_media']
    groups = {SHORT: {}, STREAM: {}}
    for row in receipt['complete']:
        path = PurePosixPath(row['media_path'])
        target = path.parent.as_posix()
        assert target in groups and re.fullmatch(r'(sample_\d+_real|\d+)\.mp4', path.name)
        assert path.name == row['name'] and row['media_path'] not in groups[target]
        groups[target][row['media_path']] = row['sha256']
    summaries = ['acquisition.json', 'audit_summary.json', 'short_media_audit.json',
                 'stream_media_audit.json', 'short_validation_questions.json',
                 'stream_main_technical_candidates.json', 'diagnosis_evidence_review_queue.json']
    run = OUT.relative_to(ROOT).as_posix()
    groups[run] = {}
    for name in summaries:
        path = OUT / name
        groups[run][path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    dirs = [shlex.quote(d) for d, files in groups.items() if files]
    assert all((PurePosixPath(REMOTE) / target).as_posix().startswith(REMOTE + '/') for target in groups)
    base = f'cd {shlex.quote(REMOTE)} && '
    checks = ' && '.join(f'test ! -e {d}' for d in dirs)
    ssh(base + checks + ' && mkdir ' + ' '.join(dirs))
    verified = {}
    for target, files in groups.items():
        if not files:
            continue
        transfer(target, list(files))
        args = ' '.join(shlex.quote(path) for path in files)
        lines = ssh(base + 'sha256sum -- ' + args).splitlines()
        hashes = {path: sha for sha, path in (line.split(maxsplit=1) for line in lines)}
        assert hashes == files, 'Remote checksum inventory mismatch'
        verified.update(hashes)
        print(f'{target}: {len(files)} SHA256 verified', flush=True)
    save(OUT / 'cluster_sync.json', dict(status='sha256_verified', remote_root=REMOTE,
        media_count=len(receipt['complete']), manifest_count=len(summaries), files=verified,
        labels_uploaded=False))


if __name__ == '__main__':
    main()
