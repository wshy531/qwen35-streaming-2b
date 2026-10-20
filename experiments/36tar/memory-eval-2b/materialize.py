"""Restore the pinned layout in a new directory; no installs, downloads or GPU work."""
import argparse
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    for flag in ['destination','repo','model','dataset']:parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args()
    dest,repo,model,dataset=[getattr(args,k).resolve() for k in ['destination','repo','model','dataset']]
    if dest.exists() or not dest.parent.is_dir():raise ValueError('new destination under existing parent required')
    if not (repo/'qwen35_local/modeling_qwen3_5.py').is_file() or not (model/'DOWNLOAD_MANIFEST.json').is_file():raise ValueError('existing repository and verified2B model required')
    media=json.loads((HERE/'catalog/media_manifest124.json').read_text(encoding='utf-8'))
    bysha={r['sha256']:r for r in media}
    requests=json.loads((HERE/'bundle/runs/2b-short-event-inputs-20261020-01/requests.json').read_text(encoding='utf-8'))
    for r in requests:
        src=dataset/bysha[r['media_sha256']]['relative_path']
        if not src.is_file() or src.stat().st_size!=bysha[r['media_sha256']]['bytes']:raise ValueError('missing video; run verify_delivery --dataset-root first')
    shutil.copytree(HERE/'bundle',dest)
    (dest/'repo').symlink_to(repo,target_is_directory=True)
    (dest/'models').mkdir();(dest/'models/Qwen3.5-2B').symlink_to(model,target_is_directory=True)
    (dest/'data/fixed-natural-eval-v1').mkdir(parents=True)
    for r in requests:
        dst=dest/r['media_path'];src=(dataset/bysha[r['media_sha256']]['relative_path']).resolve()
        if not dst.parent.resolve().is_relative_to(dest):raise ValueError('unexpected destination')
        if dst.exists():
            if not dst.is_symlink() or dst.resolve()!=src:raise ValueError('unexpected existing video')
        else:dst.symlink_to(src)
    # A previous CPU receipt is a pinned dependency record, not a new success claim.
    # The new stage has no CPU receipt or reusable GPU authorization: rerun its CPU check.
    print(json.dumps(dict(status='materialized_no_model_run',destination=str(dest),videos_linked=67,
                          pretrained_loads=0,GPU_submissions=0),ensure_ascii=False))

if __name__=='__main__':main()
