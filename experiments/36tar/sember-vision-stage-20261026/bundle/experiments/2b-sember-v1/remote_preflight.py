"""Light CPU/source check on the login node, never loads a pretrained model."""
import os
from sember_common import ROOT,OUT,DATA,read,save,sha


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    lock=read(OUT/'inputs/execution_lock.json')
    for rel,digest in lock['runtime_sha256'].items():assert sha(ROOT/rel)==digest,rel
    for name,digest in lock['input_sha256'].items():assert sha(OUT/'inputs'/name)==digest,name
    videos=read(OUT/'inputs/videos.json')
    for v in videos:
        path=(DATA/v['relative_path']).resolve();assert path.is_relative_to(DATA.resolve())
        assert path.stat().st_size==v['bytes'] and sha(path)==v['sha256'],v['video_id']
    from preflight_inputs import check_prefix_identity
    got=check_prefix_identity(videos);expected=read(OUT/'inputs/expected_RGB_prefixes.json')
    assert got==expected,'local/cluster actual PTS or RGB mismatch; do not loosen without diagnosis'
    save(OUT/'remote_source_preflight.json',dict(status='passed',runtime_files=len(lock['runtime_sha256']),
        source_files_SHA_verified=len(videos),RGB_PTS_prefixes_match_local=True,
        decoder_modes_checked=['auto4','slice1'],pretrained_loads=0,GPU_jobs=0,
        media_access='read-only exact user-provided partner snapshot',execution_lock_SHA=sha(OUT/'inputs/execution_lock.json')))
    print(dict(status='passed',runtime_files=len(lock['runtime_sha256']),source_files=len(videos),GPU_jobs=0),flush=True)


if __name__=='__main__':main()
