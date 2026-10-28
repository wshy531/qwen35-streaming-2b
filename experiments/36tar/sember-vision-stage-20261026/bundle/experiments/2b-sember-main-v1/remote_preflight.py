"""Light formal-media receipt/first-pair check; no heavy login-node full replay."""
import os
from main_common import ROOT,OUT,DATA,read,save,sha,verify
from sember_source import CausalSource


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    protocol=verify();videos=read(OUT/'inputs/videos.json');first=read(OUT/'inputs/first_pairs.json')
    receipt=read(DATA/'formal128_transfer_receipt.json');assert receipt['status']=='complete'
    transferred={v['file']:v for v in receipt['new_files']}
    previous=set(receipt['previous_files'])
    assert set(transferred)|previous=={v['relative_path'] for v in videos}
    for v in videos:
        if v['relative_path'] in transferred:assert transferred[v['relative_path']]['sha256']==v['sha256']
        path=(DATA/v['relative_path']).resolve();assert path.is_relative_to(DATA.resolve())
        assert path.stat().st_size==v['bytes']
        if v['relative_path'] in previous:assert sha(path)==v['sha256']
        source=CausalSource(path)
        try:got=[p.record() for p in source.advance(first[v['video_id']]['second']['pts_s'])]
        finally:source.close()
        assert got==[first[v['video_id']]['first'],first[v['video_id']]['second']],v['video_id']
    result=dict(status='passed',runtime_lock_SHA=sha(OUT/'inputs/execution_lock.json'),source_files=128,
        transfer_all_SHA_matches=True,first_real_pair_RGB_PTS_matches_local=128,
        full_video_replay_performed=False,actual_pretrained_loads=0,GPU_jobs=0,
        intended_measured_answers=protocol['measured_answers'])
    save(OUT/'remote_main_preflight.json',result);print(result,flush=True)


if __name__=='__main__':main()
