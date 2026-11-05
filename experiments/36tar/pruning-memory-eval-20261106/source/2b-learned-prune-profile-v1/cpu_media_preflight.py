"""Verify the actual three-video preprocessing path on CPU before GPU approval."""
import os, json
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from gpu_preflight import ROOT,PREP,read,sha,decode,runtime,make_builder


def main():
    lock=read(PREP/'execution_lock.json')
    for name,digest in lock.items(): assert sha(ROOT/name)==digest,name
    env=runtime();env['torch'].set_num_threads(1)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    builder=make_builder(env,processor)
    audit={r['video_id']:r for r in read(ROOT/'runs/learned_prune_data_prepare_20261101_01/CPU_media_audit.json')['rows']}
    count=0
    for row in read(PREP/'selection.json'):
        path=ROOT/'datasets/learned-prune-nextqa-v1'/row['source']['mirror_path']
        assert sha(path)==row['source']['mirror_lfs_sha256']
        pairs=decode(env,path,audit[row['source']['video_id']],builder)
        for _,_,batch in pairs:
            assert batch.video_grid_thw.tolist()==[[1,18,24]]
            assert list(batch.pixel_values_videos.shape)==[432,1536]
        count+=len(pairs)
    assert count==44
    result=dict(status='passed',execution_files=len(lock),real_media=3,real_preprocessed_pairs=count,
                all_sampled_RGB_matches=True,GPU=0)
    (PREP/'CPU_media_cluster.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
