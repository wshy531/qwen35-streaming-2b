import os
from development_worker import R,P,old,read,save
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
env=old.runtime();env['torch'].set_num_threads(1)
builder=old.make_builder(env,env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True))
audit={s['video_id']:s for s in read(P/'samples.json')};pairs=0
for source in read(P/'sources.json'):
    path=R/'datasets/learned-prune-nextqa-v1'/source['mirror_path']
    assert path.resolve().is_relative_to((R/'datasets/learned-prune-nextqa-v1').resolve())
    assert old.sha(path)==source['mirror_lfs_sha256']
    decoded=old.decode(env,path,audit[source['video_id']],builder);pairs+=len(decoded);del decoded
assert pairs==565
save(P/'CPU_media_cluster.json',dict(status='passed',sources=40,real_pairs=565,all_sample_pixel_SHA_verified=True,GPU=0))
print('40 media SHA and 565 real pair pixels verified')
