"""Read-only source/media check on CPU; write only a project-local receipt."""
import argparse, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-learned-prune-main-v1'))
from protocol import runtime,make_builder,decode,read,sha,DATA
PREP=ROOT/'runs/hard_mask_prepare_20261102_01'

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    lock=read(PREP/'execution_lock.json')
    for n,digest in lock.items():
        path=ROOT/n
        if not path.exists() and n.startswith('repo/'):path=ROOT/'publish/qwen35-streaming-2b'/n[5:]
        assert sha(path)==digest,n
    env=runtime();env['torch'].set_num_threads(1)
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    audits={x['video_id']:x for x in read(DATA/'CPU_media_audit.json')['rows']}
    checks=[]
    for row in read(ROOT/'runs/learned_prune_profile_prepare_20261101_01/selection.json'):
        src=row['source'];path=ROOT/'datasets/learned-prune-nextqa-v1'/src['mirror_path']
        assert sha(path)==src['mirror_lfs_sha256']
        pairs=decode(env,path,audits[src['video_id']],builder)
        assert all(batch.video_grid_thw.tolist()==[[1,18,24]] for _,_,batch in pairs)
        checks.append(dict(video_id=src['video_id'],pairs=len(pairs),sha256=src['mirror_lfs_sha256']))
    assert sum(x['pairs'] for x in checks)==44
    out=dict(frozen_files=len(lock),media=checks,real_pairs=44,GPU=0)
    a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out))

if __name__=='__main__':main()
