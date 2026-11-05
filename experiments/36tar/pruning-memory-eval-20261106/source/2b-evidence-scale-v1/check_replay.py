"""Compare full causal encoding against last-eight plus predecessor restoration."""
import os,json,hashlib,argparse
from pathlib import Path
from common import *
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    env=s.runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261102)
    processor=env['AutoProcessor'].from_pretrained(R/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=24
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval().requires_grad_(False)
    head=GroupScorer(model.config.vision_config.hidden_size).eval().requires_grad_(False)
    from transformers.feature_extraction_utils import BatchFeature
    batches=[BatchFeature(dict(pixel_values_videos=t.randn(432,1536),video_grid_thw=t.tensor([[1,18,24]]))) for _ in range(13)]
    rows=[]
    with t.no_grad(),env['fp32_reference']():
        for arm in ARMS[:-1]:
            state=None;selector=s.CoverageScorer();full=[]
            for batch in batches:
                f,ids,state=encode(model,batch,arm,head,state,selector);full.append((f.clone(),ids.clone()))
            for end in (2,8,9,13):
                first=max(0,end-8);pre=max(0,first-1);state=None;selector=s.CoverageScorer();checked=0
                for i in range(pre,end):
                    f,ids,state=encode(model,batches[i],arm,head,state,selector)
                    if i>=first:
                        assert t.equal(ids,full[i][1]) and t.equal(f,full[i][0]),(arm,end,i)
                        checked+=1
                rows.append(dict(arm=arm,total_pairs=end,kept=checked,ids_equal=True,features_bitwise_equal=True))
    result=dict(status='passed_tiny_exact_replay',checks=rows,pair_comparisons=sum(x['kept'] for x in rows),
        real_GPU_regression_still_required=True,GPU=0)
    save(a.out,result);print(json.dumps(result))
if __name__=='__main__':main()
