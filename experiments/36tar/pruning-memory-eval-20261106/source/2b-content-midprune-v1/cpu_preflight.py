"""Real vision modules with random tiny weights: correctness only, never speed/QA claims."""
import os,sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/2b-sember-v1'))
from sember_common import runtime
from content_prune import encode,CoverageScorer

def main(output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    env=runtime();t=env['torch'];t.set_num_threads(1);t.manual_seed(20261101)
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    cfg=env['tiny_config'](processor);cfg['vision_config']['depth']=4
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg,attn_implementation='sdpa')).float().eval()
    visual=model.model.visual;grid=t.tensor([[1,18,24]])
    original={k:v.clone() for k,v in visual.state_dict().items()}
    scores={m:CoverageScorer() for m in ('post_content','mid_content')};rows=[]
    with t.inference_mode(),env['fp32_reference']():
        for pair in range(3):
            pixels=t.randn(432,1536)
            ref=visual(pixels,grid_thw=grid).pooler_output
            full,_,_=encode(visual,pixels,grid,mode='full')
            assert t.equal(full,ref)
            values={}
            for mode in ('post_fixed','mid_fixed','post_content','mid_content'):
                x,ids,trace=encode(visual,pixels,grid,mode=mode,scorer=scores.get(mode),return_trace=True)
                assert x.shape==(54,32) and len(ids.unique())==54 and t.all(ids[1:]>ids[:-1]) and t.isfinite(x).all()
                assert trace['layer_tokens']==([432]*4 if mode.startswith('post') else [432]*2+[216]*2)
                if mode.startswith('post'):assert t.equal(x,ref[ids])
                values[mode]=(x,ids)
                if mode.endswith('content'):
                    tiles={(int(i)//12//2,int(i)%12//2) for i in ids}
                    assert len(tiles)==30
                rows.append(dict(pair=pair,mode=mode,**trace))
            assert t.equal(values['post_content'][1],values['mid_content'][1])
            assert t.equal(values['post_fixed'][1],values['mid_fixed'][1])
        # Reset prevents a preceding source from changing a new source's first selection.
        scores['mid_content'].reset();a=encode(visual,pixels,grid,mode='mid_content',scorer=scores['mid_content'])
        fresh=encode(visual,pixels,grid,mode='mid_content',scorer=CoverageScorer())
        assert all(t.equal(x,y) for x,y in zip(a[:2],fresh[:2]))
        # Flat scores have deterministic lower-ID ties and exact coverage/budget.
        flat=CoverageScorer();ids=flat.choose(t.zeros(432,16),(9,12))
        assert len(ids)==54 and flat.resident_bytes()==108*16*4
        for _ in range(20):assert t.equal(flat.choose(t.zeros(432,16),(9,12)),ids)
        assert flat.resident_bytes()==108*16*4
        assert all(t.equal(v,original[k]) for k,v in visual.state_dict().items())
    result=dict(status='passed',cases=rows,all_kept_exact=True,post_selection_exact=True,
        matched_middle_layer_selections=True,coverage_tiles=30,whole_group_original_positions=True,
        reset_exact=True,constant_input_ties_deterministic=True,bounded_previous_groups=True,
        unchanged_weights=True,pretrained_loads=0,GPU=0,new_answers=0,capability_or_speed_claim=False)
    with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
