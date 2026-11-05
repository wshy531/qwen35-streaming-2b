"""CPU layouts, tiny real forward and frozen media replay; no real 2B weights."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import sys
from diagnosis_runtime import *

def main(args):
    torch.set_num_threads(1);torch.manual_seed(20261105)
    env=runtime()
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    config=env['tiny_config'](builder.processor)
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config,attn_implementation='sdpa')).float().eval().requires_grad_(False)
    hashed=hashlib.sha256();pixels=hashlib.sha256();count=0;maximum=0;first_nodes=None
    qs=read(PREP/'questions.json');sources=read(PREP/'sources.json')
    for source in sources:
        nodes={i:sparse_pair(torch.zeros(108,model.get_input_embeddings().embedding_dim),torch.arange(108),i,
                    source['pairs'][i][0]['pts'],source['pairs'][i][1]['pts']) for i in source['selected_pair_indices']}
        for q in [q for q in qs if q['source_id']==source['source_id']]:
            for arm in ARMS:
                selected=choose(source,nodes,arm)
                emb,pos,ids,meta=layout(model,builder,selected,question(q,0,'B'))
                assert emb.shape[1]==ids.shape[1] and ids.shape[1]+128<=4096
                assert all(n.end<=q['query_time_s'] for n in selected)
                hashed.update(ids.cpu().numpy().tobytes());hashed.update(pos.cpu().numpy().tobytes())
                count+=1;maximum=max(maximum,ids.shape[1]+128)
        if first_nodes is None:first_nodes=(source,nodes)
    source,nodes=first_nodes; q=next(q for q in qs if q['source_id']==source['source_id'])
    with torch.no_grad(),env['fp32_reference']():
        for arm in ARMS:
            result=answer(env,model,builder,choose(source,nodes,arm),question(q,0,'B'),limit=1)
            assert result['branch_released'] and result['read_state_unchanged']
    media=pairs=0
    if args.media:
        for source in sources:
            path=ROOT/source['media_path'];assert sha(path)==source['media_sha256']
            for index,a,b in selected_pairs(env,path,source):
                batch,_=builder.pair(a,b,0,0)
                pixels.update(batch.pixel_values_videos.numpy().tobytes())
                pixels.update(batch.video_grid_thw.numpy().tobytes())
                pairs+=1
            media+=1
        assert pairs==159 and media==11
    result=dict(passed=True,layouts=count,layout_sha256=hashed.hexdigest(),maximum_context=maximum,
        tiny_answers=4,media=media,selected_pairs=pairs,pixel_sha256=pixels.hexdigest() if args.media else None,
        GPU=0,real_base_loaded=False)
    assert count==88
    closure={}
    for m in list(sys.modules.values()):
        name=getattr(m,'__file__',None)
        if not name:continue
        path=Path(name).resolve()
        if path.is_file() and path.is_relative_to(ROOT.resolve()) and path.suffix=='.py' and not path.is_relative_to((ROOT/'envs').resolve()):
            closure[path.relative_to(ROOT.resolve()).as_posix()]=sha(path)
    save(args.out.with_suffix('.imports.json'),closure)
    save(args.out,result)
    print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--media',action='store_true');main(p.parse_args())
