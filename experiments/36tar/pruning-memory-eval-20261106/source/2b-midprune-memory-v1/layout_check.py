import argparse,hashlib,json,os
from pilot import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--media',action='store_true');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1';torch.set_num_threads(1);torch.manual_seed(20261102)
    if a.media:
        spec=read(P/'protocol.json');checked=[];pairs=0;env=runtime()
        for s in read(P/'sources.json'):
            path=Path(spec['media_root'])/s['name'];assert path.resolve().parent==Path(spec['media_root']).resolve()
            assert sha(path)==s['media_sha256'];checked.append(s['name'])
            for first,second in exact_pairs(env,path,s['sampled_trace']):
                assert first.rgb.shape==second.rgb.shape==(288,384,3);pairs+=1
        assert pairs==spec['real_pairs']
        save(a.out,dict(passed=True,read_only_media=checked,pixel_verified_pairs=pairs));print('five files SHA and all sampled pixels verified');return
    env=runtime();proc=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](proc),attn_implementation='sdpa')).float().eval()
    builder=make_builder(env,proc);qs=read(P/'questions.json');layouts=0;digest_out=hashlib.sha256();maximum=0
    for s in read(P/'sources.json'):
        mem={v:Memory(v) for v in VIEWS}
        for i in range(0,len(s['sampled_trace']),2):
            first,second=s['sampled_trace'][i:i+2];x=torch.randn(108,model.config.text_config.hidden_size)
            for v,m in mem.items():
                ids=torch.arange(108) if v=='full' else torch.arange(1,108,2)
                m.append(x[ids],ids,i//2,first['pts_s'],second['pts_s'])
            for q in qs:
                if q['source_name']!=s['name'] or q['query_time_s']!=second['pts_s']:continue
                for m in mem.values():
                    for policy,b in m.banks.items():
                        _,pos,ids,meta=context(model,builder,m.export(policy),question(q,0,'B'),visual_limit=b.budget,max_new_tokens=16)
                        assert ids.shape[1]+128<=4096
                        assert all(n['end']<=q['query_time_s'] for n in meta['nodes'])
                        digest_out.update(ids.numpy().tobytes());digest_out.update(pos.numpy().tobytes());layouts+=1;maximum=max(maximum,ids.shape[1])
        for m in mem.values():m.close()
    assert layouts==342
    save(a.out,dict(passed=True,synthetic_feature_layouts=layouts,layout_sha256=digest_out.hexdigest(),maximum_tokens=maximum,output_reserve=128,
        caveat='Real feature-dependent merge topology checked at runtime; these are CPU budget and layout checks, not promised actual input hashes.'))
    print(json.dumps(read(a.out)))
if __name__=='__main__':main()
