"""CPU layouts, unbounded-builder parity, micro forward, and real media hashes."""
import argparse
import hashlib
import os
from evaluate import *

def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(1)
    torch.manual_seed(20261105)
    env=runtime()
    builder=make_builder(env,env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True))
    config=env['tiny_config'](builder.processor)
    config['vision_config']['depth']=24
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config,attn_implementation='sdpa')).float().eval()
    model.requires_grad_(False)
    stats=dict(layouts=0,maximum_full_context=0,parity_cases=0,micro_QA=0,checkpoint_parameters={},media=0)
    hashed=hashlib.sha256()
    def record(e,p,i,m):
        assert e.shape[1]+128<=65536
        hashed.update(i.numpy().tobytes());hashed.update(p.cpu().numpy().tobytes())
        stats['layouts']+=1
    for panel in ('stream','paired'):
        qs=read(PREP/f'{panel}_questions.json')
        mem={v:Memory(v) for v in ('full','mask_ce')}
        for n in range(20):
            x=torch.randn(108,model.get_input_embeddings().embedding_dim)
            for v,m in mem.items():
                ids=torch.arange(108) if v=='full' else torch.arange(1,108,2)
                m.append(x[ids],ids,n,n*2.,n*2.+1)
        for q in qs:
            prompt=question(q,0,'B')
            for v,m in mem.items():
                for policy in ('LS','M0' if v=='full' else 'M0_same8'):
                    e,p,i,meta=context(model,builder,m.export(policy),prompt,visual_limit=m.banks[policy].budget)
                    assert i.shape[1]+128<=4096
                    record(e,p,i,meta)
            if panel=='stream':
                count=read(ROOT/'runs/protection_streamingbench_prepare_20261103_01/causal_reference.json')['prefixes'][q['id']]['state']['complete_pairs']
                nodes=[sparse_pair(torch.zeros(108,model.get_input_embeddings().embedding_dim),torch.arange(108),n,n*2.,n*2.+1) for n in range(count)]
                e,p,i,meta=full_context(model,builder,nodes,prompt)
                stats['maximum_full_context']=max(stats['maximum_full_context'],i.shape[1]+128)
                record(e,p,i,meta)
        for m in mem.values():m.close()
    # Exact old-builder parity on unmerged history, including >8 pairs.
    q=read(PREP/'stream_questions.json')[0]
    for n in (1,8,20,32):
        nodes=[sparse_pair(torch.randn(108,model.get_input_embeddings().embedding_dim),torch.arange(108),i,2.*i,2.*i+1) for i in range(n)]
        a=full_context(model,builder,nodes,question(q,0,'B'),limit=2)
        b=context(model,builder,nodes,question(q,0,'B'),visual_limit=108*n,context_limit=65536,max_new_tokens=2)
        assert all(torch.equal(x,y) for x,y in zip(a[:3],b[:3]))
        stats['parity_cases']+=1
    # Same selected input through fresh chunked full and bounded QA branches.
    full=Memory('full')
    for i,n in enumerate(nodes[:8]):full.append(n.features,torch.arange(108),i,n.begin,n.end)
    with torch.no_grad(),env['fp32_reference']():
        a=full_answer(env,model,builder,full.export('M0'),question(q,0,'B'),lambda:None,limit=2)
        b=ask(env,model,builder,full,'M0',question(q,0,'B'),limit=2)
        for key in ('input_ids_sha256','position_ids_sha256','generated_ids'):assert a[key]==b[key],key
        stats['micro_QA']+=2
        head=GroupScorer(model.config.vision_config.hidden_size).eval().requires_grad_(False)
        # True processor pixel shape with tiny 24-layer visual model.
        from pilot import Packet
        np=env['np']
        x=Packet(0,0.,np.zeros((288,384,3),dtype=np.uint8))
        y=Packet(1,1.,np.full((288,384,3),127,dtype=np.uint8))
        batch,_=builder.pair(x,y,0,0)
        obs=Observe(model,{'guarded':head,'initial':head,'old_CE':head},'paired')
        obs.append(batch,0.,1.)
        for arm in PAIR_ARMS:
            r=ask_arm(env,model,builder,obs,q,arm,lambda:None,limit=1)
            assert r['branch_released'] and r['read_state_unchanged']
            stats['micro_QA']+=1
    for name,path in CHECKPOINTS.items():
        d=torch.load(ROOT/path,map_location='cpu',weights_only=True)
        assert sum(p.numel() for p in d.values())==69697
        stats['checkpoint_parameters'][name]=sum(p.numel() for p in d.values())
    if args.media:
        for s in read(PREP/'stream_sources.json'):
            path=(ROOT/s['media_path']).resolve()
            assert path.is_relative_to(ROOT.resolve()) and sha(path)==s['media_sha256']
            stats['media']+=1
        spec=read(P/'protocol.json')
        total=0
        for s in read(PREP/'paired_sources.json'):
            path=Path(spec['media_root'])/s['name']
            assert sha(path)==s['media_sha256']
            for first,second in exact_pairs(env,path,s['sampled_trace']):total+=1
            stats['media']+=1
        assert total==429 and stats['media']==77
        stats['paired_real_pixel_pairs']=total
    stats.update(passed=True,layout_sha256=hashed.hexdigest(),GPU=0)
    save(args.out,stats)
    print(json.dumps(stats))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--media',action='store_true')
    main(p.parse_args())
