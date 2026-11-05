"""All909 layouts and real RGB/tiny-model isolation. No pretrained weights/GPU."""
import hashlib
import os
import time
import torch

from common import ROOT, OUT, ARMS, read, save, sha, runtime
from memory import Archive, context


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('explicit CPU-only environment required')
    selection_receipt = check_selection()
    started = time.monotonic()
    env = runtime()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    assert not torch.cuda.is_available()
    build = env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(
        ROOT/'models/Qwen3.5-2B', local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](
        **env['tiny_config'](build.processor),attn_implementation='sdpa')).float().eval()
    groups = read(OUT/'inputs/groups.json')
    rows = read(OUT/'inputs/questions.json')
    contract_cases=selection_receipt['contract_cases']
    layouts = []
    from readout import context as old_context, Limits
    from score_memory_eval import prompt
    with torch.inference_mode(),env['fp32_reference']():
        for g in groups:
            archive = Archive()
            items = sorted([q for q in rows if q['group']==g['id']],key=lambda q:(q['pairs'],q['query_time_s'],q['id']))
            for q in items:
                while len(archive.nodes['full48']) < q['pairs']:
                    i = len(archive.nodes['full48'])
                    first, second = g['frames'][i*2],g['frames'][min(i*2+1,len(g['frames'])-1)]
                    archive.append(torch.randn(48,32),i,first['pts_s'],second['pts_s'])
                assert archive.nodes['full48'][-1].end<=q['query_time_s']+1e-9
                hash_before = archive.fingerprint()
                selector_before=archive.selector_digest()
                text = prompt(q)
                previous = None
                for arm in ARMS:
                    emb,pos,ids,meta = context(model,build,archive.nodes[arm],text)
                    assert meta['visual_tokens']==q['pairs']*(48 if arm=='full48' else 24)
                    assert meta['visual_tokens']==q['pairs']*24
                    assert meta['labels']==[f'<{n.center:.1f} seconds>' for n in archive.nodes['full48']]
                    layouts.append(dict(id=q['id'],cohort=q['cohort'],arm=arm,input_tokens=meta['input_tokens'],
                        visual_tokens=meta['visual_tokens'],suffix_tokens=meta['suffix_tokens']))
                    del emb,pos,ids
                assert hash_before==archive.fingerprint() and selector_before==archive.selector_digest()
            archive.clear()
            assert time.monotonic()-started<240,'bounded CPU layout preparation'
    assert len(layouts)==909
    # Actual decoded pixels enter a random tiny model; question branches then
    # release their caches, and the same main observation accepts the next pair.
    from compression_observer import CompressionObserver
    from replay import VideoSource
    from first_2b_baseline import tensor_digest
    long_group=next(g for g in groups if g['cohort']=='stream34')
    spec=dict(frames=long_group['frames'],media_sha256=long_group['sha256'])
    media_key='cluster_media' if (ROOT/'repo').exists() else 'local_media'
    source=VideoSource(ROOT/long_group[media_key],spec,env['FramePacket'])
    packets=[];pending=[]
    it=iter(source.frames())
    for _ in range(20):
        frame=next(it);pending.append(frame.frame_id)
        if len(pending)==2:
            packets.extend(source.take_pair(pending));pending=[]
    it.close()
    observer=CompressionObserver(env,model,build,main_tokens=384)
    answers=[]
    with torch.inference_mode(),env['fp32_reference']():
        for i in range(0,16,2):observer.append_pair(*packets[i:i+2])
        assert observer.core.resets>=1 and observer.core.vision_encodings==8
        before=tensor_digest(observer.core.main.cache)
        for arm in ARMS:
            tokens=[];answer=observer.ask_arm(arm,'What is visible?',tokens.append)
            assert answer['generated_ids']==[t['token_id'] for t in tokens]
            assert before==tensor_digest(observer.core.main.cache)
            assert observer.core.vision_encodings==8
            answers.append(dict(arm=arm,generated_tokens=len(tokens),main_and_archive_unchanged=True))
        observer.append_pair(*packets[16:18])
        assert observer.core.vision_encodings==9 and observer.core.pairs==9
        state=tensor_digest(observer.core.main.cache)
        observer.snapshot_tail(packets[18])
        assert len(observer.archive.nodes['full48'])==10 and observer.core.pairs==9
        assert state==tensor_digest(observer.core.main.cache)
        for arm in ARMS:
            tokens=[];answer=observer.ask_arm(arm,'What is visible?',tokens.append)
            assert answer['generated_ids']==[t['token_id'] for t in tokens]
            assert state==tensor_digest(observer.core.main.cache)
            answers.append(dict(arm=arm,phase='query_only_odd_tail',generated_tokens=len(tokens),main_and_archive_unchanged=True))
        try:observer.append_pair(*packets[18:20])
        except RuntimeError:pass
        else:raise AssertionError('padded query tail must not become main observation history')
        observer.close()
        assert all(not ns for ns in observer.archive.nodes.values())
    receipt=dict(selection_contract=selection_receipt,status='passed',context_layouts=len(layouts),questions=303,
        independent_reduction_cases=contract_cases,same24_visual_count_and_equal_time_labels=909,
        original_selected_coordinates_preserved=True,actual_RGB_frames=20,actual_main_vision_calls=9,
        snapshot_only_vision_calls=1,odd_tail_never_committed_to_main=True,
        tiny_QA=answers,pretrained_loads=0,GPU_submissions=0,seconds=time.monotonic()-started,
        max_input_tokens=max(x['input_tokens'] for x in layouts),
        max_suffix_tokens=max(x['suffix_tokens'] for x in layouts),
        observed_RSS_bytes=env['psutil'].Process().memory_info().rss,
        protocol_sha256=sha(OUT/'inputs/protocol.json'),layouts=layouts)
    save(OUT/'CPU_preflight.json',receipt)
    print({k:v for k,v in receipt.items() if k!='layouts'})




def check_selection():
    from selection import select,spatial_anchors,odd_indices
    torch.set_num_threads(1)
    x=torch.eye(48); anchors=spatial_anchors(); odd=odd_indices()
    assert len(anchors)==12 and len(anchors.unique())==12
    assert set(anchors.tolist()).issubset(set(odd.tolist()))
    assert all(int(((anchors//8)==r).sum())==2 for r in range(6))
    cold,_=select(x)
    assert all(torch.equal(ids,odd) for ids in cold.values())
    changed=[i for i in range(48) if i not in anchors.tolist()][:12]
    current=x.clone();current[changed]*=-1
    ids,score=select(current,x)
    assert set(changed).issubset(set(ids['change24'].tolist()))
    assert set(ids['mix24'].tolist())==set(changed)|set(anchors.tolist())
    assert all(float(score[i])==2. for i in changed)
    # Independent scalar ranking, no tensor sort or vector normalization.
    import math
    torch.manual_seed(246)
    previous=torch.randn(48,32); current=torch.randn(48,32)
    scalar=[]
    for a,b in zip(current.tolist(),previous.tolist()):
        norm=lambda row:max(math.sqrt(sum(v*v for v in row)),1e-12)
        scalar.append(max(0.,min(2.,1-sum(u*v for u,v in zip(a,b))/(norm(a)*norm(b)))))
    rank=sorted(range(48),key=lambda i:(-scalar[i],i))
    selected,_=select(current,previous)
    assert selected['change24'].tolist()==sorted(rank[:24])
    assert selected['mix24'].tolist()==sorted(anchors.tolist()+[i for i in rank if i not in anchors.tolist()][:12])
    zeros,_=select(torch.zeros(48,32),torch.zeros(48,32))
    assert zeros['change24'].tolist()==list(range(24))
    for name in ('odd24','change24','mix24'):
        a=Archive(pair_cap=2)
        a.append(previous,0,0.,1.);a.append(current,1,2.,3.)
        n=a.nodes[name][-1]; original=n.coords[:,1]*8+n.coords[:,2]
        assert torch.equal(n.features,current[original]) and torch.equal(original,selected[name])
        assert a.status()['selector_bytes']==48*32*4 and a.selector_digest() is not None
        before=(a.fingerprint(),a.selector_digest())
        try:a.append(current,2,4.,5.)
        except ValueError:pass
        else:raise AssertionError('overflow accepted')
        assert before==(a.fingerprint(),a.selector_digest())
        a.clear();assert a.previous is None and all(not ns for ns in a.nodes.values())
    # Selection/retained nodes own their state, no caller tensor aliasing.
    a=Archive();a.append(current,0,0.,1.);digest=a.selector_digest();current.zero_()
    assert digest==a.selector_digest()
    return dict(contract_cases=9,cold_start_equal_odd=True,known_changed_positions_preserved=True,
        scalar_reference=True,static12_preserved=True,zero_vector_ties=True,
        budget_and_original_values=True,previous_pair_owned=True,clear_releases_selector=True,
        failed_append_preserves_state=True)

if __name__=='__main__':main()
