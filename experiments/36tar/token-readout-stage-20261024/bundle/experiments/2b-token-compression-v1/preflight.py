"""All909 layouts and real RGB/tiny-model isolation. No pretrained weights/GPU."""
import hashlib
import os
import time
import torch

from common import ROOT, OUT, ARMS, read, save, sha, runtime
from memory import Archive, variants, context


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('explicit CPU-only environment required')
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
    # Independent loop aggregation rather than repeating scatter_add logic.
    contract_cases = 0
    for sample in (torch.randn(48,32), torch.zeros(48,32), torch.ones(48,32)):
        v = variants(sample,0,0.,1.)
        src = torch.nn.functional.normalize(sample[::2],dim=-1,eps=1e-12)
        dst = torch.nn.functional.normalize(sample[1::2],dim=-1,eps=1e-12)
        assignment = (src@dst.T).argmax(1)
        for j in range(24):
            contributors = [sample[2*j+1]]+[sample[2*i] for i in range(24) if int(assignment[i])==j]
            expected = torch.stack(contributors).mean(0)
            assert torch.allclose(v['tome24'].features[j],expected,atol=2e-7,rtol=2e-6)
            assert float(v['tome24'].mass[j])==len(contributors)
        assert float(v['tome24'].mass.sum())==48 and float(v['odd24'].mass.sum())==24
        assert torch.equal(v['odd24'].coords,v['tome24'].coords)
        assert torch.equal(v['full48'].features,sample)
        contract_cases+=1
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
                text = prompt(q)
                previous = None
                for arm in ARMS:
                    emb,pos,ids,meta = context(model,build,archive.nodes[arm],text)
                    assert meta['visual_tokens']==q['pairs']*(48 if arm=='full48' else 24)
                    if arm=='full48':
                        old_emb,old_pos,old_ids,_ = old_context(model,build,archive.nodes[arm],text,
                            limits=Limits(20000,15360,320),max_new_tokens=8)
                        assert torch.equal(emb,old_emb) and torch.equal(ids,old_ids) and torch.equal(pos,old_pos)
                        del old_emb,old_pos,old_ids
                    elif arm=='odd24':
                        previous=(ids.clone(),pos.clone(),meta['labels'])
                    else:
                        assert torch.equal(ids,previous[0]) and torch.equal(pos,previous[1]) and meta['labels']==previous[2]
                    layouts.append(dict(id=q['id'],cohort=q['cohort'],arm=arm,input_tokens=meta['input_tokens'],
                        visual_tokens=meta['visual_tokens'],suffix_tokens=meta['suffix_tokens']))
                    del emb,pos,ids
                assert hash_before==archive.fingerprint()
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
    receipt=dict(status='passed',context_layouts=len(layouts),questions=303,
        independent_reduction_cases=contract_cases,raw48_layouts_equal_existing_reader=303,
        odd_tome_IDs_and_four_axes_equal=303,actual_RGB_frames=20,actual_main_vision_calls=9,
        snapshot_only_vision_calls=1,odd_tail_never_committed_to_main=True,
        tiny_QA=answers,pretrained_loads=0,GPU_submissions=0,seconds=time.monotonic()-started,
        max_input_tokens=max(x['input_tokens'] for x in layouts),
        max_suffix_tokens=max(x['suffix_tokens'] for x in layouts),
        observed_RSS_bytes=env['psutil'].Process().memory_info().rss,
        protocol_sha256=sha(OUT/'inputs/protocol.json'),layouts=layouts)
    save(OUT/'CPU_preflight.json',receipt)
    print({k:v for k,v in receipt.items() if k!='layouts'})


if __name__=='__main__':main()
