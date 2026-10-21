"""Actual 34 long-prefix processor/position contracts; only short tiny forwards."""
import hashlib
import json
from pathlib import Path
import sys
import time

from inventory_local import ROOT,OUT


def main():
    began=time.monotonic()
    base=ROOT/'experiments/2b-natural-eval-v1';sys.path.insert(0,str(base))
    import runtime_bridge
    runtime_bridge.REPO=ROOT/'publish/qwen35-streaming-2b'
    runtime_bridge.ACTIVE=runtime_bridge.REPO/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    env=runtime_bridge.runtime('native');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    assert not torch.cuda.is_initialized()
    from native import NativeObserver,inputs_for
    from score_memory_eval import prompt
    from PIL import Image
    def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
    build=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    config=env['tiny_config'](build.processor)
    config['text_config']['max_position_embeddings']=262144
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config)).float().eval()
    assert json.loads((ROOT/'models/Qwen3.5-2B/config.json').read_text())['text_config']['max_position_embeddings']==262144
    receipts=[];tiny=False;peak=0;rows=read('stream_requests34.json');snaps=read('stream_query_snapshots34.json')
    for trace in read('actual_sampling_all6.json')['videos']:
        packets=[]
        for page in trace['pages']:
            path=ROOT/page['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==page['sha256']
            with Image.open(path) as sheet:
                for f in page['frames']:
                    rgb=sheet.crop(f['crop_xyxy']).tobytes()
                    assert hashlib.sha256(rgb).hexdigest()==f['rgb_sha256']
                    packets.append(env['FramePacket'](f['frame_id'],f['pts_s'],
                        env['np'].frombuffer(rgb,dtype=env['np'].uint8).reshape(192,256,3).copy(),f['frame_id']+1))
        metadata=dict(fps=30.,total_num_frames=trace['decoded_frames'])
        # All six actual sample indices/PTS establish exactly 30Hz for these
        # sources; fail rather than silently invent a different frame clock.
        assert all(abs(p.frame_id/30.-p.timestamp)<1e-7 for p in packets)
        for row in sorted([q for q in rows if q['source_path']==trace['source_path']],key=lambda q:(q['query_time_s'],q['id'])):
            snap=next(s for s in snaps if s['query_id']==row['id'])
            prefix=packets[:snap['committed_pairs']*2]
            with torch.inference_mode():
                inputs,delta,meta=inputs_for(env,model,build,prefix,metadata,prompt(row),'cpu')
            assert meta['observation_through_s']==snap['committed_through_s']
            receipts.append(dict(id=row['id'],**meta))
            peak=max(peak,env['psutil'].Process().memory_info().rss);assert peak<2*2**30
            del inputs,delta
        if not tiny:
            # The incremental original emitter must agree with the existing
            # original full-prefill + one-token continuation implementation.
            from types import SimpleNamespace
            source=SimpleNamespace(metadata=metadata)
            ref=NativeObserver(env,model,build,source)
            ref.append_pair(*packets[:2]);ref.append_pair(*packets[2:4])
            q='What is visible? Answer briefly.';emitted=[]
            with torch.inference_mode(),env['fp32_reference']():
                actual=ref.ask(q,read_from='native',on_token=emitted.append)
                ins,d,_=inputs_for(env,model,build,packets[:4],metadata,q,'cpu')
                model.model.rope_deltas=d;expected,_=env['complete'](model,build.processor,dict(ins),8)
            assert actual['generated_ids']==expected['generated_ids']==[i['token_id'] for i in emitted]
            assert ref.status()['pairs']==2
            ref.close();tiny=True;del ins,d
        print(dict(source=trace['source_path'],contexts=sum(r['id'] in {q['id'] for q in rows if q['source_path']==trace['source_path']} for r in receipts)),flush=True)
        del packets,prefix
    assert len(receipts)==34 and tiny
    report=dict(status='passed',contexts=receipts,maximum_input_tokens=max(r['input_tokens'] for r in receipts),
        tiny_short_generation_matches_frozen_original=True,pretrained_loads=0,GPU_jobs=0,
        random_tiny_instances=1,peak_RSS_observed_bytes=peak,seconds=time.monotonic()-began,
        native_max_position_embeddings=262144,explicit_context_guard=32768,
        limitations=['long contexts: processor/positions only; no long model forwards',
            'random short-generation equality is not pretrained capability',
            'CPU pixels reconstructed from SHA-verified review PNG crops'])
    with (OUT/'CPU_native_long_contexts34.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print({k:v for k,v in report.items() if k!='contexts'},flush=True)


if __name__=='__main__':main()
