"""All 34 full native inputs with pair-sized CPU preprocessing, no long forward."""
import hashlib
import json
from pathlib import Path
import sys
import time

from inventory_local import ROOT,OUT


def main():
    start=time.monotonic();sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
    import runtime_bridge
    runtime_bridge.REPO=ROOT/'publish/qwen35-streaming-2b'
    runtime_bridge.ACTIVE=runtime_bridge.REPO/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    env=runtime_bridge.runtime('native');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    from native import NativeObserver,inputs_for,chunked_inputs_for
    from score_memory_eval import prompt
    from PIL import Image
    from types import SimpleNamespace
    def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
    build=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    config=env['tiny_config'](build.processor);config['text_config']['max_position_embeddings']=262144
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config)).float().eval()
    layouts={r['id']:r for r in read('CPU_long_layout34_v2.json')['contexts']}
    rows=read('stream_requests34.json');snaps=read('stream_query_snapshots34.json');peak=0;audits=[];tiny=False
    for trace in read('actual_sampling_all6.json')['videos']:
        packets=[]
        for page in trace['pages']:
            path=ROOT/page['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==page['sha256']
            with Image.open(path) as image:
                for f in page['frames']:
                    rgb=image.crop(f['crop_xyxy']).tobytes();assert hashlib.sha256(rgb).hexdigest()==f['rgb_sha256']
                    packets.append(env['FramePacket'](f['frame_id'],f['pts_s'],
                        env['np'].frombuffer(rgb,dtype=env['np'].uint8).reshape(192,256,3).copy(),f['frame_id']+1))
        assert all(abs(p.frame_id/30.-p.timestamp)<1e-7 for p in packets)
        metadata=dict(fps=30.,total_num_frames=trace['decoded_frames'])
        question='What is visible? Answer briefly.'
        with torch.inference_mode():
            full,delta,_=inputs_for(env,model,build,packets[:4],metadata,question,'cpu')
            chunked,d,_=chunked_inputs_for(env,model,build,packets[:4],metadata,question,'cpu')
            assert set(full)==set(chunked) and torch.equal(delta,d)
            assert all(torch.equal(full[k],chunked[k]) for k in full),'native preprocessing changed input'
            if not tiny:
                ref=NativeObserver(env,model,build,SimpleNamespace(metadata=metadata))
                ref.append_pair(*packets[:2]);ref.append_pair(*packets[2:4]);callbacks=[]
                with env['fp32_reference']():
                    answer=ref.ask(question,read_from='native',on_token=callbacks.append)
                    model.model.rope_deltas=delta;old,_=env['complete'](model,build.processor,dict(full),8)
                assert answer['generated_ids']==old['generated_ids']==[e['token_id'] for e in callbacks]
                ref.close();tiny=True
            del full,chunked,delta,d
        for row in sorted([r for r in rows if r['source_path']==trace['source_path']],key=lambda r:(r['query_time_s'],r['id'])):
            snap=next(s for s in snaps if s['query_id']==row['id'])
            with torch.inference_mode():
                inputs,delta,meta=chunked_inputs_for(env,model,build,packets[:snap['committed_pairs']*2],
                    metadata,prompt(row),'cpu')
            assert hashlib.sha256(inputs['input_ids'].numpy().tobytes()).hexdigest()==layouts[row['id']]['input_ids_sha256']
            assert hashlib.sha256(inputs['position_ids'].numpy().tobytes()).hexdigest()==layouts[row['id']]['positions_sha256']
            assert meta['observation_through_s']==snap['committed_through_s']
            audits.append(dict(id=row['id'],**meta,pixel_values_shape=list(inputs['pixel_values_videos'].shape)))
            peak=max(peak,env['psutil'].Process().memory_info().rss)
            assert peak<1536*2**20,dict(RSS=peak,query=row['id'])
            del inputs,delta
        print(dict(source=trace['source_path'],contexts_completed=len(audits),observed_peak_RSS=peak),flush=True)
        del packets
    assert len(audits)==34 and not torch.cuda.is_initialized()
    result=dict(status='passed',contexts=audits,full_processor_short_equality_sources=6,
        original_short_tiny_callback_generation_equal=tiny,peak_observed_RSS_bytes=peak,
        maximum_input_tokens=max(r['input_tokens'] for r in audits),seconds=time.monotonic()-start,
        pretrained_model_loads=0,GPU_jobs=0,long_model_forwards=0,random_tiny_instances=1,
        runtime_sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        limitations=['pair-chunk CPU preprocessing, concatenated original full visual model input',
          'long actual pixel/layout checks; only short random-network forward',
          'PNG pixels fingerprinted to original actual PTS sampling'])
    with (OUT/'CPU_native_pixels34_v3.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print({k:v for k,v in result.items() if k not in ('contexts','runtime_sources_sha256')},flush=True)


if __name__=='__main__':main()
