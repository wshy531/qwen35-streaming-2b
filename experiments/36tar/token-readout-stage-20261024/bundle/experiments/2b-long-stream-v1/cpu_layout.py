"""Memory-light layout preflight; never claims full-prefix pixel preprocessing."""
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
    from PIL import Image
    from score_memory_eval import prompt
    from native import inputs_for,NativeObserver
    from types import SimpleNamespace
    def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
    processor=env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    build=env['PairInputBuilder'](processor,canvas=(192,256),resize_mode='letterbox')
    config=env['tiny_config'](processor);config['text_config']['max_position_embeddings']=262144
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**config)).float().eval()
    assert json.loads((ROOT/'models/Qwen3.5-2B/config.json').read_text())['text_config']['max_position_embeddings']==262144
    receipts=[];rows=read('stream_requests34.json');snapshots=read('stream_query_snapshots34.json')
    def layout(frames,question):
        assert len(frames)%2==0 and all(abs(f['frame_id']/30.-f['pts_s'])<1e-7 for f in frames)
        timestamps=processor._calculate_timestamps([f['frame_id'] for f in frames],30.,2)
        visual=''.join(f'<{t:.1f} seconds>'+processor.vision_start_token+processor.video_token*48+
            processor.vision_end_token for t in timestamps)
        rendered=build.template(question).replace(build.marker,visual)
        ids=build.tokenize(rendered);grid=torch.tensor([[len(frames)//2,12,16]])
        oracle=grid.repeat_interleave(grid[:,0],0);oracle[:,0]=1
        mm_types=torch.zeros_like(ids);mm_types[ids==processor.video_token_id]=2
        coords,delta=model.model.get_rope_index(input_ids=ids,mm_token_type_ids=mm_types,video_grid_thw=oracle)
        pos=torch.cat((torch.arange(ids.shape[1])[None,None,:],coords),0)
        assert ids.shape[1]+8<=32768 and int(pos.max())<262144
        return ids,pos,grid,delta
    # Pixel/full-processor equality only for four real frames, clearly scoped.
    trace=read('actual_sampling_all6.json')['videos'][0];page=trace['pages'][0]
    path=ROOT/page['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==page['sha256']
    packets=[]
    with Image.open(path) as image:
        for f in page['frames'][:4]:
            rgb=image.crop(f['crop_xyxy']).tobytes();assert hashlib.sha256(rgb).hexdigest()==f['rgb_sha256']
            packets.append(env['FramePacket'](f['frame_id'],f['pts_s'],
                env['np'].frombuffer(rgb,dtype=env['np'].uint8).reshape(192,256,3).copy(),f['frame_id']+1))
    question='What is visible? Answer briefly.';metadata=dict(fps=30.,total_num_frames=trace['decoded_frames'])
    with torch.inference_mode():
        actual,delta,_=inputs_for(env,model,build,packets,metadata,question,'cpu')
        ids,pos,grid,d=layout(trace['frames'][:4],question)
        assert torch.equal(ids,actual.input_ids) and torch.equal(pos,actual.position_ids)
        assert torch.equal(grid,actual.video_grid_thw) and torch.equal(d,delta)
        ref=NativeObserver(env,model,build,SimpleNamespace(metadata=metadata))
        ref.append_pair(*packets[:2]);ref.append_pair(*packets[2:]);emitted=[]
        with env['fp32_reference']():
            output=ref.ask(question,read_from='native',on_token=emitted.append)
            model.model.rope_deltas=delta
            old,_=env['complete'](model,processor,dict(actual),8)
        assert output['generated_ids']==old['generated_ids']==[e['token_id'] for e in emitted]
        ref.close()
    del actual,packets,ref
    traces={t['source_path']:t for t in read('actual_sampling_all6.json')['videos']}
    peak=0
    for row in rows:
        snap=next(s for s in snapshots if s['query_id']==row['id'])
        frames=traces[row['source_path']]['frames'][:snap['committed_pairs']*2]
        with torch.inference_mode():ids,pos,grid,delta=layout(frames,prompt(row))
        receipts.append(dict(id=row['id'],input_tokens=ids.shape[1],visual_tokens=len(frames)//2*48,
            maximum_position=int(pos.max()),input_ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            positions_sha256=hashlib.sha256(pos.numpy().tobytes()).hexdigest(),
            real_frames=len(frames),committed_through_s=frames[-1]['pts_s']))
        peak=max(peak,env['psutil'].Process().memory_info().rss);assert peak<1536*2**20
    assert len(receipts)==34 and not torch.cuda.is_initialized()
    result=dict(status='passed',contexts=receipts,maximum_input_tokens=max(r['input_tokens'] for r in receipts),
        short_actual4_frame_processor_layout_equal=True,tiny_original_incremental_generation_equal=True,
        pretrained_model_loads=0,GPU_jobs=0,random_tiny_instances=1,peak_RSS_observed_bytes=peak,
        seconds=time.monotonic()-start,explicit_guard=32768,native_max_position_embeddings=262144,
        limitations=['34 long layouts only; full long pixel preprocessing NOT passed locally',
          'only4 frames have full processor and random tiny forward comparison',
          'no pretrained capability or real-time claim'])
    with (OUT/'CPU_long_layout34_v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print({k:v for k,v in result.items() if k!='contexts'},flush=True)


if __name__=='__main__':main()
