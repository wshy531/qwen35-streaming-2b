"""170 long layouts from synthetic fixtures and five tiny real-pixel answers."""
import argparse
import gzip
import json
from pathlib import Path
import time

from evaluate_compare import ROOT,read,verify,save,sha


def main(run,local):
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    start=time.monotonic();protocol,rows,sources,snapshots=verify(run,local=local)
    import runtime_bridge
    if local:
        runtime_bridge.REPO=ROOT/'publish/qwen35-streaming-2b'
        runtime_bridge.ACTIVE=runtime_bridge.REPO/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    env=runtime_bridge.runtime('memory');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    from observer_compare import CompareObserver,ARMS
    from replay import VideoSource
    from readout import context,Limits
    from score_memory_eval import prompt
    from capability_pilot import answer as old_answer
    from forest import Node,validate
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    cfg=env['tiny_config'](builder.processor);cfg['text_config']['max_position_embeddings']=262144
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg)).float().eval()
    by_id={r['id']:r for r in rows};contexts=0;max_input=0
    with torch.inference_mode(),env['fp32_reference']():
        for fixture in json.loads(gzip.decompress((run/'fixtures.json.gz').read_bytes())):
            for arm,packed in fixture['arms'].items():
                nodes=[]
                for entry in packed:
                    m=entry['metadata'];n=m['tokens']
                    node=Node(torch.randn(n,32),torch.tensor(entry['mass'],dtype=torch.float32).reshape(n,1),
                        torch.tensor(entry['coords'],dtype=torch.long),m['first'],m['last'],m['begin'],m['end'],m['center'],m['groups'],m['depth'])
                    validate(node);nodes.append(node)
                _,pos,_,meta=context(model,builder,nodes,prompt(by_id[fixture['id']]),limits=Limits(32768,768,16,128),max_new_tokens=8)
                assert meta['visual_tokens']<=(768 if arm.endswith('768') else 384) and int(pos.max())<262144
                contexts+=1;max_input=max(max_input,meta['input_tokens'])
    media=ROOT/sources[0]['media_path']
    if local:
        old=read(ROOT/'runs/2b_long_stream_prepare_20261020_01/stream_sources6.json')
        media=Path(next(s['media_path'] for s in old if s['source_path']==sources[0]['source_path']))
    source=VideoSource(media,sources[0],env['FramePacket']);it=iter(source.frames());packets=[]
    try:
        for _ in range(16):
            frame=next(it);packets.append(source.buffer.pop(frame.frame_id))
    finally:it.close();source.buffer.clear()
    obs=CompareObserver(env,model,builder,main_tokens=384);answers=0
    with torch.inference_mode(),env['fp32_reference']():
        for i in range(0,12,2):obs.append_pair(*packets[i:i+2])
        assert obs.core.vision_encodings==6 and obs.core.resets>=1
        emitted=[];result=obs.ask(prompt(rows[0]),read_from='five_banks',on_token=emitted.append)
        for arm in ARMS:
            assert result['arms'][arm]['generated_ids']==[t['token_id'] for t in emitted if t['arm']==arm]
            old=old_answer(env,model,builder,obs.nodes()[arm],prompt(rows[0]))
            assert old['generated_ids']==result['arms'][arm]['generated_ids'];answers+=1
        for i in range(12,16,2):obs.append_pair(*packets[i:i+2])
        assert obs.core.vision_encodings==8 and obs.core.pairs==8
    obs.close();assert contexts==170 and answers==5 and not torch.cuda.is_initialized()
    rss=env['psutil'].Process().memory_info().rss;assert rss<3*2**30
    proof=dict(status='passed',protocol_sha256=sha(run/'protocol.json'),contexts=contexts,tiny_answers=answers,
        actual_RGB_frames=16,shared_encodings=8,main_reset_query_isolation_and_continuation=True,
        max_input_tokens=max_input,callbacks_match_frozen_greedy=True,
        pretrained_loads=0,GPU_jobs=0,seconds=time.monotonic()-start,RSS_observed_bytes=rss,
        limits='random32 fixture layouts and short random network, not real2B semantic evidence')
    save(run/('cpu_local.json' if local else 'cpu_cluster.json'),proof);print(proof)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--local',action='store_true')
    a=p.parse_args();main(a.run,a.local)
