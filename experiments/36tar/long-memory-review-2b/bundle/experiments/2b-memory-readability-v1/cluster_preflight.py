"""Short CPU interface checks, never pretrained weights or CUDA."""
import argparse
from pathlib import Path
import time

from evaluate import ROOT,verify,save,sha
from cpu_diagnostic import synthetic


def main(run):
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    start=time.monotonic();protocol,rows=verify(run)
    from runtime_bridge import runtime
    env=runtime('memory');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    assert not torch.cuda.is_available()
    from diagnostic import choices,answer,LIMITS,ARMS
    from readout import context
    from online_bank import nodes_digest
    from replay import VideoSource
    from score_memory_eval import prompt
    from capability_pilot import answer as frozen_answer
    build=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    cfg=env['tiny_config'](build.processor);cfg['text_config']['max_position_embeddings']=262144
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg)).float().eval()
    audits=[];peak=0;forward=False
    with torch.inference_mode(),env['fp32_reference']():
        for row in rows:
            assert sha(ROOT/row['media_path'])==row['media_sha256']
            raw,bank=synthetic(env,row);arms=choices(raw,bank,row)
            before={a:nodes_digest(ns) for a,ns in arms.items()}
            for arm in ARMS:
                emb,pos,_,meta=context(model,build,arms[arm],prompt(row),limits=LIMITS,max_new_tokens=8)
                assert emb.shape[1]+8<=32768 and int(pos.max())<262144
                audits.append(dict(id=row['id'],arm=arm,input_tokens=meta['input_tokens'],visual_tokens=meta['visual_tokens']))
                del emb,pos
            if not forward:
                emitted=[];actual=answer(env,model,build,bank,prompt(row),emitted.append)
                old=frozen_answer(env,model,build,bank,prompt(row))
                assert actual['generated_ids']==old['generated_ids']==[t['token_id'] for t in emitted]
                forward=True
            assert all(nodes_digest(arms[a])==h for a,h in before.items())
            peak=max(peak,env['psutil'].Process().memory_info().rss);assert peak<3*2**30
            del raw,bank,arms
        row=rows[0];source=VideoSource(ROOT/row['media_path'],
            dict(source_path=row['source_path'],media_sha256=row['media_sha256'],frames=row['committed_frames']),env['FramePacket'])
        it=iter(source.frames());packets=[]
        try:
            for _ in range(2):
                f=next(it);packets.append(source.buffer.pop(f.frame_id))
        finally:it.close();source.buffer.clear()
        inp,_=build.pair(*packets,0,0)
        vision=model.model.get_video_features(inp.pixel_values_videos,inp.video_grid_thw,return_dict=True)
        features=torch.cat(vision.pooler_output,0);assert features.shape==(48,32)
        node=env['load']().forest.raw(env['load']().features.raw_block(features,0,packets[0].timestamp,packets[1].timestamp))
        emitted=[];result=answer(env,model,build,[node],prompt(row),emitted.append)
        assert result['generated_ids']==[t['token_id'] for t in emitted]
    assert len(audits)==36 and forward and not torch.cuda.is_initialized()
    save(run/'cpu_cluster.json',dict(status='passed',protocol_sha256=sha(run/'protocol.json'),
        contexts=audits,maximum_input_tokens=max(a['input_tokens'] for a in audits),
        tiny_answers=2,actual_RGB_frames=2,all6_media_SHA_verified=True,
        frozen_reader_matches=True,pretrained_loads=0,GPU_jobs=0,
        seconds=time.monotonic()-start,peak_observed_RSS_bytes=peak,
        limitations='synthetic layouts plus two-frame random tiny forward; no real2B semantic evidence'))
    print(dict(status='passed',contexts=36,tiny_answers=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
