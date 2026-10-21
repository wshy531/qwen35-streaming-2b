"""Real sampled pixels + random tiny network. Never loads pretrained weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from inventory_local import ROOT,OUT


def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    begin=time.monotonic()
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(Path(__file__).parent),
        '-p','test_online_bank.py','-v'],capture_output=True,text=True,timeout=90)
    save('CPU_incremental_bank_contracts.json',dict(exit_code=result.returncode,stdout=result.stdout,
        stderr=result.stderr,model_loads=0,GPU_jobs=0))
    assert result.returncode==0
    import torch,psutil
    torch.set_num_threads(1);torch.manual_seed(123)
    active=ROOT/'publish/qwen35-streaming-2b/experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    sys.path.insert(0,str(active));sys.path.append(str(ROOT/'experiments/2b-natural-eval-v1'))
    from qwen35_local import Qwen3_5ForConditionalGeneration,Qwen3_5Config
    from streaming.hybrid import HybridSequence
    from streaming.online import PairInputBuilder,fp32_reference,greedy_decode
    from streaming.window import FramePacket
    from transformers import AutoProcessor
    from first_2b_baseline import tiny_config,tensor_digest
    from observer import Observer,emit_decode
    from PIL import Image
    import numpy as np
    processor=AutoProcessor.from_pretrained(ROOT/'models/Qwen3.5-2B',local_files_only=True)
    model=Qwen3_5ForConditionalGeneration(Qwen3_5Config(**tiny_config(processor))).float().eval()
    assert not torch.cuda.is_initialized()
    builder=PairInputBuilder(processor,canvas=(192,256),resize_mode='letterbox')
    observer=Observer({'HybridSequence':HybridSequence,'fp32_reference':fp32_reference},model,builder,
        policy='M2',main_tokens=384,question_tokens=128,answer_tokens=4)
    trace=json.loads((OUT/'actual_sampling_all6.json').read_text(encoding='utf-8'))['videos'][0]
    first=trace['frames'][:16];audits=[];events=[]
    question='What is visible? Answer briefly.'

    def packet(meta):
        page=next(p for p in trace['pages'] if any(f['frame_id']==meta['frame_id'] for f in p['frames']))
        item=next(f for f in page['frames'] if f['frame_id']==meta['frame_id'])
        with Image.open(ROOT/page['path']) as image:rgb=image.crop(item['crop_xyxy']).tobytes()
        assert hashlib.sha256(rgb).hexdigest()==meta['rgb_sha256']
        return FramePacket(meta['frame_id'],meta['pts_s'],np.frombuffer(rgb,dtype=np.uint8).reshape(192,256,3).copy(),meta['frame_id']+1)

    for i in range(0,len(first),2):
        before_encodes=observer.vision_encodings
        state=observer.append_pair(packet(first[i]),packet(first[i+1]))
        assert observer.vision_encodings==before_encodes+1
        assert observer.bank.status()['processed_pairs']==i//2+1
        if i//2 in (1,3,7):
            before=tensor_digest(observer.main.cache);before_status=observer.status()
            bank_hash=observer.bank.fingerprint()
            for source in ('main','bank'):
                emitted=[]
                answer=observer.ask(question,read_from=source,on_token=lambda e:emitted.append(e))
                assert [e['token_id'] for e in emitted]==answer['generated_ids']
                assert all(a['monotonic_s']<=b['monotonic_s'] for a,b in zip(emitted,emitted[1:]))
                assert answer['first_token_seconds']<=answer['last_token_seconds']
                assert tensor_digest(observer.main.cache)==before and observer.status()==before_status
                assert observer.bank.fingerprint()==bank_hash
                audits.append(dict(pair=i//2,source=source,callback_events=len(emitted),
                    generated_ids=answer['generated_ids'],main_state_bitwise_unchanged=True,bank_unchanged=True,
                    observation_through_s=answer['observation_through_s']))
            events.append(state)
        assert psutil.Process().memory_info().rss<1536*2**20
    assert observer.resets>=1 and observer.bank.status()['processed_pairs']==8
    # Check the new incremental emitter against the frozen greedy loop on forks.
    a=observer.main.fork();b=observer.main.fork()
    suffix=builder.question_ids(question)
    with fp32_reference():
        la=a.append(input_ids=suffix,logits_mode='last');lb=b.append(input_ids=suffix,logits_mode='last')
        recorded=[]
        new=emit_decode(a,la,processor,4,{processor.tokenizer.eos_token_id},recorded.append)
        old,_=greedy_decode(b,lb,4,{processor.tokenizer.eos_token_id})
    assert new['generated_ids']==old
    a.cache.reset();b.cache.reset()
    final=observer.status();observer.close()
    save('CPU_tiny_observer_preflight.json',dict(status='passed',pretrained_model_loads=0,GPU_jobs=0,
        random_tiny_model_instances=1,hidden_size=32,actual_pairs=8,actual_frames=16,
        visual_encodings=final['vision_encodings'],query_branches=len(audits),
        resets=final['resets'],bank_pairs_after_main_reset=8,main_and_bank_isolation=audits,
        incremental_emitter_matches_frozen_greedy=True,CUDA_initialized=torch.cuda.is_initialized(),
        final_state=final,CPU_RSS_at_completion=psutil.Process().memory_info().rss,
        seconds=time.monotonic()-begin,
        limitations=['16 real frames from first source, not full long-video model execution',
          'random32-wide features do not measure pretrained2B capability',
          'wall-clock playback and GPU resources not tested']))
    print(dict(status='passed',actual_pairs=8,queries=6,resets=final['resets'],pretrained_loads=0,GPU_jobs=0))


if __name__=='__main__':main()
