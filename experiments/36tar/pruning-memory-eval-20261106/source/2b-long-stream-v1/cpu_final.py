"""Final current-code contracts before any GPU request; no pretrained weights."""
import hashlib
import json
from pathlib import Path
import sys
import time

from inventory_local import ROOT,OUT


def main():
    began=time.monotonic();sys.path.insert(0,str(ROOT/'experiments/2b-natural-eval-v1'))
    import runtime_bridge
    runtime_bridge.REPO=ROOT/'publish/qwen35-streaming-2b'
    runtime_bridge.ACTIVE=runtime_bridge.REPO/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
    env=runtime_bridge.runtime('memory');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    from observer import Observer
    from readout import context,Limits
    from capability_pilot import text_context,answer
    from score_memory_eval import prompt
    from first_2b_baseline import tensor_digest
    from replay import replay,VideoSource
    from native import chunked_inputs_for
    from schedule import Frame
    from cpu_replay import AuditObserver
    from PIL import Image
    def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
    rows=read('stream_requests34.json');snaps=read('stream_query_snapshots34.json')
    build=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](build.processor))).float().eval()
    first_spec=read('stream_sources6.json')[0]
    source=VideoSource(ROOT/first_spec['media_path'],first_spec,env['FramePacket'])
    iterator=iter(source.frames());ids=[];live_packets=[]
    for _ in range(4):
        frame=next(iterator);ids.append(frame.frame_id)
        if len(ids)==2:live_packets.extend(source.take_pair(ids));ids=[]
    assert set(source.metadata)=={'fps','total_num_frames'}
    with torch.inference_mode():
        native_input,_,_=chunked_inputs_for(env,model,build,live_packets,source.metadata,'Describe briefly.','cpu')
    assert native_input['input_ids'].shape[1]>0
    iterator.close();del native_input,live_packets,source
    lengths={r['id']:build.question_ids(prompt(r)).shape[1] for r in rows}
    assert max(lengths.values())<=128,lengths
    trace=read('actual_sampling_all6.json')['videos'][0];packets=[]
    for page in trace['pages'][:1]:
        with Image.open(ROOT/page['path']) as image:
            for f in page['frames'][:16]:
                rgb=image.crop(f['crop_xyxy']).tobytes();assert hashlib.sha256(rgb).hexdigest()==f['rgb_sha256']
                packets.append(env['FramePacket'](f['frame_id'],f['pts_s'],
                    env['np'].frombuffer(rgb,dtype=env['np'].uint8).reshape(192,256,3).copy(),f['frame_id']+1))
    contexts=branches=0;states=[]
    with torch.inference_mode(),env['fp32_reference']():
        for policy in (None,'M0','M2'):
            obs=Observer(env,model,build,policy=policy,main_tokens=384,question_tokens=128,answer_tokens=8)
            for i in range(0,16,2):obs.append_pair(*packets[i:i+2])
            assert obs.resets>=1
            for row in rows:
                q=prompt(row)
                if policy:
                    _,_,_,meta=context(model,build,obs.bank.export(),q,
                        limits=Limits(8192,384,8,128),max_new_tokens=8)
                    assert meta['visual_tokens']<=384;contexts+=1
                elif row is rows[0]:
                    for r in rows:
                        _,_,_,meta=text_context(env,model,build,prompt(r));assert meta['visual_tokens']==0;contexts+=1
            before=tensor_digest(obs.main.cache);status=obs.status()
            for source in (('main','text') if policy is None else ('bank','text')):
                emitted=[];result=obs.ask(prompt(rows[0]),read_from=source,on_token=emitted.append)
                assert result['generated_ids']==[e['token_id'] for e in emitted]
                assert tensor_digest(obs.main.cache)==before and obs.status()==status
                if source=='text':
                    old=answer(env,model,build,[],prompt(rows[0]));assert old['generated_ids']==result['generated_ids']
                branches+=1
            states.append(dict(policy=policy,resets=obs.resets,pairs=obs.pairs))
            obs.close()
    # Recheck the final prompt/dispatch code over all real metadata. The full
    # media decoder's earlier receipt is separate; this does not decode again.
    class MetadataSource:
        def __init__(self,spec):self.spec=spec;self.buffer={};self.completed=False;self.peak_buffers=0
        def frames(self):
            for f in self.spec['frames']:
                self.buffer[f['frame_id']]=f
                self.peak_buffers=max(self.peak_buffers,len(self.buffer));assert self.peak_buffers<=3
                yield Frame(f['frame_id'],f['pts_s'])
            self.completed=True
        def take_pair(self,ids):
            from types import SimpleNamespace
            return tuple(SimpleNamespace(frame_id=m['frame_id'],timestamp=m['pts_s'],rgb=packets[0].rgb)
                for m in (self.buffer.pop(i) for i in ids))
        decode_seconds=transform_seconds=hash_seconds=0.
        @property
        def sampled(self):return len(self.spec['frames'])
    count=0
    for spec in read('stream_sources6.json'):
        source=MetadataSource(spec)
        result=replay(source,[r for r in rows if r['source_path']==spec['source_path']],AuditObserver(),
            [s for s in snaps if s['source_path']==spec['source_path']],lambda e:None,lambda e:None,read_from='audit')
        count+=result['answers']
    assert contexts==102 and count==34 and branches==6 and not torch.cuda.is_initialized()
    proof=dict(status='passed',actual_frames=16,actual_pair_encodings=24,isolated_query_branches=branches,
        main_bitwise_isolation=True,text_callback_matches_frozen_text_greedy=True,
        full_question_contexts=contexts,maximum_question_suffix_tokens=max(lengths.values()),
        final_prompt_metadata_replay_queries=count,actual_source4_frame_native_input=True,
        states=states,pretrained_loads=0,GPU_jobs=0,
        random_tiny_instances=1,seconds=time.monotonic()-began,
        runtime_sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    with (OUT/'CPU_final_current_code_v2.json').open('x',encoding='utf-8') as f:json.dump(proof,f,indent=2)
    print({k:v for k,v in proof.items() if k!='runtime_sources_sha256'},flush=True)


if __name__=='__main__':main()
