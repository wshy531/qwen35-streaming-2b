"""Short CPU-only interface checks in the already prepared cluster environment."""
import argparse
import hashlib
from pathlib import Path
import time

from evaluate import ROOT,verify,runtime,read,sha,write


def main(run,stage):
    import os
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    start=time.monotonic();protocol,rows,sources,snaps=verify(run)
    local=read(run/'local_cpu_proofs.json');assert local['native_passed'] and local['memory_passed']
    env=runtime(stage);torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    assert not torch.cuda.is_available()
    from replay import VideoSource
    from score_memory_eval import prompt
    build=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(ROOT/'models/Qwen3.5-2B',
        local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    cfg=env['tiny_config'](build.processor);cfg['text_config']['max_position_embeddings']=262144
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**cfg)).float().eval()
    assert max(build.question_ids(prompt(q)).shape[1] for q in rows)<=128
    if stage=='native':
        for s in sources:assert sha(ROOT/s['media_path'])==s['media_sha256']
    source=VideoSource(ROOT/sources[0]['media_path'],sources[0],env['FramePacket'])
    it=iter(source.frames());packets=[];pending=[]
    for _ in range(16):
        f=next(it);pending.append(f.frame_id)
        if len(pending)==2:packets.extend(source.take_pair(pending));pending=[]
    it.close()
    contexts=answers=0
    with torch.inference_mode(),env['fp32_reference']():
        if stage=='native':
            from native import NativeObserver,layout_for,inputs_for,chunked_inputs_for
            from types import SimpleNamespace
            expected={q['id']:q for q in read(run/'expected_layout34.json')}
            for row in rows:
                spec=next(s for s in sources if s['source_path']==row['source_path'])
                snap=next(s for s in snaps if s['query_id']==row['id'])
                fs=spec['frames'][:snap['committed_pairs']*2]
                args=[SimpleNamespace(frame_id=f['frame_id'],timestamp=f['pts_s']) for f in fs]
                assert all(abs(a.frame_id/30.-a.timestamp)<1e-7 for a in args)
                ins,_=layout_for(model,build,args,dict(fps=30.),prompt(row),torch.tensor([[len(fs)//2,12,16]]),'cpu')
                assert hashlib.sha256(ins['input_ids'].numpy().tobytes()).hexdigest()==expected[row['id']]['input_ids_sha256']
                assert hashlib.sha256(ins['position_ids'].numpy().tobytes()).hexdigest()==expected[row['id']]['positions_sha256']
                contexts+=1
            q=prompt(rows[0]);whole,delta,_=inputs_for(env,model,build,packets[:4],source.metadata,q,'cpu')
            chunks,d,_=chunked_inputs_for(env,model,build,packets[:4],source.metadata,q,'cpu')
            assert set(whole)==set(chunks) and all(torch.equal(whole[k],chunks[k]) for k in whole)
            assert torch.equal(delta,d)
            obs=NativeObserver(env,model,build,source);obs.append_pair(*packets[:2]);obs.append_pair(*packets[2:4])
            tokens=[];actual=obs.ask(q,read_from='native',on_token=tokens.append)
            model.model.rope_deltas=delta;old,_=env['complete'](model,build.processor,dict(whole),8)
            assert actual['generated_ids']==old['generated_ids']==[e['token_id'] for e in tokens]
            obs.close();answers=1
        else:
            from observer import Observer
            from first_2b_baseline import tensor_digest
            from readout import context,Limits
            for policy in ('M0','M2'):
                obs=Observer(env,model,build,policy=policy,main_tokens=384,question_tokens=128,answer_tokens=8)
                for i in range(0,16,2):obs.append_pair(*packets[i:i+2])
                for row in rows:
                    context(model,build,obs.bank.export(),prompt(row),limits=Limits(8192,384,8,128),max_new_tokens=8)
                    contexts+=1
                before=tensor_digest(obs.main.cache)
                for route in ('main','bank','text'):
                    tokens=[];result=obs.ask(prompt(rows[0]),read_from=route,on_token=tokens.append)
                    assert result['generated_ids']==[t['token_id'] for t in tokens]
                    assert tensor_digest(obs.main.cache)==before;answers+=1
                assert obs.resets>=1 and obs.bank.status()['processed_pairs']==8
                obs.close()
    assert not torch.cuda.is_initialized()
    write(run/f'cpu_{stage}.json',dict(status='passed',stage=stage,contexts=contexts,tiny_answers=answers,
        actual_sampled_frames=16,seconds=time.monotonic()-start,pretrained_loads=0,GPU_jobs=0,
        protocol_sha256=sha(run/'protocol.json'),local_proof_sha256=sha(run/'local_cpu_proofs.json')))
    print(dict(status='passed',stage=stage,contexts=contexts,tiny_answers=answers),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path)
    p.add_argument('--stage',required=True,choices=['native','memory']);a=p.parse_args();main(a.run,a.stage)
