"""One bounded 7B native/streamed qualification and four-case cost measurement."""
import argparse,gc,hashlib,json,os,time,traceback
from collections import deque
from pathlib import Path
import torch
import transformers
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from readout import Memory,ARMS,TOKENS,answer,question,context,generate_context,native_inputs,pair_inputs
from video_source import CausalPairs

ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_transfer_prepare_20261107_01'
MODEL=ROOT/'models/Qwen2.5-VL-7B-Instruct'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*2**20):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def safe(value):
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)):return [safe(v) for v in value]
    if isinstance(value,Path):return str(value)
    return value
def save(p,value):Path(p).write_text(json.dumps(safe(value),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

@torch.no_grad()
def qualify(model,processor,memory,q,frames,out,guard,count):
    prompt=question(q,0,'B');nodes=memory.select('recent8frames',q)
    emb,pos,ids=context(model,processor,nodes,prompt)
    native=native_inputs(processor,list(frames),prompt).to('cuda')
    assert torch.equal(ids,native['input_ids'])
    expected_pos,_=model.model.get_rope_index(native['input_ids'],video_grid_thw=native['video_grid_thw'],
        second_per_grid_ts=native['second_per_grid_ts'])
    assert torch.equal(pos,expected_pos)
    full=model(**native,use_cache=False,logits_to_keep=1).logits[0,-1].float().cpu()
    count['vision']+=1
    own_ids,stream=generate_context(model,emb,pos,limit=4,guard=guard)
    count['qualification_tokens']+=len(own_ids)
    model.model.rope_deltas=None
    native_generation=model.generate(**native,max_new_tokens=4,do_sample=False)
    count['vision']+=1
    native_ids=native_generation[0,ids.shape[1]:].tolist()
    count['qualification_tokens']+=len(native_ids)
    diff=(full-stream).abs()
    total_variation=float((full.softmax(-1)-stream.softmax(-1)).abs().sum()/2)
    old_close=torch.isclose(full,stream,atol=1e-4,rtol=1e-4)
    passed=float(diff.max())<=1e-3 and total_variation<=1e-4 and int(full.argmax())==int(stream.argmax()) and own_ids==native_ids
    result=dict(passed=passed,full_input_tokens=ids.numel(),positions_exact=True,input_ids_exact=True,
        max_abs=float(diff.max()),probability_total_variation=total_variation,
        legacy_close_1e4=bool(old_close.all()),legacy_violations=int((~old_close).sum()),
        native_generated_ids=native_ids,actual_chunk512_generated_ids=own_ids,
        first_argmax_equal=int(full.argmax())==int(stream.argmax()),
        thresholds=dict(max_abs=1e-3,total_variation=1e-4,generated_ids_exact=True))
    torch.save(dict(native=full,chunk512=stream),out/'qualification_logits.pt')
    save(out/'qualification.json',result)
    assert passed,'7B native qualification failed; no automatic retry'

def main(out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_CPUS_PER_TASK')=='4'
    assert transformers.__version__=='5.2.0'
    assert not out.exists() and out.resolve().is_relative_to((ROOT/'runs').resolve())
    for rel,digest in read(PREP/'execution_lock.json').items():assert sha(ROOT/rel)==digest,rel
    assert read(PREP/'CPU_cluster.json')['passed'] and read(PREP/'CPU_local.json')['passed']
    assert read(PREP/'asset_completed.json')['files']==15
    # Model SHA is verified immediately before submission; no network or downloads here.
    out.mkdir();start=time.monotonic();count=dict(answers=0,pairs=0,vision=0,tokens=0,qualification_tokens=0)
    def guard():
        if time.monotonic()-start>57*60:raise TimeoutError('57-minute internal stop')
        if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>64*2**20:raise RuntimeError('result byte cap')
        log=ROOT/'runs'/('7b-memory-slurm-'+os.environ['SLURM_JOB_ID']+'.log')
        if log.exists() and log.stat().st_size>16*2**20:raise RuntimeError('log byte cap')
    try:
        torch.set_num_threads(1);assert torch.cuda.device_count()==1
        device=torch.cuda.get_device_properties(0)
        save(out/'device.json',dict(name=device.name,total_memory=device.total_memory))
        assert 'A800' in device.name and device.total_memory>=64*2**30,'requires the budgeted A800 with sufficient memory'
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        model,loading=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,
            local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa',output_loading_info=True)
        save(out/'loading_info.json',loading)
        assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
        model.eval().requires_grad_(False)
        assert all(p.dtype==torch.float32 for p in model.parameters())
        assert sum(p.numel() for p in model.parameters())==8292166656
        processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        versions={n:p._version for n,p in model.named_parameters()}
        save(out/'loaded.json',dict(seconds=time.monotonic()-start,parameters=sum(p.numel() for p in model.parameters()),
            model_class=type(model).__module__+'.'+type(model).__name__,transformers=transformers.__version__))
        tasks=read(PREP/'profile.json');traces=read(PREP/'traces.json');groups={};qualified=False
        for q in tasks:groups.setdefault((q['suite'],q['source_id']),[]).append(q)
        with torch.no_grad(),(out/'answers.jsonl').open('x',encoding='utf-8') as f:
            for (suite,source),qs in sorted(groups.items()):
                qs.sort(key=lambda q:(q['query_time_s'],q['id']))
                spec=traces[suite+'|'+source];path=Path(spec['media_path'])
                if not path.is_absolute():path=ROOT/path
                assert sha(path)==qs[0]['media_sha256']
                source_start=time.monotonic();source_answers=[]
                memory=Memory();frames=deque(maxlen=8);index=0
                def commit(first,second,padding=False):
                    nonlocal index
                    guard()
                    if suite=='D18':
                        for packet,ref in zip((first,second),spec['sampled_trace'][2*index:2*index+2]):
                            assert packet.frame_id==ref['frame_id'] and abs(packet.timestamp-ref['pts_s'])<1e-6
                    batch=pair_inputs(processor,[first.rgb,second.rgb],device='cuda')
                    features=model.get_video_features(**batch,return_dict=True).pooler_output[0]
                    assert features.shape==(TOKENS,3584)
                    memory.append(features,index,first.timestamp,second.timestamp)
                    frames.extend((first.rgb,second.rgb));index+=1
                    count['pairs']+=1;count['vision']+=1
                driver=CausalPairs(path,commit)
                try:
                    for q in qs:
                        state=driver.advance(q['query_time_s'])
                        if suite!='D18':
                            expected=dict(spec['states'][q['id']]);expected['maximum_pending_RGB_bytes']=280*392*3
                            assert state==expected,(state,expected)
                        if not qualified:
                            qualify(model,processor,memory,q,frames,out,guard,count)
                            qualified=True
                        for arm in ARMS:
                            guard();torch.cuda.synchronize();begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
                            result=answer(model,processor,memory,q,arm,guard)
                            torch.cuda.synchronize()
                            result.update(id=q['id'],suite=suite,source_id=source,arm=arm,
                                answer_seconds=time.monotonic()-begin,peak_allocated=torch.cuda.max_memory_allocated(),
                                profile_only=True,shared_reference_archive=True,bank_budget=1120,recent_budget=560)
                            source_answers.append(dict(id=q['id'],arm=arm,seconds=result['answer_seconds'],
                                input_tokens=result['input_tokens'],peak_allocated=result['peak_allocated']))
                            assert all(n['end']<=q['query_time_s']+1e-6 for n in result['nodes'])
                            f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
                            count['answers']+=1;count['tokens']+=len(result['generated_ids'])
                            print(json.dumps(dict(id=q['id'],arm=arm,seconds=time.monotonic()-start,**count)),flush=True)
                    with (out/'source_costs.jsonl').open('a',encoding='utf-8') as sf:
                        sf.write(json.dumps(dict(suite=suite,source_id=source,pairs=index,
                            seconds=time.monotonic()-source_start,answers=source_answers,
                            shared_reference_archive=True,standalone_end_to_end=False))+'\n')
                finally:driver.close()
                del memory,frames;gc.collect();torch.cuda.empty_cache()
        assert count['answers']==20 and count['tokens']<=2560 and count['pairs']==561 and count['vision']<=564
        assert count['tokens']+count['qualification_tokens']<=2568
        assert all(p._version==versions[n] and p.grad is None for n,p in model.named_parameters())
        save(out/'completed.json',dict(**count,seconds=time.monotonic()-start,profile_only=True,updates=0))
    except BaseException as exc:
        save(out/'failed.json',dict(error=repr(exc),traceback=traceback.format_exc(),**count));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
