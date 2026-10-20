"""R0 one-round all171 comparison. No change to storage or pretrained weights."""
import argparse,gc,os,time,json
from pathlib import Path
from datetime import datetime,timezone
from runtime_bridge import ROOT,HERE,MODEL,REPO,ACTIVE,runtime
from evaluate import read,save,sha,check
from temporal_route import select
ARMS=['all_memory','long_only','recent4','temporal_route']
def verify(run):
    p=read(run/'protocol.json');base=ROOT/'runs/2b-natural-eval-inputs-20261020-01';_,groups=check(base)
    if sha(base/'source_lock.json')!=p['base_source_lock_sha256'] or sha(base/'requests.json')!=p['requests_sha256']:raise ValueError('base inputs/sources changed')
    for name,h in p['new_sources'].items():
        if sha(HERE/name)!=h:raise ValueError('candidate source changed')
    identity=read(base/'cpu_memory/preflight.json')['identity']
    for src,key in [(REPO/'qwen35_local/modeling_qwen3_5.py','original_model_sha256'),(ACTIVE/'qwen35_local/modeling_qwen3_5.py','stream_model_sha256'),(ACTIVE/'streaming/online.py','online_sha256'),(ACTIVE/'streaming/hybrid.py','hybrid_sha256'),(MODEL/'DOWNLOAD_MANIFEST.json','manifest_sha256')]:
        if sha(src)!=identity[key]:raise ValueError('model identity drift')
    return p,groups
def cpu(run):
    p,groups=verify(run);env=runtime('memory');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    if torch.cuda.is_available():raise ValueError('CPU only')
    from capability_pilot import answer
    from fixed_stage2 import policies,node_digest
    legacy=env['load']();nodes=[legacy.forest.raw(legacy.features.raw_block(torch.randn(48,32),i,2.*i,2.*i+1)) for i in range(45)]
    bank=policies(env,nodes)[0]['M2'];before=node_digest(bank)
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(MODEL,local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](builder.processor),attn_implementation='sdpa')).eval()
    count=0;trigger=0;tiny={}
    with torch.inference_mode(),env['fp32_reference']():
        for rows in groups.values():
            for r in rows:
                selections,meta=select(bank,r['model_visible']['question']);trigger+=meta['used_long_only']
                for arm,ns in selections.items():
                    q=env['prompt'](r['model_visible']);emb,pos,ids,m=env['context'](model,builder,ns,q,limits=env['Limits'](8192,384,128),max_new_tokens=8)
                    if m['visual_tokens']>384 or arm in ['long_only','recent4'] and m['visual_tokens']>192:raise AssertionError('read budget')
                    if count==0:tiny[arm]=answer(env,model,builder,ns,q)['generated_ids']
                if node_digest(bank)!=before:raise AssertionError('bank mutated')
                if node_digest(selections['temporal_route'])!=node_digest(selections['long_only' if meta['used_long_only'] else 'all_memory']):raise AssertionError('routing mismatch')
                count+=1
    for q,expected in [('What happened AFTER she sat?',True),('What was there at the beginning?',True),('What happened in the middle?',True),('How many cats are visible?',False)]:
        if select(bank,q)[1]['has_temporal_cue']!=expected:raise AssertionError('cue contract')
    fallback,fmeta=select(bank[-3:],'What happened before this?')
    if not fmeta['empty_long_fallback'] or node_digest(fallback['temporal_route'])!=node_digest(bank[-3:]):raise AssertionError('empty old fallback')
    save(run/'cpu_preflight.json',dict(status='passed',contexts=684,questions=count,tiny_answers=4,synthetic_bank_temporal_triggers=trigger,bank_unchanged=True,empty_history_fallback=True,pretrained_loads=0,protocol_sha256=sha(run/'protocol.json')))
    print(json.dumps(dict(status='passed',contexts=684,tiny_answers=4)),flush=True)
def worker(run,out):
    p,groups=verify(run);proof=read(run/'cpu_preflight.json')
    if proof['status']!='passed' or proof['protocol_sha256']!=sha(run/'protocol.json'):raise ValueError('CPU proof')
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('allocated Slurm GPU required')
    out.mkdir(exist_ok=False);env=runtime('memory');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    if datetime.now(timezone.utc)>=datetime.fromisoformat(p['deadline_UTC']):raise TimeoutError('window ended')
    free,_=torch.cuda.mem_get_info();ram=env['psutil'].virtual_memory().available
    save(out/'guard.json',dict(free_gpu_bytes=free,free_ram_bytes=ram,job_id=os.environ['SLURM_JOB_ID'],source_sha256=sha(env['actual_source'])))
    if free<16*2**30 or ram<12*2**30:raise MemoryError('GPU/RAM guard')
    start=time.monotonic();model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(MODEL,local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    save(out/'loaded.json',dict(model_loads=1,load_seconds=time.monotonic()-start))
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(MODEL,local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    from fixed_stage2 import decode,encode,policies,node_digest
    from capability_pilot import answer
    records=0;generated=0;traces=[];banks=[]
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as f,torch.inference_mode(),env['fp32_reference']():
        for media,rows in groups.items():
            packets,_=decode(env,rows);specs=rows[0]['model_visible']['frames'];nodes=encode(env,model,builder,packets,specs);exports,status,events=policies(env,nodes);bank=exports['M2'];before=node_digest(bank)
            banks.append(dict(media=media,state=status['M2'],events=events))
            for r in rows:
                choices,meta=select(bank,r['model_visible']['question']);traces.append(dict(id=r['id'],cohort=r['cohort'],**meta,selections={a:[n.metadata() for n in ns] for a,ns in choices.items()}))
                for arm,ns in choices.items():
                    if time.monotonic()-start>1170 or datetime.now(timezone.utc)>=datetime.fromisoformat(p['deadline_UTC']):raise TimeoutError('wall/window bound')
                    begin=time.monotonic();a=answer(env,model,builder,ns,env['prompt'](r['model_visible']))
                    if a['visual_tokens']>384 or arm in ['long_only','recent4'] and a['visual_tokens']>192:raise AssertionError('candidate read budget')
                    entry=dict(id=r['id'],cohort=r['cohort'],arm=arm,status='ok',**a,seconds=time.monotonic()-begin)
                    f.write(json.dumps(entry,ensure_ascii=False)+'\n');f.flush();records+=1;generated+=len(a['generated_ids'])
                    if node_digest(bank)!=before:raise AssertionError('query altered forest')
                    if torch.cuda.memory_allocated()>24*2**30 or env['psutil'].Process().memory_info().rss>20*2**30:raise MemoryError('allocation/RSS guard')
                print(json.dumps(dict(done=records,cohort=r['cohort'])),flush=True)
            del nodes,bank,exports,choices,packets;gc.collect();torch.cuda.empty_cache()
    if records!=684:raise AssertionError('fixed171x4 required')
    save(out/'routing_traces.json',traces);save(out/'banks.json',banks);save(out/'completed.json',dict(status='complete',records=records,generated_tokens=generated,model_loads=1,seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),query_bank_isolation=True,protocol_sha256=sha(run/'protocol.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','worker']);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path);a=p.parse_args()
    if not a.run.resolve().is_relative_to((ROOT/'runs').resolve()):raise ValueError('project-only run')
    if a.mode=='cpu':cpu(a.run)
    else:worker(a.run,a.out)
