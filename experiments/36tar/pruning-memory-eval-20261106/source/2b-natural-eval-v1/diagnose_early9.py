"""One fixed four-cell compression/interference diagnostic; not a new selector."""
import argparse,gc,json,os,time
from pathlib import Path
from datetime import datetime,timezone
from runtime_bridge import ROOT,HERE,MODEL,REPO,ACTIVE,runtime
from evaluate import read,save,sha,check
ARMS=['old_full','old_merged','old_full_recent4','old_merged_recent4','recent4','text_only','full_replay']
def selections(env,nodes,first):
    legacy=env['load']();old=[n.copy() for n in nodes[first:first+4]];recent=[n.copy() for n in nodes[-4:]]
    if len(old)!=4 or old[-1].last>=recent[0].first:raise ValueError('four complete old pairs before recent4 required')
    root=old[0].copy()
    for n in old[1:]:root=legacy.forest.merge(root,n,within=True)
    return dict(old_full=old,old_merged=[root],old_full_recent4=[*old,*recent],old_merged_recent4=[root,*recent],recent4=recent,text_only=[],full_replay=nodes)
def verify(run):
    p=read(run/'protocol.json');base=ROOT/'runs/2b-natural-eval-inputs-20261020-01';check(base)
    if sha(run/'requests.json')!=p['request_sha256'] or sha(HERE/'diagnose_early9.py')!=p['driver_sha256'] or sha(base/'source_lock.json')!=p['base_source_lock_sha256']:raise ValueError('diagnostic source/input drift')
    identity=read(base/'cpu_memory/preflight.json')['identity']
    for source,key in [(REPO/'qwen35_local/modeling_qwen3_5.py','original_model_sha256'),(ACTIVE/'qwen35_local/modeling_qwen3_5.py','stream_model_sha256'),(ACTIVE/'streaming/online.py','online_sha256'),(ACTIVE/'streaming/hybrid.py','hybrid_sha256'),(MODEL/'DOWNLOAD_MANIFEST.json','manifest_sha256')]:
        if sha(source)!=identity[key]:raise ValueError('base implementation/model identity drift')
    rs=read(run/'requests.json')
    if len(rs)!=9 or len({r['id'] for r in rs})!=9:raise ValueError('all fixed9 required')
    original={r['id']:r for r in read(base/'requests.json')}
    for r in rs:
        if set(r)!={'request','old_first_pair','id'} or r['request']!=original[r['id']]:raise ValueError('diagnostic request drift/leak')
    return p,rs
def cpu(run):
    p,rs=verify(run);env=runtime('memory');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    if torch.cuda.is_available():raise ValueError('CPU preflight only')
    proc=env['AutoProcessor'].from_pretrained(MODEL,local_files_only=True);builder=env['PairInputBuilder'](proc,canvas=(192,256),resize_mode='letterbox')
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](proc),attn_implementation='sdpa')).eval()
    legacy=env['load']();checks=[]
    from capability_pilot import answer
    from fixed_stage2 import node_digest
    with torch.inference_mode(),env['fp32_reference']():
        for i,row in enumerate(rs):
            fs=row['request']['model_visible']['frames'];nodes=[]
            for j in range(0,len(fs),2):
                a,b=fs[j],fs[min(j+1,len(fs)-1)];nodes.append(legacy.forest.raw(legacy.features.raw_block(torch.randn(48,32),j//2,a['pts_s'],b['pts_s'])))
            choices=selections(env,nodes,row['old_first_pair']);q=env['prompt'](row['request']['model_visible'])
            for arm,ns in choices.items():
                before=node_digest(ns)
                if ns:emb,pos,ids,meta=env['context'](model,builder,ns,q,limits=env['Limits'](8192,6144,128),max_new_tokens=8)
                if i==0:
                    result=answer(env,model,builder,ns,q)
                    if not result['generated_ids']:raise AssertionError('tiny generation empty')
                if node_digest(ns)!=before:raise AssertionError('query changed bank')
            checks.append(dict(id=row['id'],old_pairs=list(range(row['old_first_pair'],row['old_first_pair']+4)),recent_pairs=list(range(len(nodes)-4,len(nodes))),visual_tokens={a:sum(len(n.features) for n in ns) for a,ns in choices.items()}))
    save(run/'cpu_preflight.json',dict(status='passed',questions=9,contexts_checked=63,tiny_answers=7,real_model_loads=0,protocol_sha256=sha(run/'protocol.json'),checks=checks))
    print(json.dumps(dict(status='passed',contexts=63,tiny_answers=7)),flush=True)
def worker(run,out):
    p,rs=verify(run);proof=read(run/'cpu_preflight.json')
    if proof['status']!='passed' or proof['protocol_sha256']!=sha(run/'protocol.json'):raise ValueError('diagnostic CPU proof')
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('allocated Slurm GPU required')
    if datetime.now(timezone.utc)>=datetime.fromisoformat(p['deadline_UTC']):raise TimeoutError('window ended')
    out.mkdir(exist_ok=False);env=runtime('memory');torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    free,_=torch.cuda.mem_get_info();ram=env['psutil'].virtual_memory().available
    save(out/'guard.json',dict(free_gpu_bytes=free,free_ram_bytes=ram,job_id=os.environ['SLURM_JOB_ID'],source_sha256=sha(env['actual_source'])))
    if free<16*2**30 or ram<12*2**30:raise MemoryError('16GiB GPU/12GiB RAM guard')
    start=time.monotonic();model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(MODEL,local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    save(out/'loaded.json',dict(model_loads=1,load_seconds=time.monotonic()-start))
    if model.config.text_config.hidden_size!=2048 or next(model.parameters()).dtype!=torch.float32:raise ValueError('2B FP32 required')
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(MODEL,local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    from fixed_stage2 import decode,encode,node_digest
    from capability_pilot import answer
    grouped={}
    for row in rs:grouped.setdefault(row['request']['media_path'],[]).append(row)
    written=0;generated=0;timelines=[]
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as f,torch.inference_mode(),env['fp32_reference']():
        for rows in grouped.values():
            packets,_=decode(env,[x['request'] for x in rows]);specs=rows[0]['request']['model_visible']['frames'];nodes=encode(env,model,builder,packets,specs)
            for row in rows:
                choices=selections(env,nodes,row['old_first_pair']);hashes={a:node_digest(ns) for a,ns in choices.items()}
                timelines.append(dict(id=row['id'],selections={a:[n.metadata() for n in ns] for a,ns in choices.items()}))
                for arm,ns in choices.items():
                    if time.monotonic()-start>570 or datetime.now(timezone.utc)>=datetime.fromisoformat(p['deadline_UTC']):raise TimeoutError('diagnostic wall/window bound')
                    begin=time.monotonic();a=answer(env,model,builder,ns,env['prompt'](row['request']['model_visible']))
                    entry=dict(id=row['id'],arm=arm,status='ok',**a,seconds=time.monotonic()-begin)
                    f.write(json.dumps(entry,ensure_ascii=False)+'\n');f.flush();written+=1;generated+=len(a['generated_ids'])
                    if any(node_digest(choices[a])!=h for a,h in hashes.items()):raise AssertionError('query mutated evidence')
                    if torch.cuda.memory_allocated()>24*2**30 or env['psutil'].Process().memory_info().rss>20*2**30:raise MemoryError('allocation/RSS guard')
                    print(json.dumps(dict(done=written,arm=arm)),flush=True)
                del choices
            del nodes,packets;gc.collect();torch.cuda.empty_cache()
    if written!=63:raise AssertionError('all9x7 required')
    save(out/'timelines.json',timelines);save(out/'completed.json',dict(status='complete',records=written,generated_tokens=generated,model_loads=1,seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),query_bank_isolation=True,protocol_sha256=sha(run/'protocol.json')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','worker']);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path);a=p.parse_args()
    if not a.run.resolve().is_relative_to((ROOT/'runs').resolve()):raise ValueError('project run required')
    if a.mode=='cpu':cpu(a.run)
    else:worker(a.run,a.out)
