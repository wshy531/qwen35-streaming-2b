"""One fixed natural-video evaluation, CPU preflight then Slurm-only workers."""
import argparse,gc,hashlib,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
from runtime_bridge import ROOT,HERE,REPO,ACTIVE,MODEL,runtime
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def check(run):
    protocol=read(run/'protocol.json')
    if sha(run/'requests.json')!=protocol['request_sha256']:raise ValueError('request drift')
    for rel,h in read(run/'source_lock.json').items():
        if sha(HERE/rel)!=h:raise ValueError('source drift '+rel)
    grouped={}
    rows=read(run/'requests.json')
    if len(rows)!=171 or len({r['id'] for r in rows})!=171:raise ValueError('fixed 171 IDs required')
    for r in rows:
        if set(r)!={'id','cohort','media_path','media_sha256','model_visible'} or set(r['model_visible'])!={'question','options','query_time_s','frames'}:raise ValueError('answer-blind request allowlist')
        grouped.setdefault(r['media_path'],[]).append(r)
    if len(grouped)!=67:raise ValueError('fixed67 media required')
    for rows in grouped.values():
        if any(r['model_visible']['frames']!=rows[0]['model_visible']['frames'] or r['model_visible']['query_time_s']!=rows[0]['model_visible']['query_time_s'] for r in rows):raise ValueError('same-source prefix mismatch')
    return protocol,grouped
def cpu(run,stage):
    protocol,groups=check(run);out=run/('cpu_'+stage);out.mkdir(exist_ok=False)
    env=runtime(stage);torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    if torch.cuda.is_available():raise ValueError('preflight CPU only')
    import fixed_stage2 as fixed
    from capability_pilot import answer
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(MODEL,local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    model=env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](**env['tiny_config'](builder.processor),attn_implementation='sdpa')).eval()
    maximum=0;count=0;pixels=0;framecount=0;finite=False
    with torch.inference_mode(),env['fp32_reference']():
        if stage=='native':
            for rows in groups.values():
                packets,metadata=fixed.decode(env,rows);specs=rows[0]['model_visible']['frames'];framecount+=len(specs)
                for j,r in enumerate(rows):
                    kwargs,delta,meta=fixed.native_inputs(env,model,builder,packets,specs,metadata,env['prompt'](r['model_visible']),'cpu')
                    maximum=max(maximum,meta['input_tokens']);count+=1
                    if j==0:
                        pairpixels=[]
                        for i in range(0,len(specs),2):
                            a,b=specs[i],specs[min(i+1,len(specs)-1)]
                            pair,_=builder.pair(packets[a['frame_id']],packets[b['frame_id']],0,0)
                            pairpixels.append(pair.pixel_values_videos)
                        if not torch.equal(kwargs.pixel_values_videos,torch.cat(pairpixels)):raise AssertionError('native/pair RGB preprocessing differs')
                        pixels+=1;del pairpixels,pair
                    if not finite:
                        model.model.rope_deltas=delta;_,first=env['complete'](model,builder.processor,dict(kwargs),1)
                        if not env['np'].isfinite(first).all():raise AssertionError('tiny prefill nonfinite')
                        finite=True
                    del kwargs,delta
                del packets;gc.collect()
                print(json.dumps(dict(cpu_sources=pixels,questions=count)),flush=True)
        else:
            legacy=env['load']();nodes=[]
            for i in range(45):nodes.append(legacy.forest.raw(legacy.features.raw_block(torch.randn(48,32),i,2.*i,2.*i+1)))
            exports,status,events=fixed.policies(env,nodes)
            choices=dict(full_pairwise_reference=nodes,**exports,text_only=[])
            longest=max([r for rows in groups.values() for r in rows],key=lambda r:builder.question_ids(env['prompt'](r['model_visible'])).shape[1])
            for arm,ns in choices.items():
                before=fixed.node_digest(ns);a=answer(env,model,builder,ns,env['prompt'](longest['model_visible']))
                if before!=fixed.node_digest(ns):raise AssertionError('query mutated bank')
                if arm in ['M0','M1','M2'] and a['visual_tokens']>384:raise AssertionError('candidate budget')
            wide=[legacy.forest.raw(legacy.features.raw_block(torch.randn(48,2048),i,2.*i,2.*i+1)) for i in range(12)]
            wide_exports,wide_status,_=fixed.policies(env,wide)
            if any(n.features.shape[1]!=2048 for ns in wide_exports.values() for n in ns):raise AssertionError('2B width lost')
            try:legacy.features.raw_block(torch.zeros(48,2049),0,0.,1.)
            except ValueError:pass
            else:raise AssertionError('width upper bound bypassed')
            for rows in groups.values():
                for r in rows:
                    if builder.question_ids(env['prompt'](r['model_visible'])).shape[1]>128:raise ValueError('full question suffix exceeds128')
            finite=True;count=171
    identity=dict(original_model_sha256=sha(REPO/'qwen35_local/modeling_qwen3_5.py'),stream_model_sha256=sha(ACTIVE/'qwen35_local/modeling_qwen3_5.py'),online_sha256=sha(ACTIVE/'streaming/online.py'),hybrid_sha256=sha(ACTIVE/'streaming/hybrid.py'),manifest_sha256=sha(MODEL/'DOWNLOAD_MANIFEST.json'))
    result=dict(status='passed',stage=stage,questions_checked=count,sources_pixels_equal=pixels,frames_checked=framecount,max_native_tokens=maximum,finite_tiny_generation=finite,pretrained_loads=0,protocol_sha256=sha(run/'protocol.json'),source_lock_sha256=sha(run/'source_lock.json'),identity=identity)
    if stage=='memory':result.update(query_bank_isolation=True,wide_2048_states=wide_status,old_algorithms_unchanged=True)
    save(out/'preflight.json',result);print(json.dumps({k:v for k,v in result.items() if k!='wide_2048_states'}),flush=True)
def worker(run,stage,out):
    protocol,groups=check(run)
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Slurm allocation required')
    for s in ['native','memory']:
        proof=read(run/('cpu_'+s)/'preflight.json')
        if proof['status']!='passed' or proof['protocol_sha256']!=sha(run/'protocol.json') or proof['source_lock_sha256']!=sha(run/'source_lock.json'):raise ValueError('CPU proof identity')
    identity=proof['identity']
    for path,key in [(REPO/'qwen35_local/modeling_qwen3_5.py','original_model_sha256'),(ACTIVE/'qwen35_local/modeling_qwen3_5.py','stream_model_sha256'),(ACTIVE/'streaming/online.py','online_sha256'),(ACTIVE/'streaming/hybrid.py','hybrid_sha256'),(MODEL/'DOWNLOAD_MANIFEST.json','manifest_sha256')]:
        if sha(path)!=identity[key]:raise ValueError('runtime identity drift')
    out.mkdir(exist_ok=False);env=runtime(stage);torch=env['torch'];torch.set_num_threads(1);torch.manual_seed(123)
    import fixed_stage2 as fixed
    from capability_pilot import answer
    free,_=torch.cuda.mem_get_info();ram=env['psutil'].virtual_memory().available
    save(out/'guard.json',dict(free_gpu_bytes=free,free_ram_bytes=ram,job=os.environ['SLURM_JOB_ID'],source=str(env['actual_source']),source_sha256=sha(env['actual_source'])))
    if free<16*2**30 or ram<12*2**30:raise MemoryError('preload 16GiB GPU /12GiB RAM guard')
    manifest=read(MODEL/'DOWNLOAD_MANIFEST.json')
    for item in manifest['files']:
        if (MODEL/item['name']).stat().st_size!=item['bytes']:raise ValueError('weight asset size drift')
    start=time.monotonic();deadline=datetime.fromisoformat(protocol['deadline_UTC'])
    model=env['Qwen3_5ForConditionalGeneration'].from_pretrained(MODEL,local_files_only=True,dtype=torch.float32,device_map='cuda',attn_implementation='sdpa').eval()
    if next(model.parameters()).dtype!=torch.float32 or model.config.text_config.hidden_size!=2048:raise ValueError('2B FP32 required')
    save(out/'loaded.json',dict(model_loads=1,load_seconds=time.monotonic()-start))
    builder=env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(MODEL,local_files_only=True),canvas=(192,256),resize_mode='letterbox')
    written=0;generated=0;timelines=[];encoded=0;peakrss=0
    with (out/'predictions.jsonl').open('x',encoding='utf-8') as f,torch.inference_mode(),env['fp32_reference']():
        for media,rows in groups.items():
            packets,metadata=fixed.decode(env,rows);specs=rows[0]['model_visible']['frames']
            if stage=='memory':
                nodes=fixed.encode(env,model,builder,packets,specs);encoded+=len(nodes)
                exports,status,events=fixed.policies(env,nodes);selections=dict(full_pairwise_reference=nodes,**exports,text_only=[])
                hashes={k:fixed.node_digest(ns) for k,ns in selections.items()}
                timelines.append(dict(media=media,cohort=rows[0]['cohort'],states=status,events=events,full_reference_tokens=sum(len(n.features) for n in nodes),full_reference_bytes=sum(n.metadata()['tensor_bytes'] for n in nodes),snapshot_tail_duplicate=bool(len(specs)%2)))
                save(out/'timelines.json',timelines)
            for r in rows:
                for arm in protocol['arms'][stage]:
                    if time.monotonic()-start>protocol['worker_seconds']-30 or datetime.now(timezone.utc)>=deadline:raise TimeoutError('fixed wall/window bound')
                    begin=time.monotonic()
                    try:
                        question=env['prompt'](r['model_visible'])
                        a=(fixed.native_answer(env,model,builder,packets,specs,metadata,question,[],arm) if stage=='native' else answer(env,model,builder,selections[arm],question))
                        entry=dict(id=r['id'],cohort=r['cohort'],arm=arm,status='ok',**a)
                    except (ValueError,FloatingPointError) as e:entry=dict(id=r['id'],cohort=r['cohort'],arm=arm,status='readout_error',text='',error=str(e))
                    entry['seconds']=time.monotonic()-begin;entry['input_frames_sha256']=hashlib.sha256(json.dumps(specs,sort_keys=True).encode()).hexdigest()
                    f.write(json.dumps(entry,ensure_ascii=False)+'\n');f.flush();written+=1;generated+=len(entry.get('generated_ids',[]))
                    if stage=='memory' and any(fixed.node_digest(ns)!=hashes[k] for k,ns in selections.items()):raise AssertionError('query changed stored bank')
                    peakrss=max(peakrss,env['psutil'].Process().memory_info().rss)
                    if torch.cuda.memory_allocated()>24*2**30:raise MemoryError('24GiB GPU allocation guard')
                    if peakrss>20*2**30:raise MemoryError('20GiB owned RSS guard')
                    print(json.dumps(dict(done=written,cohort=r['cohort'],arm=arm,status=entry['status'])),flush=True)
            if stage=='memory':del selections,exports,status,events,nodes
            del packets;gc.collect();torch.cuda.empty_cache()
    if written!=171*len(protocol['arms'][stage]):raise AssertionError('prediction completeness')
    save(out/'completed.json',dict(status='complete',stage=stage,records=written,generated_tokens=generated,actual_pair_encodings=encoded,model_loads=1,seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_sampled_RSS_bytes=peakrss,query_bank_isolation=stage=='memory',protocol_sha256=sha(run/'protocol.json')))
    del model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','worker']);p.add_argument('stage',choices=['native','memory']);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();run=a.run.resolve()
    if not run.is_relative_to((ROOT/'runs').resolve()):raise ValueError('project run path required')
    if a.mode=='cpu':cpu(run,a.stage)
    else:worker(run,a.stage,a.out.resolve())
