"""Independently audit collected real long-video outputs with the frozen parser."""
from bisect import bisect_right
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'repos/online-memory'
ARMS=('M0','M2','M2_A06','LS_shared384_v1','reservoir4','text_only')
METRICS=('strict_correct','content_correct','format_compliant')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def prefix_digest(frames):
    return hashlib.sha256(b''.join((json.dumps(f,sort_keys=True,separators=(',',':'))+'\n').encode()
                                    for f in frames)).hexdigest()


def main():
    job=sys.argv[1]
    require(job.isdigit(),'numeric job ID required')
    run=ROOT/('runs/long-video-gpu-'+job)
    states=[]
    for path in sorted((ROOT/'logs').glob('cpu-release-20261021-long-video-gpu-watch-*.json')):
        receipt=read(path)
        if receipt['returncode']!=0:continue
        for line in receipt['stdout'].splitlines():
            row=json.loads(line)
            if row.get('kind')=='command' and row['args'][0]=='sacct':
                states.extend(s.split('|') for s in row['stdout'].splitlines() if s.split('|')[0]==job)
    require(states and states[-1][2:4]==['COMPLETED','0:0'],'Slurm completion0:0 required')
    plan=read(ROOT/'reports/long-video-gpu-plan-20261021-01.json')
    spec=read(run/'gpu_protocol.json')
    require(sha(run/'gpu_protocol.json')==plan['GPU_protocol_sha256'],'approved GPU protocol changed')
    require(all(spec[k]==v for k,v in plan['profile'].items()),'GPU fixed settings changed')
    for name,expected in plan['CPU_artifacts_sha256'].items():
        require(sha(run/name)==expected,'CPU/input artifact changed: '+name)
    for name,expected in read(run/'source_lock.json').items():
        require(sha(REPO/name)==expected,'CPU-executed source changed: '+name)
    for name,expected in spec['extension_sha256'].items():
        require(sha(REPO/name)==expected,'GPU-executed extension changed: '+name)
    require(not (run/'long_video_gpu_failure.json').exists() and not (run/'long_video_score_failure.json').exists(),
            'failure receipt present')
    done=read(run/'gpu/completed.json')
    require(done['status']=='complete' and done['pretrained_loads']==1
            and done['GPU_protocol_sha256']==sha(run/'gpu_protocol.json')
            and done['CPU_preflight_sha256']==sha(run/'cpu/preflight.json'),'completion proof binding')
    require(done['environment']==read(run/'cpu/preflight.json')['environment'],'CPU/GPU environment changed')
    require(read(run/'gpu/loaded.json')['pretrained_loads']==1
            and read(run/'gpu/load_started.json')['maximum_pretrained_loads']==1,'real load count receipts')
    test_log=run/'gpu/extension_contracts.txt'
    require(sha(test_log)==done['extension_contract_log_sha256'] and done['extension_contract_tests']==22,
            'extension contract receipt mismatch')
    import re
    test_text=test_log.read_text(encoding='utf-8')
    require(re.findall(r'^Ran (\d+) tests? in ',test_text,re.M)==['22']
            and re.findall(r'^OK$',test_text,re.M)==['OK']
            and not re.search(r'^FAILED|^ERROR:|^FAIL:',test_text,re.M),'new contract suite did not pass')
    require(set(done['output_sha256'])=={'predictions.jsonl','events.jsonl'},'raw output set')
    for name,expected in done['output_sha256'].items():
        require(sha(run/'gpu'/name)==expected,'worker/raw output hash drift: '+name)
    queries=read(run/'requests.json')
    byid={q['id']:q for q in queries}
    sources={s['source_path']:s for s in read(run/'sources.json')}
    snapshots={s['query_id']:s for s in read(run/'snapshots.json')}
    require(len(byid)==len(queries)==34 and len(sources)==6 and set(snapshots)==set(byid),'input denominator')
    predictions=rows(run/'gpu/predictions.jsonl')
    require(len(predictions)==204 and {(p['id'],p['arm']) for p in predictions}=={(q,a) for q in byid for a in ARMS},
            'complete unique answer grid')
    appends=defaultdict(list);resources={};query_events={}
    maximum_tokens=defaultdict(int);maximum_bytes=defaultdict(int)
    for event in rows(run/'gpu/events.jsonl'):
        source=event['source_path']
        require(source in sources,'unknown event source')
        if event['event']=='append':
            arm=event['arm'];state=event['state']
            require(arm in ARMS[:-1],'unknown visual bank')
            require(state['valid'] and not state['closed'] and not state['full_history_retained']
                    and not state['language_cache_retained'],'invalid retained state')
            require(state['visual_tokens']<=384 and len(state['nodes'])<=8 and state['metadata_bytes']<=32768
                    and state['resident_bytes']==state['tensor_bytes']+state['detector_bytes']
                    and state['resident_bytes']<=8*394560+8192,'bank resident bound')
            appends[source,arm].append(event)
            maximum_tokens[arm]=max(maximum_tokens[arm],state['visual_tokens'])
            maximum_bytes[arm]=max(maximum_bytes[arm],state['resident_bytes'])
        elif event['event']=='resources':
            key=(source,event['pair'])
            require(key not in resources,'duplicate process resource row')
            resources[key]=event
        elif event['event']=='query':
            require(event['id'] not in query_events and event['all_banks_unchanged'],'query isolation or duplicate')
            query_events[event['id']]=event
        else:raise ValueError('unknown event type')
    require(set(appends)=={(s,a) for s in sources for a in ARMS[:-1]} and set(query_events)==set(byid),
            'source/bank/query event grid')
    expected_counts={}
    for source,metadata in sources.items():
        frames=metadata['frames']
        horizon=max(q['query_time_s'] for q in queries if q['source_path']==source)
        received=bisect_right([f['pts_s'] for f in frames],horizon)
        count=received//2
        expected_counts[source]=(received,count)
        for arm in ARMS[:-1]:
            stream=appends[source,arm]
            require(len(stream)==count,'continuous prefix append count')
            for i,event in enumerate(stream):
                a,b=frames[2*i:2*i+2]
                require(event['pair']==i and event['frame_ids']==[a['frame_id'],b['frame_id']]
                        and event['begin']==a['pts_s'] and event['end']==b['pts_s'] and event['padded_tail'] is False
                        and event['state']['pairs_received']==i+1,'causal pair identity or padding mismatch')
                ns=event['state']['nodes']
                require(sum(n['tokens'] for n in ns)==event['state']['visual_tokens'],'node token accounting')
                if arm in ('M2','M2_A06','LS_shared384_v1'):
                    require(ns[0]['first']==0 and ns[-1]['last']==i
                            and all(a['last']+1==b['first'] for a,b in zip(ns,ns[1:]))
                            and sum(n['original_pairs'] for n in ns)==i+1
                            and sum(n['mass'] for n in ns)==48*(i+1),'history provenance or mass accounting')
        for i in range(count):
            record=resources[source,i]
            total=sum(appends[source,a][i]['state']['resident_bytes'] for a in ARMS[:-1])
            require(record['combined_bank_bytes']==total,'combined bank bytes accounting')
    require(sum(n for n,c in expected_counts.values())==2726
            and sum(c for n,c in expected_counts.values())==1360
            and len(resources)==1360,'complete actual sampling grid')
    identities={}
    for p in predictions:
        q=byid[p['id']];source=q['source_path'];frames=sources[source]['frames']
        prefix=p['prefix'];n=bisect_right([f['pts_s'] for f in frames],q['query_time_s']);count=n//2
        expected=dict(arrived_frames=n,committed_frames=2*count,committed_pairs=count,
            pending_frame=frames[n-1] if n%2 else None,arrived_frames_sha256=prefix_digest(frames[:n]),
            committed_frames_sha256=prefix_digest(frames[:2*count]),
            historical_committed_frames_sha256=hashlib.sha256(json.dumps(frames[:2*count],sort_keys=True).encode()).hexdigest(),
            committed_through_s=frames[2*count-1]['pts_s'],observation_age_s=q['query_time_s']-frames[2*count-1]['pts_s'],
            query_padding=False)
        require(prefix==expected and p['source_path']==source and p['query_time_s']==q['query_time_s'],
                'prediction exact causal prefix mismatch')
        query_event=query_events[p['id']]
        require(all(query_event[k]==v for k,v in expected.items()),'query log/prediction prefix mismatch')
        old=snapshots[p['id']]
        require((old['received_frames'],old['committed_pairs'],old['committed_frames_sha256'],old['pending_frame_id'])
                ==(n,count,expected['historical_committed_frames_sha256'],frames[n-1]['frame_id'] if n%2 else None),
                'historical sampling snapshot mismatch')
        require(p['status']=='ok' and isinstance(p['text'],str) and 0<len(p['generated_ids'])<=8
                and all(type(i) is int and i>=0 for i in p['generated_ids']),'answer status/token bound')
        require(isinstance(p['input_ids_sha256'],str) and len(p['input_ids_sha256'])==64
                and p['input_tokens']<=8192 and p['context_tokens']<=8192,'language input metadata')
        if p['arm']!='text_only':
            state=appends[source,p['arm']][count-1]['state']
            require(p['memory_unchanged'] and p['node_metadata']==state['nodes']
                    and p['visual_tokens']==state['visual_tokens']
                    and p['pair_indices']==[n['first'] for n in state['nodes']],'readout bank/prefix mismatch')
            key=(source,p['arm'],count)
            require(key not in identities or identities[key]==p['node_digest'],'same-prefix readout node digest drift')
            identities[key]=p['node_digest']
        else:require(p['visual_tokens']==0 and p['pair_indices']==[],'text control reads visual bank')
    total_tokens=sum(len(p['generated_ids']) for p in predictions)
    require(done['main_answers']==204 and done['generated_tokens']==total_tokens<=1632
            and done['source_pairs_encoded']==done['visual_forward_calls']==1360
            and done['reset_checks']==30 and done['all_QA_banks_unchanged'],'real completion counts')
    require(len(done['sources'])==6 and {s['source_path'] for s in done['sources']}==set(sources),'source completion grid')
    for item in done['sources']:
        n,c=expected_counts[item['source_path']]
        require(item['sampled_frames']==n and item['encoded_pairs']==c and item['unmatched_final_frame']==n%2
                and item['reset_checks']==5 and item['all_QA_banks_unchanged'],'source reset or coverage')
    require(done['peak_combined_bank_bytes']==max(r['combined_bank_bytes'] for r in resources.values())
            and done['peak_GPU_allocated']<=24*2**30 and done['peak_RSS_bytes']<=20*2**30
            and done['seconds']<=2550,'peak resources/worker budget')
    scorer_path=REPO/'experiments/36tar/memory-eval-2b/bundle/experiments/2b-short-event-v1/scoring.py'
    module_spec=importlib.util.spec_from_file_location('_independent_long_video_scoring',scorer_path)
    scoring=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(scoring)
    catalog={r['id']:r for r in read(run/'scoring_ONLY.json')}
    groups=dict(original28_raw_labels=[q for q in byid if q.startswith('ovo:')],
        historical_supported16=[q for q in byid if catalog[q]['declared_admission']=='supported'],
        historical_ambiguous11=[q for q in byid if catalog[q]['declared_admission']=='ambiguous'],
        historical_cutoff1=[q for q in byid if catalog[q]['declared_admission']=='not_established_at_committed_cutoff'],
        manual6_source_declared=[q for q in byid if q.startswith('local_longstream_v1:')])
    require([len(v) for v in groups.values()]==[28,16,11,1,6],'label strata denominator')
    details=read(run/'score/details.json');summary=read(run/'score/summary.json')
    bydetail={(r['id'],r['arm']):r for r in details}
    require(len(details)==204 and set(bydetail)=={(q,a) for q in byid for a in ARMS},'scoring grid')
    computed={}
    for p in predictions:
        key=(p['id'],p['arm']);q=byid[p['id']]
        measure=scoring.measure(p['text'],len(q['options']),catalog[p['id']]['label_index'])
        require(all(bydetail[key][k]==v for k,v in measure.items()),'independent frozen score mismatch')
        computed[key]=measure
    tables={};paired={}
    for name,ids in groups.items():
        tables[name]={a:dict(questions=len(ids),**{m:sum(computed[q,a][m] for q in ids) for m in METRICS}) for a in ARMS}
        paired[name]={}
        for ref in ('M0','M2','M2_A06','reservoir4'):
            paired[name][ref]={}
            for arm in (a for a in ARMS if a not in (ref,'text_only')):
                paired[name][ref][arm]={}
                for metric in METRICS[:2]:
                    gains=[q for q in ids if not computed[q,ref][metric] and computed[q,arm][metric]]
                    losses=[q for q in ids if computed[q,ref][metric] and not computed[q,arm][metric]]
                    paired[name][ref][arm][metric]=dict(corrections=gains,new_losses=losses,net=len(gains)-len(losses))
    require(tables==summary['tables'] and paired==summary['paired_comparisons'],'independent grouped/paired mismatch')
    per_source={}
    for source in sorted(sources):
        per_source[source]={}
        for name,ids in groups.items():
            selected=[q for q in ids if byid[q]['source_path']==source]
            if selected:per_source[source][name]={a:dict(questions=len(selected),strict_correct=sum(
                computed[q,a]['strict_correct'] for q in selected)) for a in ARMS}
    require(per_source==summary['per_source'],'independent per-source score mismatch')
    review=read(run/'score/review_without_predictions.json')
    require(len(review)==34 and {r['id'] for r in review}==set(byid)
            and all(r['review_status']=='pending' and r['semantic_answer_judgment'] is None
                    and not {'label_index','answer_index','text','predictions'}&set(r) for r in review),'human review scope')
    require(summary['confirmed_accuracy'] is None and summary['confirmed_accuracy_denominator']==0
            and summary['human_review']=='pending' and summary['independent_holdout'] is False,'unreviewed accuracy claim')
    require(summary['GPU_protocol_sha256']==sha(run/'gpu_protocol.json')
            and summary['completed_sha256']==sha(run/'gpu/completed.json')
            and summary['raw_output_hashes']==done['output_sha256'],'score output binding')
    changed=set()
    for name in groups:
        for ref in ('M0','M2','M2_A06','reservoir4'):
            item=paired[name][ref]['LS_shared384_v1']['strict_correct']
            changed.update(item['corrections']+item['new_losses'])
    pred_bykey={(p['id'],p['arm']):p for p in predictions}
    cases=[]
    for qid in sorted(changed):
        q=byid[qid];label=catalog[qid];prefix=pred_bykey[qid,'M2']['prefix']
        frames=sources[q['source_path']]['frames'];hits=[]
        for time_s in label['declared_witness_times']:
            index=min(range(len(frames)),key=lambda i:abs(frames[i]['pts_s']-time_s))
            if abs(frames[index]['pts_s']-time_s)<=.05:hits.append(index//2)
        boundary=max(0,prefix['committed_pairs']-8)
        age='no_declared_witness' if not hits else 'declared_witness_old_only' if max(hits)<boundary else 'declared_witness_recent_or_cross'
        cases.append(dict(id=qid,question=q['question'],options=q['options'],truth=chr(65+label['label_index']),
            source_path=q['source_path'],query_time_s=q['query_time_s'],declared_admission=label['declared_admission'],
            declared_witness_pairs=hits,evidence_position_proxy=age,long_memory_necessity='pending',
            semantic_review='pending',answers={a:dict(text=pred_bykey[qid,a]['text'],
                strict_correct=computed[qid,a]['strict_correct'],leading_correct=computed[qid,a]['content_correct']) for a in ARMS},
            node_metadata={a:pred_bykey[qid,a]['node_metadata'] for a in ARMS[:-1]}))
    result=dict(status='independently_verified_real_long_video_development_comparison',job_id=int(job),CPU_job=217847,
        slurm_state=states[-1][2],exit_code=states[-1][3],elapsed=states[-1][4],main_answers=204,
        sampled_frames=2726,complete_pairs=1360,visual_forward_calls=1360,pretrained_loads=1,generated_tokens=total_tokens,
        query_snapshots_matched=34,reset_checks=30,new_extension_contracts_passed=22,
        maximum_visual_tokens=dict(maximum_tokens),maximum_resident_bytes=dict(maximum_bytes),
        peak_combined_bank_MiB=done['peak_combined_bank_bytes']/2**20,
        peak_GPU_allocated_GiB=done['peak_GPU_allocated']/2**30,peak_GPU_reserved_GiB=done['peak_GPU_reserved']/2**30,
        peak_RSS_GiB=done['peak_RSS_bytes']/2**30,worker_seconds=done['seconds'],stage_seconds=done['stage_seconds'],
        scores=tables,paired_comparisons=paired,per_source=per_source,LS_changed_cases=cases,
        format_noncompliance=[dict(id=p['id'],arm=p['arm'],text=p['text']) for p in predictions
            if not computed[p['id'],p['arm']]['format_compliant']],
        historical_teammate_answer_identity_claim=False,independent_holdout=False,
        confirmed_accuracy=None,human_content_review='pending',stable_long_memory_improvement_established=False,
        artifacts_sha256={p.relative_to(run).as_posix():sha(p) for p in sorted(run.rglob('*')) if p.is_file()})
    target=ROOT/('reports/job-'+job+'-long-video-GPU-result-20261021.json')
    with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:result[k] for k in ('status','job_id','elapsed','generated_tokens','scores',
        'peak_GPU_allocated_GiB','peak_RSS_GiB')},ensure_ascii=False))


if __name__=='__main__':main()
