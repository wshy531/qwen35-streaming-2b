"""Independently audit collected replay evidence; stdlib and frozen scorer only."""
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'repos/online-memory'
ARMS=('M0','M2','reservoir4','text_only')

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(encoding='utf-8') as f: return [json.loads(s) for s in f if s.strip()]
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,message):
    if not ok: raise ValueError(message)

def main():
    jobid=sys.argv[1]
    require(jobid.isdigit(),'numeric job ID required')
    run=ROOT/('runs/gpu-replay-'+jobid)
    complete=read(run/'gpu/completed.json')
    protocol=read(run/'protocol.json')
    cpu=read(run/'cpu/preflight.json')
    summary=read(run/'score/summary.json')
    for name,value in protocol['input_hashes'].items():
        require(sha(run/name)==value,'protocol input hash drift: '+name)
    for rel,value in read(run/'source_lock.json').items():
        require(sha(REPO/rel)==value,'local/executed source drift: '+rel)
    require(complete['status']=='complete' and complete['pretrained_loads']==1,'incomplete/invalid load count')
    require(complete['protocol_sha256']==sha(run/'protocol.json')
            and complete['preflight_sha256']==sha(run/'cpu/preflight.json'),'completion binding')
    require(complete['environment']==cpu['environment'],'CPU/GPU environment drift')
    require(not (run/'gpu_failure.json').exists() and not (run/'score_failure.json').exists(),'failure receipt present')
    requests=read(run/'requests.json')
    byid={r['id']:r for r in requests}
    groups={r['media_sha256']:r['model_visible']['frames'] for r in requests}
    require(len(byid)==len(requests)==171 and len(groups)==67,'request denominator')
    pred=rows(run/'gpu/predictions.jsonl')
    expected={(r['id'],r['arm']):r for r in read(run/'expected_predictions.json')}
    keys={(r['id'],r['arm']) for r in pred}
    require(len(pred)==684 and keys=={(qid,arm) for qid in byid for arm in ARMS},'incomplete answer grid')
    for p in pred:
        ref=expected[p['id'],p['arm']]
        require(p['status']=='ok' and all(p['replay'].values()),'answer/replay failure')
        for key in ('generated_ids','input_ids_sha256','input_frames_sha256'):
            require(p[key]==ref[key],'independent historical comparison: '+key)
        frames=groups[byid[p['id']]['media_sha256']]
        require(p['input_frames_sha256']==hashlib.sha256(json.dumps(frames,sort_keys=True).encode()).hexdigest(),
                'prediction frame specification hash')
        if p['arm']!='text_only':
            require(p['node_metadata']==ref['node_metadata'] and p['memory_unchanged'],'bank replay/isolation')
    probes=rows(run/'gpu/probes.jsonl')
    expected_probes={(h,n,arm) for h,points in protocol['checkpoints'].items() for n in points for arm in ARMS[:-1]}
    require(len(probes)==len(expected_probes)==72 and
            {(r['media_sha256'],r['pairs'],r['arm']) for r in probes}==expected_probes,'probe grid')
    require(all(p['memory_unchanged'] and p['prefix_only'] and not p['accuracy_scored'] for p in probes),
            'probe isolation/scoring scope')
    generated=sum(len(r['generated_ids']) for r in pred+probes)
    require(all(0<len(r['generated_ids'])<=8 for r in pred+probes)
            and generated==complete['generated_tokens'] and generated<=6048,'generation budget')
    events=rows(run/'gpu/events.jsonl')
    streams=defaultdict(list)
    process=[]
    maximum_tokens=defaultdict(int)
    maximum_bytes=defaultdict(int)
    for e in events:
        if e['event']=='process_resources':
            process.append(e)
            continue
        require(e['event']=='append' and e['arm'] in ARMS[:-1],'unknown event')
        streams[e['media_sha256'],e['arm']].append(e)
        state=e['state']
        require(state['valid'] and not state['closed'] and not state['full_history_retained']
                and not state['language_cache_retained'],'invalid retained state')
        require(state['visual_tokens']<=384 and len(state['nodes'])<=8,'visual memory bound')
        require(state['resident_bytes']==state['tensor_bytes']+state['detector_bytes']
                and state['resident_bytes']<=8*394560+8192 and state['metadata_bytes']<=32768,
                'resident/metadata byte bound')
        maximum_tokens[e['arm']]=max(maximum_tokens[e['arm']],state['visual_tokens'])
        maximum_bytes[e['arm']]=max(maximum_bytes[e['arm']],state['resident_bytes'])
    require(set(streams)=={(h,arm) for h in groups for arm in ARMS[:-1]},'source/policy append coverage')
    for (h,arm),stream in streams.items():
        frames=groups[h]
        require(len(stream)==(len(frames)+1)//2,'prefix append count')
        for i,e in enumerate(stream):
            pair=frames[2*i:2*i+2]
            padded=len(pair)==1
            if padded: pair=pair+pair
            require(e['pair']==i and e['frame_ids']==[f['frame_id'] for f in pair]
                    and e['begin']==pair[0]['pts_s'] and e['end']==pair[1]['pts_s']
                    and e['padded_tail']==padded and e['state']['pairs_received']==i+1,
                    'sampled pair chronology or padding mismatch')
    pairs=sum((len(fs)+1)//2 for fs in groups.values())
    require(pairs==1402 and complete['source_pairs_encoded']==pairs==complete['visual_forward_calls']
            and len(process)==pairs,'visual encoding count')
    require(complete['main_answers']==684 and complete['probe_answers']==72
            and complete['reset_checks']==201 and complete['all_QA_banks_unchanged'],'completion coverage')
    require(complete['peak_GPU_allocated']<=24*2**30 and complete['peak_RSS_bytes']<=20*2**30
            and complete['seconds']<=2550,'runtime resource bound')
    require(complete['peak_combined_bank_bytes']==max(e['combined_bank_bytes'] for e in process),
            'combined memory peak receipt')
    scorer_path=REPO/'experiments/36tar/memory-eval-2b/bundle/experiments/2b-short-event-v1/scoring.py'
    spec=importlib.util.spec_from_file_location('_replay_audit_frozen_scorer',scorer_path)
    scorer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    catalog={r['id']:r for r in read(run/'scoring_ONLY.json')}
    details=read(run/'score/details.json')
    indexed={(r['id'],r['arm']):r for r in details}
    require(len(details)==684 and set(indexed)==keys,'scoring denominator')
    totals={arm:Counter() for arm in ARMS}
    for p in pred:
        k=catalog[p['id']]
        measure=scorer.measure(p['text'],len(k['options']),k['answer_index'])
        row=indexed[p['id'],p['arm']]
        require(all(row[key]==value for key,value in measure.items()),'independent scoring mismatch')
        for key in ('strict_correct','content_correct','format_compliant'):
            totals[p['arm']][key]+=measure[key]
    require(Counter(r['stratum'] for r in catalog.values())==Counter(F=58,T=60,C=53),'original stratum labels')
    require(sorted(Counter(r['cohort'] for r in catalog.values()).values())==[47,50,74],'original data cohorts')
    for dimension in ('cohort','stratum','evidence_age_proxy'):
        for value in {r[dimension] for r in details}:
            for arm in ARMS:
                selected=[r for r in details if r[dimension]==value and r['arm']==arm]
                table=summary['tables'][dimension+'/'+value][arm]
                require(table['questions']==len(selected),'group scoring denominator')
                for metric in ('strict_correct','content_correct','format_compliant'):
                    require(table[metric]==sum(r[metric] for r in selected),'group scoring count')
    review=read(run/'score/review_without_predictions.json')
    require(len(review)==171 and {r['id'] for r in review}==set(byid),'human review coverage')
    require(all(r['independent_review_status']=='pending' and r['semantic_answer_judgment'] is None
                and 'answer_index' not in r and 'text' not in r for r in review),'human review contaminated')
    require(summary['human_review_status']=='pending' and not summary['independent_holdout']
            and not summary['semantic_accuracy_reviewed'],'development/review scope')
    for name,value in summary['raw_output_hashes'].items():
        require(sha(run/'gpu'/name)==value,'score/raw binding drift')
    report=dict(job_id=int(jobid),status='independently_verified_replay',CPU_job_id=217794,
                main_answers=684,probe_answers=72,generated_tokens=generated,real_FP32_loads=1,
                original_sources=67,sampled_frames=sum(map(len,groups.values())),encoded_pairs=pairs,
                visual_forward_calls=complete['visual_forward_calls'],reset_checks=201,
                every_historical_answer_replayed=True,maximum_visual_tokens=dict(maximum_tokens),
                maximum_resident_bytes=dict(maximum_bytes),
                combined_bank_MiB=complete['peak_combined_bank_bytes']/2**20,
                peak_GPU_allocated_GiB=complete['peak_GPU_allocated']/2**30,
                peak_GPU_reserved_GiB=complete['peak_GPU_reserved']/2**30,
                peak_RSS_GiB=complete['peak_RSS_bytes']/2**30,stage_seconds=complete['stage_seconds'],
                worker_seconds=complete['seconds'],scores={a:dict(v) for a,v in totals.items()},
                cohorts=dict(Counter(r['cohort'] for r in catalog.values())),
                strata=dict(Counter(r['stratum'] for r in catalog.values())),
                human_review='pending',opened_development_not_holdout=True,
                ability_improvement_claim=False,per_arm_speed_claim=False,
                artifacts_sha256={p.relative_to(run).as_posix():sha(p) for p in run.rglob('*') if p.is_file()})
    output=ROOT/('reports/job-'+jobid+'-result-20261021.json')
    with output.open('x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ('job_id','status','main_answers','probe_answers','generated_tokens',
                      'encoded_pairs','reset_checks','scores','peak_GPU_allocated_GiB','peak_RSS_GiB')}))

if __name__=='__main__': main()
