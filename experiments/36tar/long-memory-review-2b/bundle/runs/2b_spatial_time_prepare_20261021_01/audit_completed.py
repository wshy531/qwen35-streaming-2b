"""Independent full-run audit; no model calls and no altered scoring."""
from collections import Counter
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
ARMS=('M0_384','M2_384','M0_768','M2_768','S24_384')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    folder=OUT/'gpu_217797';result=folder/'results';inputs=folder/'input'
    complete=read(result/'completed.json');protocol=read(inputs/'protocol.json')
    assert complete['status']=='complete' and complete['answers']==170 and complete['visual_encodings']==1426
    assert complete['pretrained_loads']==1 and complete['generated_tokens']<=1360
    assert complete['no_full_history_feature_archive'] and not complete['wallclock']
    assert complete['protocol_sha256']==sha(inputs/'protocol.json')==sha(OUT/'runtime_inputs_v2/protocol.json')
    for name,h in protocol['input_sha256'].items():assert sha(inputs/name)==h,name
    for name,h in read(inputs/'runtime_lock.json').items():
        p=ROOT/name if not name.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(p)==h,name
    for name,h in read(OUT/'analysis_lock.json').items():assert sha(ROOT/name)==h,name
    auth=read(inputs/'GPU_authorization.json');assert auth['approved'] and auth['protocol_sha256']==sha(inputs/'protocol.json')
    assert read(inputs/'cpu_cluster.json')['protocol_sha256']==sha(inputs/'protocol.json')
    sources={s['source_path']:s for s in read(inputs/'sources.json')}
    snaps={s['query_id']:s for s in read(inputs/'snapshots.json')}
    rows={r['id']:r for r in read(inputs/'requests.json')}
    old_pairs={}
    for line in (ROOT/'runs/2b_long_stream_prepare_20261020_01/gpu_217776/memory/events.jsonl').open(encoding='utf-8'):
        e=json.loads(line)
        if e['arm']=='M2' and e['event']['kind']=='pair':old_pairs[e['source_path'],e['event']['pair_index']]=e['state']['bank']['details']
    assert len(old_pairs)==1426
    pair_counts=Counter();query_counts=Counter();ends={};answers={};checkpoints=[]
    maxima=dict(main_cache_bytes=0,shared_descriptor_bytes=0,input_tokens=0,**{a+'_tensor_bytes':0 for a in ARMS})
    def bank_details(state):
        b=state['banks'];return {a:b[a]['details'] if a in ('M0_384','M2_384') else b[a] for a in ARMS}
    def audit_state(state,pairs):
        assert state['pairs']==state['vision_encodings']==pairs and not state['invalid']
        b=bank_details(state);recent=min(4,pairs)
        assert b['M2_384']['recent_event_starts']==b['M2_768']['recent_event_starts']==b['S24_384']['recent_event_starts']
        assert b['M2_384']['within_merges']==b['M2_768']['within_merges']==b['S24_384']['within_merges']
        reference=b['M2_384']['nodes'][-recent:]
        for a in ARMS:
            nodes=b[a]['nodes'];budget=768 if a.endswith('768') else 384
            assert nodes[-recent:]==reference
            assert sum(n['tokens'] for n in nodes)<=budget
            payload=sum(n['tensor_bytes'] for n in nodes)
            assert payload<=budget*(2048*4+4+3*8)
            maxima[a+'_tensor_bytes']=max(maxima[a+'_tensor_bytes'],payload)
            assert nodes[-1]['last']==pairs-1
            assert all(n['tokens']==48 and n['exact_pair'] for n in nodes[-recent:])
            if a.startswith('M0'):
                cap=budget//48;assert [n['first'] for n in nodes]==list(range(max(0,pairs-cap),pairs))
            else:
                assert nodes[0]['first']==0 and all(x['last']+1==y['first'] for x,y in zip(nodes,nodes[1:]))
                assert abs(sum(n['mass'] for n in nodes)-48*pairs)<1e-3
                old_tokens=24 if a=='S24_384' else 48
                assert all(n['tokens']==old_tokens for n in nodes[:-recent])
                assert b[a]['closed_tokens']<=budget-192-old_tokens and b[a]['open_tokens']<=old_tokens
                if a=='S24_384':assert b[a]['spatial_reductions']==max(0,pairs-4)
            profiles=state['mass_profiles'][a];assert len(profiles)==len(nodes)
            for n,m in zip(nodes,profiles):
                assert (m['first'],m['last'])==(n['first'],n['last'])
                assert 0<m['max_mass_fraction']<=1 and 1<=m['max_token_mass']<=n['mass']
                assert abs(m['max_token_mass']/n['mass']-m['max_mass_fraction'])<1e-6
        maxima['main_cache_bytes']=max(maxima['main_cache_bytes'],state['main_cache_bytes'])
        maxima['shared_descriptor_bytes']=max(maxima['shared_descriptor_bytes'],state['banks']['M2_384']['descriptor_bytes'])
    for line in (result/'events.jsonl').open(encoding='utf-8'):
        e=json.loads(line);source=e['source_path'];ev=e['event'];kind=ev['kind'];frames=sources[source]['frames']
        if kind=='pair':
            i=pair_counts[source];assert ev['pair_index']==i
            assert e['state']['banks']['M2_384']['details']==old_pairs[source,i]
            assert ev['frame_ids']==[f['frame_id'] for f in frames[2*i:2*i+2]]
            assert [ev['begin_s'],ev['end_s']]==[f['pts_s'] for f in frames[2*i:2*i+2]]
            pair_counts[source]+=1;audit_state(e['state'],i+1)
        elif kind=='query':
            i=ev['query_id'];assert i in rows and query_counts[i]==0;query_counts[i]+=1
            expected=snaps[i];assert ev=={k:expected[k] for k in ev}
            assert e['committed_frames_sha256']==expected['committed_frames_sha256']
            assert ev['committed_pairs']==pair_counts[source] and ev['committed_through_s']<=rows[i]['query_time_s']
            state=e['observation_state'];audit_state(state,pair_counts[source]);b=bank_details(state)
            assert e['answer']['question_did_not_change_main_or_banks'] and set(e['answer']['arms'])==set(ARMS)
            for a,answer in e['answer']['arms'].items():
                assert (i,a) not in answers and len(answer['generated_ids'])<=8
                assert answer['visual_tokens']<=(768 if a.endswith('768') else 384) and answer['input_tokens']+8<=32768
                maxima['input_tokens']=max(maxima['input_tokens'],answer['input_tokens']);answers[i,a]=answer
            checkpoints.append(dict(id=i,source=source,banks=b,mass_profiles=state['mass_profiles']))
        elif kind=='end':
            assert source not in ends and ev['committed_pairs']==pair_counts[source]
            assert e['producer']['sampled']==len(frames) and e['producer']['peak_RGB_buffers']<=3
            ends[source]=e
        else:raise AssertionError(kind)
    assert sum(pair_counts.values())==1426 and set(ends)==set(sources)
    assert sum(query_counts.values())==34 and set(query_counts)==set(rows)
    assert set(answers)=={(i,a) for i in rows for a in ARMS}
    callbacks={k:[] for k in answers};times={k:[] for k in answers}
    for line in (result/'tokens.jsonl').open(encoding='utf-8'):
        t=json.loads(line);k=(t['query_id'],t['arm']);assert t['token_index']==len(callbacks[k])
        callbacks[k].append(t['token_id']);times[k].append(t['monotonic_s'])
    assert sum(map(len,callbacks.values()))==complete['generated_tokens']
    assert all(callbacks[k]==a['generated_ids'] and times[k]==sorted(times[k]) for k,a in answers.items())
    assert sum(e['state']['resets'] for e in ends.values())==complete['shared_main_resets']==41
    manifest=read(folder/'manifest.json');assert not manifest['queue'].strip() and '217797|COMPLETED|0:0|' in manifest['sacct']
    report=dict(status='passed',answers=170,queries=34,sources=6,pairs=1426,
        actual_sampled_frames=sum(len(s['frames']) for s in sources.values()),
        causal_prefix_all34_valid=True,all170_callbacks_match=True,
        main_query_and_bank_isolation_asserted_in_worker=True,same_boundaries_recent_and_within_merge_counts=True,
        all1426_original_M2_bank_metadata_reproduce_previous=True,all_bank_caps_and_mass_valid=True,
        query_checkpoint_details=checkpoints,maxima=maxima,completed=complete,
        locks_and_authorization_valid=True,queue_empty=True,sacct=manifest['sacct'],
        limitations=['metadata/source coverage not readability','shared process not per-arm speed',
            'resident bank tensors exclude weights, main state and temporary branches','34 checkpoints are correlated'])
    with (OUT/'independent_final_audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(dict(status='passed',maxima=maxima))


if __name__=='__main__':main()
