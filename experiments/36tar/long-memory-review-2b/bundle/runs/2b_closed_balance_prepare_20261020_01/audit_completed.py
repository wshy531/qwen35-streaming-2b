"""Independent causal-input, resource and callback audit of the one A06 run."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
ARMS=('M0','M2','A06')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    folder=OUT/'gpu_217786';result=folder/'results';inputs=folder/'input'
    complete=read(result/'completed.json');protocol=read(inputs/'protocol.json')
    assert complete['status']=='complete' and complete['answers']==102 and complete['visual_encodings']==1426
    assert complete['pretrained_loads']==1 and complete['generated_tokens']<=816
    assert complete['no_full_history_feature_archive'] and not complete['wallclock']
    assert complete['protocol_sha256']==sha(inputs/'protocol.json')==sha(OUT/'runtime_inputs/protocol.json')
    for name,h in protocol['input_sha256'].items():assert sha(inputs/name)==h,name
    for name,h in read(inputs/'runtime_lock.json').items():
        p=ROOT/name if not name.startswith('repo/') else ROOT/'publish/qwen35-streaming-2b'/name[5:]
        assert sha(p)==h,name
    for name,h in read(OUT/'analysis_lock.json').items():assert sha(ROOT/name)==h,name
    auth=read(inputs/'GPU_authorization.json');assert auth['approved'] and auth['protocol_sha256']==sha(inputs/'protocol.json')
    assert read(inputs/'cpu_cluster.json')['protocol_sha256']==sha(inputs/'protocol.json')
    sources={s['source_path']:s for s in read(inputs/'sources.json')}
    snapshots={s['query_id']:s for s in read(inputs/'snapshots.json')}
    rows={r['id']:r for r in read(inputs/'requests.json')}
    old_pairs={}
    old_log=ROOT/'runs/2b_long_stream_prepare_20261020_01/gpu_217776/memory/events.jsonl'
    for line in old_log.open(encoding='utf-8'):
        e=json.loads(line)
        if e['arm']=='M2' and e['event']['kind']=='pair':
            old_pairs[e['source_path'],e['event']['pair_index']]=e['state']['bank']['details']
    assert len(old_pairs)==1426
    pair_counts=Counter();query_counts=Counter();ends={};answers={};depths=[];checkpoints=[]
    maxima=dict(main_cache_bytes=0,M0_feature_bytes=0,M2_feature_bytes=0,A06_feature_bytes=0,shared_descriptor_bytes=0,input_tokens=0)
    def audit_state(state,pairs):
        assert state['pairs']==state['vision_encodings']==pairs and not state['invalid']
        banks=state['banks'];recent=min(4,pairs)
        m2=banks['M2']['details'];a06=banks['A06'];m0=banks['M0']['details']
        assert m2['recent_event_starts']==a06['recent_event_starts']
        assert m2['within_merges']==a06['within_merges'] and m2['long_merges']==a06['long_merges']
        assert m2['tokens']==a06['tokens']<=384 and banks['M0']['visual_tokens']<=384
        assert m2['closed_tokens']<=144 and a06['closed_tokens']<=144 and a06['open_tokens']<=48
        assert m2['resident_tensor_bytes']==a06['resident_tensor_bytes']<=3156480
        assert m2['nodes'][-recent:]==a06['nodes'][-recent:]==m0['nodes'][-recent:]
        for collection in (m2,a06):
            nodes=collection['nodes'];assert nodes[0]['first']==0 and nodes[-1]['last']==pairs-1
            assert all(a['last']+1==b['first'] for a,b in zip(nodes,nodes[1:]))
            assert abs(sum(n['mass'] for n in nodes)-48*pairs)<1e-3
        maxima['main_cache_bytes']=max(maxima['main_cache_bytes'],state['main_cache_bytes'])
        for arm,value in [('M0',banks['M0']['feature_tensor_bytes']),('M2',m2['resident_tensor_bytes']),('A06',a06['resident_tensor_bytes'])]:
            maxima[arm+'_feature_bytes']=max(maxima[arm+'_feature_bytes'],value)
        maxima['shared_descriptor_bytes']=max(maxima['shared_descriptor_bytes'],banks['M2']['descriptor_bytes'])
    for line in (result/'events.jsonl').open(encoding='utf-8'):
        e=json.loads(line);source=e['source_path'];event=e['event'];kind=event['kind'];frames=sources[source]['frames']
        if kind=='pair':
            i=pair_counts[source];assert event['pair_index']==i
            assert e['state']['banks']['M2']['details']==old_pairs[source,i],'baseline bank metadata drift'
            assert event['frame_ids']==[f['frame_id'] for f in frames[2*i:2*i+2]]
            assert [event['begin_s'],event['end_s']]==[f['pts_s'] for f in frames[2*i:2*i+2]]
            pair_counts[source]+=1;audit_state(e['state'],i+1)
        elif kind=='query':
            ident=event['query_id'];assert ident in rows and query_counts[ident]==0;query_counts[ident]+=1
            expected=snapshots[ident];assert event=={k:expected[k] for k in event}
            assert e['committed_frames_sha256']==expected['committed_frames_sha256']
            assert event['committed_pairs']==pair_counts[source] and event['committed_through_s']<=rows[ident]['query_time_s']
            audit_state(e['observation_state'],pair_counts[source])
            assert e['answer']['question_did_not_change_main_or_banks']
            for arm,answer in e['answer']['arms'].items():
                assert arm in ARMS and (ident,arm) not in answers and len(answer['generated_ids'])<=8
                assert answer['visual_tokens']<=384 and answer['input_tokens']+8<=32768
                maxima['input_tokens']=max(maxima['input_tokens'],answer['input_tokens']);answers[ident,arm]=answer
            stats={}
            for arm in ('M2','A06'):
                b=e['observation_state']['banks'][arm];b=b['details'] if arm=='M2' else b
                closed=b['nodes'][:b['closed_tokens']//48]
                stats[arm]=dict(max_closed_depth=max((n['depth'] for n in closed),default=0),
                    closed_nodes=closed,within_merges=b['within_merges'],long_merges=b['long_merges'])
            depths.append({a:stats[a]['max_closed_depth'] for a in ('M2','A06')})
            checkpoints.append(dict(id=ident,source=source,**stats))
        elif kind=='end':
            assert source not in ends and event['committed_pairs']==pair_counts[source]
            assert e['producer']['sampled']==len(frames) and e['producer']['peak_RGB_buffers']<=3
            ends[source]=e
        else:raise AssertionError(kind)
    assert sum(pair_counts.values())==1426 and set(ends)==set(sources)
    assert sum(query_counts.values())==34 and set(query_counts)==set(rows)
    assert set(answers)=={(i,a) for i in rows for a in ARMS}
    callbacks={key:[] for key in answers};times={key:[] for key in answers}
    for line in (result/'tokens.jsonl').open(encoding='utf-8'):
        t=json.loads(line);key=(t['query_id'],t['arm']);assert t['token_index']==len(callbacks[key])
        callbacks[key].append(t['token_id']);times[key].append(t['monotonic_s'])
    assert sum(map(len,callbacks.values()))==complete['generated_tokens']
    assert all(callbacks[k]==a['generated_ids'] and times[k]==sorted(times[k]) for k,a in answers.items())
    assert sum(e['state']['resets'] for e in ends.values())==complete['shared_main_resets']==41
    manifest=read(folder/'manifest.json');assert not manifest['queue'].strip()
    assert '217786|COMPLETED|0:0|' in manifest['sacct']
    report=dict(status='passed',answers=102,queries=34,sources=6,pairs=1426,
        actual_sampled_frames=sum(len(s['frames']) for s in sources.values()),
        causal_prefix_all34_valid=True,all102_callbacks_match=True,main_query_and_bank_isolation_asserted_in_worker=True,
        same_actual_boundaries_and_recent_node_metadata=True,same_merge_counts_and_payload_budgets=True,
        all1426_original_M2_bank_metadata_reproduce_previous=True,
        depth_summary={a:dict(maximum=max(d[a] for d in depths),median=statistics.median(d[a] for d in depths)) for a in ('M2','A06')},
        query_checkpoint_details=checkpoints,maxima=maxima,completed=complete,
        locks_and_authorization_valid=True,queue_empty=True,sacct=manifest['sacct'],
        limitations=['checkpoint depths are34 correlated observations, not34 independent sources',
            'mass per output token not captured this run; do not claim concentration improved',
            'shared process resources cannot be split into independent speed comparisons',
            'node tensor budgets exclude model weights, main state and temporary QA allocations'])
    with (OUT/'independent_final_audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(dict(status='passed',depth_summary=report['depth_summary'],maxima=maxima))


if __name__=='__main__':main()
