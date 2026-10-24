"""Independent standard-library audit of one completed LS_depth run."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
ARMS=('M0','reservoir4','M2','A06','LS','LS_depth')


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def lines(path):return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(result,out):
    run=HERE/'inputs';lock=read(run/'execution_lock.json')
    for name,digest in lock['input_sha256'].items():assert sha(run/name)==digest,name
    rows={q['id']:q for q in read(run/'questions.json')};groups={g['id']:g for g in read(run/'groups.json')}
    predictions=lines(result/'answers.jsonl');tokens=lines(result/'tokens.jsonl');events=lines(result/'events.jsonl')
    complete=read(result/'completed.json')
    assert len(predictions)==len({(r['id'],r['arm']) for r in predictions})==1818
    assert {(r['id'],r['arm']) for r in predictions}=={(uid,arm) for uid in rows for arm in ARMS}
    token_groups=defaultdict(list)
    for t in tokens:token_groups[t['query_id'],t['arm']].append(t)
    for p in predictions:
        q=rows[p['id']];ts=token_groups[p['id'],p['arm']]
        assert p['status']=='ok' and p['group']==q['group'] and p['cohort']==q['cohort']
        assert p['generated_ids']==[t['token_id'] for t in ts]
        assert [t['token_index'] for t in ts]==list(range(len(ts))) and 1<=len(ts)<=8
        assert all(a['monotonic_s']<=b['monotonic_s'] for a,b in zip(ts,ts[1:]))
        assert p['visual_tokens']==sum(n['tokens'] for n in p['nodes'])<=384
        assert 0<p['input_tokens']<=2040 and p['banks_unchanged'] and p['branch_released'] and p['no_main_language_cache']
        assert all(n['end']<=q['query_time_s']+1e-9 for n in p['nodes'])
        assert all(a['last']<b['first'] and a['end']<b['begin'] for a,b in zip(p['nodes'],p['nodes'][1:]))
    end=[e for e in events if e.get('kind')=='group_end']
    assert len(end)==112 and {e['group'] for e in end}==set(groups)
    assert sum(e['state']['vision_encodings'] for e in end)==3981
    peaks={a:dict(tensor_bytes=0,depth=0) for a in ARMS}
    states=[e['state'] for e in end]
    states += [e['state'] for e in events if e.get('event',{}).get('kind')=='pair']
    for state in states:
        assert state['no_main_language_cache'] and state['banks']['no_full_history_feature_list']
        assert state['banks']['shared_descriptor_bytes']==8192
        for arm,b in state['banks']['banks'].items():
            assert b['visual_tokens']<=384 and b['resident_tensor_bytes']<=3156480
            peaks[arm]['tensor_bytes']=max(peaks[arm]['tensor_bytes'],b['resident_tensor_bytes'])
            details=b['details']
            if arm in ('LS','LS_depth'):assert details['history_tokens']<=192 and details['recent_tokens']<=192
            ns=details.get('nodes',details.get('details',{}).get('nodes',[]))
            peaks[arm]['depth']=max(peaks[arm]['depth'],max((n['depth'] for n in ns),default=0))
    queries=[e for e in events if e.get('event',{}).get('kind')=='query']
    pairs=[e for e in events if e.get('event',{}).get('kind')=='pair']
    eof=[e for e in events if e.get('event',{}).get('kind')=='end']
    assert len(queries)==34 and len(pairs)==1426 and len(eof)==6
    snapshots={s['query_id']:s for s in read(run/'snapshots34.json')}
    continued=[]
    for e in queries:
        event=e['event'];expected=snapshots[event['query_id']]
        assert event=={k:expected[k] for k in event}
        assert e['committed_frames_sha256']==expected['committed_frames_sha256']
        assert e['observation_state']['pairs']==event['committed_pairs']
        assert set(e['answer']['arms'])==set(ARMS) and e['answer']['all_QA_branches_released']
        future=[p for p in pairs if p['source_path']==e['source_path'] and p['event']['pair_index']>=event['committed_pairs']]
        assert future,'no post-question continuation'
        continued.append(event['query_id'])
    assert complete['status']=='complete' and complete['answers']==1818 and complete['pretrained_loads']==1
    assert complete['vision_encodings']==3981 and complete['generated_tokens']==len(tokens)<=14544
    assert complete['seconds']<3300 and complete['execution_lock_sha256']==sha(run/'execution_lock.json')
    assert complete['no_full_history_archive'] and complete['no_persistent_main_language_state']
    receipt=dict(status='passed',answers=1818,token_callbacks=len(tokens),groups=112,visual_encodings=3981,
        long_queries=34,long_pairs=1426,long_EOF=6,queries_followed_by_new_real_pairs=len(continued),
        bounded_payload_and_causal_nodes=True,observed_state_peaks=peaks,complete=complete,
        source_sha256={p.name:sha(p) for p in result.iterdir() if p.is_file()},
        limitations=['mechanism audit is separate from answer correctness',
            'source-range coverage is not preserved visual detail','one fixed opened-development comparison'])
    with out.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,ensure_ascii=False,indent=2)
    print(dict(status='passed',answers=1818,callbacks=len(tokens),long_queries=34))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.results,a.out)
