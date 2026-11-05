"""Audit committed cutoffs, callback delivery and bounded state separately from accuracy."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from inventory_local import OUT


def audit(folder):
    snapshots={s['query_id']:s for s in json.loads((OUT/'stream_query_snapshots34.json').read_text(encoding='utf-8'))}
    counts=defaultdict(lambda:dict(pairs=0,queries=0,ends=0,resets_max=0,main_cache_bytes_max=0,
        bank_feature_tensor_bytes_max=0,bank_visual_tokens_max=0))
    query={};tokens=defaultdict(list);completions={}
    for stage in ('native','memory'):
        p=folder/stage
        if (p/'completed.json').exists():completions[stage]=json.loads((p/'completed.json').read_text())
        if (p/'tokens.jsonl').exists():
            for line in (p/'tokens.jsonl').read_text().splitlines():
                item=json.loads(line);tokens[item['arm'],item['query_id']].append(item)
        if not (p/'events.jsonl').exists():continue
        for line in (p/'events.jsonl').read_text().splitlines():
            item=json.loads(line);arm=item['arm'];e=item['event'];c=counts[arm]
            if e['kind']=='pair':
                c['pairs']+=1;s=item['state']
                assert s['pairs']==e['pair_index']+1 and s['last_pts']==e['end_s']
                if arm!='native':
                    assert s['main_seen_tokens']<=1912 and s['vision_encodings']==s['pairs']
                    c['resets_max']=max(c['resets_max'],s['resets'])
                    c['main_cache_bytes_max']=max(c['main_cache_bytes_max'],s['main_cache_bytes'])
                    if s['bank'] is not None:
                        b=s['bank'];assert b['visual_tokens']<=384 and b['feature_tensor_bytes']<=3156480
                        assert b['processed_pairs']==s['pairs']
                        c['bank_feature_tensor_bytes_max']=max(c['bank_feature_tensor_bytes_max'],b['feature_tensor_bytes'])
                        c['bank_visual_tokens_max']=max(c['bank_visual_tokens_max'],b['visual_tokens'])
            elif e['kind']=='query':
                c['queries']+=1;k=(arm,e['query_id']);assert k not in query;query[k]=item
                snap=snapshots[e['query_id']]
                assert item['committed_frames_sha256']==snap['committed_frames_sha256']
                assert item['answer']['observation_through_s']==snap['committed_through_s']<=e['query_time_s']
                if arm in ('M0','M2','main_only','text_only'):
                    assert item['answer']['main_counters_unchanged'] and item['answer']['bank_unchanged']
            elif e['kind']=='end':
                c['ends']+=1;assert item['producer']['peak_RGB_buffers']<=3
                assert item['state']['pairs']==e['committed_pairs']
    for k,item in query.items():
        emitted=tokens[k];ids=item['answer']['generated_ids']
        assert [e['token_id'] for e in emitted]==ids
        assert [e['token_index'] for e in emitted]==list(range(len(ids)))
        assert all(a['monotonic_s']<=b['monotonic_s'] for a,b in zip(emitted,emitted[1:]))
        a=item['answer'];assert a['first_token_seconds']<=a['last_token_seconds']<=a['total_seconds']
    complete=set(completions)=={'native','memory'}
    if complete:
        assert len(query)==170 and sum(len(v) for v in tokens.values())<=1360
        assert all(counts[a]['queries']==34 for a in ('native','main_only','M0','M2','text_only'))
        assert all(counts[a]['pairs']==1426 and counts[a]['ends']==6 for a in ('native','main_only','M0','M2'))
        assert sum(v['pretrained_loads'] for v in completions.values())==2
    return dict(status='complete_contracts_passed' if complete else 'partial_contracts_only',
        queries=len(query),callback_tokens=sum(len(v) for v in tokens.values()),arms=dict(counts),
        stages=completions,no_future_or_query_padding=True,callbacks_match_saved_answer_tokens=True,
        bank_and_main_counters_isolated=True,
        limitations=['real-run main tensor bitwise isolation not measured at every query; random tiny CPU proof separate',
          'RSS is event-time observed maximum, not continuous process peak',
          'cache/bank byte bounds exclude model and transient query/input tensors',
          'deterministic causal replay; no wall-clock input or drop-rate claim'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);a=p.parse_args();result=audit(a.results)
    with a.out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
