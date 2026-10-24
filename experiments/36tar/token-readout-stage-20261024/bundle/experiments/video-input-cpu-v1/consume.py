"""Exercise old query semantics without importing a model or torch."""
import hashlib,json
from schedule import Query,events


def consume(source,group,questions,snapshots):
    rows=sorted(questions,key=lambda q:(q['query_time_s'],q['id']))
    if group['cohort']!='stream34':
        pending=None;pairs=received=0
        for frame in source.frames():
            packet=source.take_pair([frame.frame_id])[0];received+=1
            if pending is None:pending=packet
            else:pairs+=1;pending=None
        tail=pending is not None
        if tail:pairs+=1
        pending=None
        assert source.completed
        result=[]
        for q in rows:
            assert q['pairs']==pairs and q['query_time_s']>=group['frames'][-1]['pts_s']
            result.append(dict(id=q['id'],query_time_s=q['query_time_s'],pairs=pairs,received=received,
                               query_only_tail_padding=tail,no_future_frame_delivered=True))
        source.close();return result
    expected={s['query_id']:s for s in snapshots}
    prefix=hashlib.sha256(b'[');committed=0;checks=[]
    for event in events(source.frames(),[Query(q['id'],q['query_time_s']) for q in rows],end_s=group['duration_s']):
        if event['kind']=='pair':
            packets=source.take_pair(event['frame_ids'])
            for packet in packets:
                meta=group['frames'][committed]
                assert packet.frame_id==meta['frame_id'] and abs(packet.timestamp-meta['pts_s'])<1e-7
                if committed:prefix.update(b', ')
                prefix.update(json.dumps(meta,sort_keys=True).encode());committed+=1
            del packets
        elif event['kind']=='query':
            wanted=expected[event['query_id']]
            assert event=={k:wanted[k] for k in event}
            h=prefix.copy();h.update(b']');assert h.hexdigest()==wanted['committed_frames_sha256']
            checks.append(dict(event=event,committed_frames_sha256=h.hexdigest()))
        elif event['kind']=='end':
            assert source.completed and committed==event['committed_pairs']*2
            assert set(source.buffer)==({event['pending_frame_id']} if event['pending_frame_id'] is not None else set())
            source.close()
    assert len(checks)==len(rows) and not source.buffer
    return checks
