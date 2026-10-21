"""Freeze answer-blind timed requests; preserve all original annotations locally."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from inventory_local import ROOT,OUT
from schedule import Frame,Query,events


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    originals=read('existing6_EPM28_candidates_LOCAL_ONLY.json')
    manual=read('manual6_delayed_questions_LOCAL_ONLY.json')
    reviews={q['id']:q for q in read('official28_content_review.json')}
    requests=[];labels=[]
    for q in originals+manual:
        official=q['id'].startswith('ovo:')
        raw=q['original'] if official else q
        requests.append(dict(id=q['id'],source_path=q['source_path'],query_time_s=q['query_time_s'],
            question=raw['question'],options=raw['options'],
            cohort='official_OVO_unchanged' if official else 'manual_development'))
        labels.append(dict(id=q['id'],answer_index=raw['gt'] if official else raw['answer_index'],
            verdict=reviews[q['id']]['verdict'] if official else q['verdict'],
            independent_second_review=False,old_evidence_necessity_proven=False))
    assert len(requests)==len({q['id'] for q in requests})==34
    assert Counter(q['cohort'] for q in requests)=={'official_OVO_unchanged':28,'manual_development':6}
    sources=[];snapshots=[]
    for t in read('actual_sampling_all6.json')['videos']:
        rows=sorted([q for q in requests if q['source_path']==t['source_path']],key=lambda q:(q['query_time_s'],q['id']))
        for e in events((Frame(f['frame_id'],f['pts_s']) for f in t['frames']),
                [Query(q['id'],q['query_time_s']) for q in rows],end_s=t['duration_s']):
            if e['kind']!='query':continue
            prefix=t['frames'][:e['committed_pairs']*2]
            snapshots.append(dict(source_path=t['source_path'],**e,
                committed_frames_sha256=hashlib.sha256(json.dumps(prefix,sort_keys=True).encode()).hexdigest()))
        sources.append({k:t[k] for k in ('source_path','media_path','media_sha256','duration_s','frames')})
    save('stream_requests34.json',requests)
    save('stream_labels34_LOCAL_ONLY.json',labels)
    save('stream_sources6.json',sources)
    save('stream_query_snapshots34.json',snapshots)
    print(dict(questions=len(requests),sources=len(sources),snapshots=len(snapshots),pretrained_loads=0))


if __name__=='__main__':main()
