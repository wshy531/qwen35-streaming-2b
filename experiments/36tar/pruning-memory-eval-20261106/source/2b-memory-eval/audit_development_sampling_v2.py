"""One bounded real 1FPS development index; no model outputs or heldout frames.

Compare with the already registered .5FPS trace and human-review witnesses.
Coverage is input evidence only, not an answer score or exclusive necessity.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[2]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def interval_hits(frames,intervals):
    if not intervals:return None
    return [any(a-1e-9<=f['pts_s']<=b+1e-9 for f in frames) for a,b in intervals]


def witness_matches(frames,witnesses):
    lookup={(f['frame_id'],round(f['pts_s'],7),f['rgb_sha256']) for f in frames}
    return [dict(frame_id=w['frame_id'],pts_s=w['pts_s'],
        exact_input_match=(w['frame_id'],round(w['pts_s'],7),w['rgb_sha256']) in lookup)
        for w in witnesses if all(k in w for k in ['frame_id','pts_s','rgb_sha256'])]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve()
    if out.parent!=(ROOT/'runs').resolve():raise ValueError('own output required')
    out.mkdir(exist_ok=False)
    release=ROOT/'runs/dataset_build_20261018_01/content_release_v1'
    manifest=ROOT/'experiments/2b-memory-eval/data/candidate_manifest.json'
    content=read(release/'summary.json')
    if sha(manifest)!=content['candidate_manifest_sha256']:raise ValueError('raw manifest drift')
    overlay=[q for q in read(release/'question_annotation_overlay.json')['questions'] if q['original']['split']=='development']
    media=[m for m in read(release/'media_inventory.json')['files'] if m['split']=='development']
    original=[q['original'] for q in overlay]
    if len(original)!=46 or len(media)!=20:raise ValueError('fixed20 development sources and46 rows required')
    original_sha=hashlib.sha256(json.dumps(original,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    old={t['source_path']:t for t in read(release/'actual_sampling_traces.json')['videos'] if t['split']=='development'}
    started=time.monotonic();new={}
    from audit_sampling import sample
    for i,m in enumerate(media):
        path=ROOT/m['media_path'];h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        if h.hexdigest()!=m['sha256']:raise ValueError('development media drift')
        query=max(q['original']['query_time_s'] for q in overlay if q['original']['source_path']==m['source_path'])
        trace=sample(path,query,1.,seconds=120)
        trace.update(source_path=m['source_path'],media_sha256=m['sha256'],split='development')
        new[m['source_path']]=trace
        print(json.dumps(dict(source=i+1,total=20,frames=len(trace['frames']),seconds=time.monotonic()-started)),flush=True)
        if time.monotonic()-started>570:raise TimeoutError('bounded development index')
    save(out/'actual_1fps_development_traces.json',dict(status='actual_CPU_decode_not_model_run',videos=list(new.values())))
    result=[]
    for q in overlay:
        r=q['original'];source=r['source_path'];query=r['query_time_s']
        frames05=[f for f in old[source]['frames'] if f['pts_s']<=query+1e-9]
        frames1=[f for f in new[source]['frames'] if f['pts_s']<=query+1e-9]
        intervals=r.get('evidence_intervals');witnesses=q['review'].get('witnesses',[])
        h05,h1=interval_hits(frames05,intervals),interval_hits(frames1,intervals)
        w05,w1=witness_matches(frames05,witnesses),witness_matches(frames1,witnesses)
        result.append(dict(id=q['id'],source_path=source,query_time_s=query,author_intervals=intervals,
            frames_05fps=len(frames05),frames_1fps=len(frames1),
            author_hits_05fps=h05,author_hits_1fps=h1,
            review_witnesses_05fps=w05,review_witnesses_1fps=w1,
            review_verdict=q['review']['verdict'],
            added_author_interval_hit=bool(h1 is not None and all(h1) and not all(h05))))
    # Actual frame identity, not timestamp-only matching, determines subset.
    for r in result:
        s=r['source_path'];query=r['query_time_s']
        key=lambda f:(f['frame_id'],f['rgb_sha256'])
        f05={key(f) for f in old[s]['frames'] if f['pts_s']<=query+1e-9}
        f1={key(f) for f in new[s]['frames'] if f['pts_s']<=query+1e-9}
        r['05fps_is_subset_of_1fps']=f05<=f1
    save(out/'question_input_coverage.json',result)
    gqa=[r for r in result if r['author_hits_05fps'] is not None]
    summary=dict(status='CPU_development_sampling_audit_complete_not_final_protocol',questions=len(result),sources=len(media),
        actual_1fps_frames=sum(len(t['frames']) for t in new.values()),
        gqa_questions_with_author_intervals=len(gqa),
        all_author_intervals_hit_05fps=sum(all(r['author_hits_05fps']) for r in gqa),
        all_author_intervals_hit_1fps=sum(all(r['author_hits_1fps']) for r in gqa),
        added_author_interval_hit_ids=[r['id'] for r in result if r['added_author_interval_hit']],
        any_exact_review_witness_05fps=sum(any(w['exact_input_match'] for w in r['review_witnesses_05fps']) for r in result),
        any_exact_review_witness_1fps=sum(any(w['exact_input_match'] for w in r['review_witnesses_1fps']) for r in result),
        witnessed_question_denominator=sum(bool(r['review_witnesses_1fps']) for r in result),
        non_subset_ids=[r['id'] for r in result if not r['05fps_is_subset_of_1fps']],
        original_development_rows_sha256=original_sha,raw_manifest_sha256=sha(manifest),
        elapsed_seconds=time.monotonic()-started,model_loads=0,answers=0,holdout_frames_reviewed=0,
        limitations=['Author intervals not verified jointly necessary or exclusive.',
            'Exact sampled witness match does not prove192x256 recognizability.',
            'No1FPS pretrained answers: do not infer accuracy change or freeze1FPS from coverage alone.',
            'Unreviewed/uncertain questions remain in records; no output-based selection.'])
    save(out/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
