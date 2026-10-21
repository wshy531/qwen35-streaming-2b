"""Read existing OVO media headers and pinned questions; no frames/model/network."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import av

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/2b_long_stream_prepare_20261020_01'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    official=ROOT/'experiments/2b-memory-eval/data/upstream/ovo/ovo_bench_new.json'
    questions=json.loads(official.read_text(encoding='utf-8-sig'))
    media={}
    for manifest in sorted((ROOT/'datasets/2b-memory-eval').glob('ovo*/download_manifest.json')):
        for f in json.loads(manifest.read_text(encoding='utf-8-sig'))['files']:
            name=f.get('source_path',f.get('source'))
            if not name:raise ValueError('unknown manifest source schema')
            local=Path(f['local_path']) if 'local_path' in f else manifest.parent/f['path']
            if not local.is_absolute():local=ROOT/local
            if not local.is_file():raise ValueError('missing owned media: '+str(local))
            assert local.resolve().is_relative_to((ROOT/'datasets/2b-memory-eval').resolve())
            if name in media:
                assert media[name]['declared_sha256']==f['sha256']
                continue
            media[name]=dict(source_path=name,local_path=str(local),bytes=local.stat().st_size,
                declared_sha256=f['sha256'],manifest=str(manifest.relative_to(ROOT)))
    started=time.monotonic()
    rows=[]
    for name,m in sorted(media.items()):
        assert time.monotonic()-started<90
        with av.open(m['local_path']) as c:
            stream=c.streams.video[0]
            duration=float(stream.duration*stream.time_base) if stream.duration is not None else float(c.duration/av.time_base)
            meta=dict(duration_s=duration,fps=float(stream.average_rate) if stream.average_rate else None,
                width=stream.codec_context.width,height=stream.codec_context.height)
        matched=[q for q in questions if q['video']==name]
        qs=[]
        for q in matched:
            t=q.get('realtime')
            scalar=isinstance(t,(int,float)) and not isinstance(t,bool)
            qs.append(dict(id='ovo:'+str(q['id']),task=q['task'],query_time_s=t,
                scalar_timestamp=scalar,in_media_time_range=bool(scalar and 0<=t<=duration+.1),
                original=q,content_admission='not_established_by_this_header_inventory'))
        timestamps=sorted({q['query_time_s'] for q in qs if q['in_media_time_range']})
        historical=[q for q in qs if q['in_media_time_range'] and q['task']=='EPM']
        rows.append(dict(**m,**meta,questions=qs,question_count=len(qs),
            distinct_query_times=timestamps,
            in_range_queries_after180s=sum(q['in_media_time_range'] and q['query_time_s']>=180 for q in qs),
            EPM_questions=len(historical),EPM_distinct_times=sorted({q['query_time_s'] for q in historical})))
    summary=dict(status='actual_local_header_inventory_not_content_review_or_model_test',
        media=len(rows),annotation_rows=sum(r['question_count'] for r in rows),
        task_counts=dict(Counter(q['task'] for r in rows for q in r['questions'])),
        videos_at_least_180s=sum(r['duration_s']>=180 for r in rows),
        videos_at_least_300s=sum(r['duration_s']>=300 for r in rows),
        videos_at_least_600s=sum(r['duration_s']>=600 for r in rows),
        maximum_duration_s=max(r['duration_s'] for r in rows),
        videos_with_multiple_in_range_query_times=sum(len(r['distinct_query_times'])>=2 for r in rows),
        EPM_questions=sum(r['EPM_questions'] for r in rows),
        EPM_sources=sum(bool(r['EPM_questions']) for r in rows),
        official_annotations_sha256=digest(official),header_probe_seconds=time.monotonic()-started,
        no_new_download=True,no_cluster_connection=True,model_loads=0,
        limits=['Only headers read, no pixels reviewed',
          'declared asset hashes inherited from existing manifests, not rehashed by this inventory',
          'multiple timestamps do not establish difficulty or necessary old evidence',
          'other OVO tasks need original task semantics; ASI is not automatically historical QA'])
    with (OUT/'local_long_video_inventory.json').open('x',encoding='utf-8') as f:
        json.dump(dict(summary=summary,media=rows),f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False))
    for r in rows:
        if r['duration_s']>=300:
            print(json.dumps(dict(source=r['source_path'],seconds=r['duration_s'],questions=r['question_count'],
                EPM=r['EPM_questions'],timestamps=r['distinct_query_times']),ensure_ascii=False))


if __name__=='__main__':main()
