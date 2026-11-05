"""Freeze existing long-stream candidates and a metadata-only schedule rehearsal."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from inventory_local import ROOT,OUT
from schedule import Frame,Query,events


def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def main():
    inventory=json.loads((OUT/'local_long_video_inventory.json').read_text(encoding='utf-8'))
    selected=[m for m in inventory['media'] if m['source_path'].startswith('Ego4D/') and m['EPM_questions']]
    assert len(selected)==6
    rows=[];proofs=[]
    for m in selected:
        qs=sorted([q for q in m['questions'] if q['task']=='EPM' and q['in_media_time_range']],
                  key=lambda q:(q['query_time_s'],q['id']))
        rows.extend(dict(**q,source_path=m['source_path'],local_path=m['local_path'],
            media_sha256=m['declared_sha256'],source_duration_s=m['duration_s'],
            annotation_source='pinned official OVO EPM',
            status='candidate_only_needs_actual_prefix_pixel_review',
            original_query_time_unchanged=True) for q in qs)
        rehearsal=list(events((Frame(i,float(i)) for i in range(math.floor(m['duration_s'])+1)),
            [Query(q['id'],q['query_time_s']) for q in qs],end_s=m['duration_s']))
        answered=[e for e in rehearsal if e['kind']=='query']
        assert len(answered)==len(qs)
        assert all(e['committed_through_s'] is None or e['committed_through_s']<=e['query_time_s'] for e in answered)
        proofs.append(dict(source=m['source_path'],mode='synthetic_integer_grid_NOT_actual_video_PTS',
            query_events=answered,pairs=sum(e['kind']=='pair' for e in rehearsal)))
    assert len(rows)==28 and len({r['id'] for r in rows})==28
    save('existing6_EPM28_candidates_LOCAL_ONLY.json',rows)
    save('metadata_schedule_rehearsal.json',dict(status='metadata_only_passed',videos=6,questions=28,
        actual_RGB_decoded=False,model_loads=0,GPU_jobs=0,streams=proofs))
    code=Path(__file__).parent
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(code),'-p','test_schedule.py','-v'],
        capture_output=True,text=True,timeout=30)
    save('CPU_schedule_contracts.json',dict(exit_code=test.returncode,stdout=test.stdout,stderr=test.stderr,
        model_loads=0,GPU_jobs=0,scope='scheduler only, no model/media backend integration'))
    assert test.returncode==0
    save('source_lock.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob('*.py')})
    print(dict(videos=6,candidate_questions=28,queries_after300s=sum(r['query_time_s']>=300 for r in rows),
        actual_pixel_review=False,CPU_schedule_contracts=7,metadata_rehearsal=True,GPU_jobs=0))


if __name__=='__main__':main()
