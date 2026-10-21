"""All500 question metadata screened, no visual admission or output selection."""
import collections,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/2b_memory_value_design_20261021_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def stamp(t):
    if not isinstance(t,dict) or not re.fullmatch(r'DAY[1-7]',str(t.get('date',''))):return None
    s=t.get('time')
    if not isinstance(s,str) or not re.fullmatch(r'\d{8}',s):return None
    h,m,sec=int(s[:2]),int(s[2:4]),int(s[4:6])
    if h>23 or m>59 or sec>59:return None
    # Last two digits not interpreted as centiseconds without official clock verification.
    return int(t['date'][3:])*86400+h*3600+m*60+sec


def main():
    qa=read(OUT/'official_QA.json');files=read(OUT/'day1_listing.json');rows=[]
    assert len(qa)==500 and len({q['ID'] for q in qa})==500
    for q in qa:
        qt=stamp(q['query_time']);target=q['target_time'];times=[];reason=[]
        if 'time_list' in target:
            reason.append('multiple_witnesses_require_explicit_audit')
            times=[stamp(dict(date=target.get('date'),time=t)) for t in target['time_list']]
        else:times=[stamp(target)]
        if qt is None or any(t is None for t in times):reason.append('noncanonical_time_needs_source_audit')
        if qt is not None and any(t is not None and t>qt for t in times):reason.append('future_target_needs_audit')
        if q['need_audio']:reason.append('audio_required')
        if q['need_name']:reason.append('identity_grounding_required')
        delay=None if qt is None or not times or any(t is None for t in times) else qt-min(times)
        rows.append(dict(id=q['ID'],query_day=q['query_time']['date'],type=q['type'],
            need_audio=q['need_audio'],need_name=q['need_name'],reasons=reason,
            nominal_oldest_delay_seconds=delay,metadata_visual_no_name=not q['need_audio'] and not q['need_name'],
            visually_verified=False,admitted=False))
    pure=[r for r in rows if r['metadata_visual_no_name']];day1=[r for r in pure if r['query_day']=='DAY1']
    videos=[f for f in files if f['type']=='file' and f['path'].endswith('.mp4')]
    plan=dict(status='metadata_screened_NOT_video_or_question_admission',question_rows=500,
        no_audio_no_name_metadata_rows=len(pure),day1_no_audio_no_name_rows=len(day1),
        canonical_no_audio_no_name_rows=sum(not r['reasons'] for r in pure),
        no_audio_no_name_delay_at_least1h=sum(r['nominal_oldest_delay_seconds'] is not None and r['nominal_oldest_delay_seconds']>=3600 for r in pure),
        reasons=dict(collections.Counter(reason for r in rows for reason in r['reasons'])),
        day1_files=len(videos),day1_video_bytes=sum(f['size'] for f in videos),
        actual_video_hours_verified=0,admitted_questions=0,video_bytes_downloaded=0,
        qa_participant='A1_JAKE only; do not assume all6 participants have public official QA',
        next='scope exact video subset and byte budget; actual PTS/gaps and pure visual witnesses before admission',
        limitations=['flags not proof questions answerable from pixels','multi-target strings and multiple witnesses preserved for manual audit',
            'filename time not verified media duration','potential1h questions not evidence of context overflow'],rows=rows,
        candidate_day1_video_list=[dict(path=f['path'],bytes=f['size'],sha256=f['lfs']['oid']) for f in videos])
    with (OUT/'metadata_screening500.json').open('x',encoding='utf-8') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
    print({k:v for k,v in plan.items() if k not in ('rows','candidate_day1_video_list')})


if __name__=='__main__':main()
