"""Pin all official questions on the existing24 comparison sources, no models."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from score_memory_eval import digest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/natural_eval_v2_20261019_01'
BASE=ROOT/'runs/dataset_build_20261018_01'
DATA=ROOT/'experiments/2b-memory-eval/data'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    old=read(DATA/'candidate_manifest.json');oldids={q['id'] for q in old}
    selected={q['source_id'] for q in old if q['dataset']=='nextgqa' and q['split']=='comparison'}
    media={m['source_path']:m for m in read(BASE/'content_release_v1/media_inventory.json')['files']}
    mapping=read(DATA/'upstream/nextgqa/map_vid_vidorID.json');g=read(DATA/'upstream/nextgqa/gsub_val.json')
    rows=[]
    for r in csv.DictReader((DATA/'upstream/nextgqa/val.csv').open(encoding='utf-8')):
        vid=r['video_id']
        if vid not in selected:continue
        source=mapping[vid];asset=media[source];options=[r[f'a{i}'] for i in range(5)]
        assert options.count(r['answer'])==1 and asset['split']=='comparison' and asset['parent_identity_status']=='verified'
        duration=g[vid]['duration'];spans=g[vid]['location'].get(r['qid'],[])
        row=dict(id=f'nextgqa:val:{vid}:{r["qid"]}',dataset='nextgqa',source_id=vid,source_path=source,task=r['type'],
            question=r['question'],options=options,answer_index=options.index(r['answer']),evidence_intervals=spans,
            query_time_s=duration,query_policy='original_full_clip_end',duration_s=duration,split='comparison',
            source_group_id=asset['source_group_id'],parent_identity_status='verified',media_sha256=asset['sha256'],
            media_path=asset['media_path'],original235_member=f'nextgqa:val:{vid}:{r["qid"]}' in oldids)
        rows.append(row)
    rows.sort(key=lambda r:(r['source_path'],int(r['id'].split(':')[-1])))
    assert len(rows)==149 and len({r['source_group_id'] for r in rows})==24
    new=[r for r in rows if not r['original235_member']];assert len(new)==78
    save(OUT/'original149.json',rows);save(OUT/'additional78.json',new)
    traces=[]
    for t in read(BASE/'gqa48_actual_index.json')['videos']:
        if t['split']!='comparison':continue
        assert t['source_sha256']==media[t['source_path']]['sha256']
        traces.append(dict(source_path=t['source_path'],source_sha256=t['source_sha256'],fps=1.,canvas_hw=[192,256],frames=t['frames_1fps']))
    save(OUT/'sampling_1fps.json',traces)
    save(OUT/'scope_lock.json',dict(status='scope_and_annotation_protocol_locked_not_formal_model_evaluation',
        questions=149,additional_questions=78,comparison_sources=24,additional_independent_videos=0,
        original149_sha256=sha(OUT/'original149.json'),additional78_sha256=sha(OUT/'additional78.json'),
        upstream_sha256={n:sha(DATA/'upstream/nextgqa'/n) for n in ['val.csv','gsub_val.json','map_vid_vidorID.json']},
        unchanged_raw235_sha256=sha(DATA/'candidate_manifest.json'),sampling_1fps_sha256=sha(OUT/'sampling_1fps.json'),
        source_split_rule='same canonical24 comparison video IDs; no development/holdout media reassignment',
        review_rule='one pass over existing sheets; preserve original wording/options/GT; no model outputs; old unresolved cases stay unresolved',
        sampling_rule='1FPS first real frame at/after each target, query cutoff, existing192x256 letterbox; same for every arm; no annotation-guided model input',
        sampling_reason='151/151 prior37 witness frames match1FPS, only66/151 match.5FPS; development CPU trace also exists; no model score used to choose sampling',
        evaluation_strata=['observable_fact_or_action','temporal_relation','contextual_explanation','supplementary_or_unresolved'],
        difficulty_rule='annotated temporal operations and evidence-age/retention pressure; never claim calibrated easy/medium/hard before results',
        stop='Complete78 one-pass dispositions and full149 accounting, then publish counts, limitations and readiness; no GPU before dataset arrangement.',
        pretrained_loads=0,holdout_images=0,new_downloads=0,new_screenshots=0,canonical_dedup_level='author video ID, not proof of distinct filming sessions'))
    print(json.dumps(dict(status='scope_locked',videos=24,questions=149,new_questions=78,tasks=dict(Counter(q['task'] for q in rows))),ensure_ascii=False))


if __name__=='__main__':main()
