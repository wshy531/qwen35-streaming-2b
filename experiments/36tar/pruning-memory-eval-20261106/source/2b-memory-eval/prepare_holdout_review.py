"""Locked holdout94 annotation plan, causal1FPS traces and exact-canvas sheets.

No model output, weights, relabeling or automatic formal admission. Author
intervals guide reviewer images only; final model prefixes remain uniform1FPS.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import av
from PIL import Image,ImageDraw,ImageFont
from audit_sampling import canvas_bytes,sample
from audit_development_evidence import crosstask_question,normalize
from prepare_grounded_review import select_witnesses

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'runs/dataset_build_20261018_01'
OUT=ROOT/'runs/holdout_qualification_20261019_01'
DATA=ROOT/'experiments/2b-memory-eval/data'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def annotations():
    coin=read(ROOT/'runs/development_audit_20261017_02/COIN.json')['database']
    cook=read(BASE/'source_metadata/youcookii_annotations_trainval_coot_copy.json')['database']
    result={}
    for key,value in coin.items():
        result['COIN/'+key+'.mp4']=dict(member='official_COIN.json/'+key,
            steps=[dict(action=normalize(a['label']),start_s=a['segment'][0],end_s=a['segment'][1]) for a in value['annotation']])
    for key,value in cook.items():
        result['youcook2/'+key+'.mp4']=dict(member='youcookii_annotations_trainval_coot_copy.json/'+key,
            steps=[dict(action=normalize(a['sentence']),start_s=a['segment'][0],end_s=a['segment'][1]) for a in value['annotations']])
    for p in (BASE/'source_metadata').glob('hirest_all_data_*.json'):
        for task,videos in read(p).items():
            for key,value in videos.items():
                a,b=value['bounds']
                result[f'hirest/{Path(key).stem}_{a}_{b}.mp4']=dict(member=p.name+'/'+task+'/'+key,
                    steps=[dict(action=normalize(s['heading']),start_s=s['absolute_bounds'][0]-a,end_s=s['absolute_bounds'][1]-a) for s in value['steps']])
    return result


def temporal(row,ann):
    entry=ann.get(row['source_path'])
    if not entry:return dict(status='source_annotation_or_pattern_unavailable')
    # Existing helper expects stem-keyed source dictionaries.
    return crosstask_question(row,{Path(row['source_path']).stem:entry})


def prepare():
    raw=read(DATA/'candidate_manifest.json');holdout=[q for q in raw if q['split']=='holdout']
    assert len(holdout)==94 and len({q['source_path'] for q in holdout})==40
    assert sha(DATA/'candidate_manifest.json')=='cd1d888eb0a0c00f300649f41800e60fd02b002e91f41f154b42e041fcc28161'
    assets={m['source_path']:m for m in read(BASE/'content_release_v1/media_inventory.json')['files']}
    ann=annotations();entries=[]
    for q in holdout:
        m=assets[q['source_path']];audit=temporal(q,ann) if q['task']=='ASI' else None
        flags=[]
        if m['parent_identity_status']!='verified':flags.append('parent_recording_unresolved')
        if q['task']=='HLD':flags.append('negative_full_prefix_not_reviewed')
        if m['content_domain']=='simulated_tour_of_real_scanned_scene':flags.append('simulated_scene_supplement_not_natural_human_action_core')
        if audit and audit['status']=='future_answer_evidence_not_a_past_recall_probe':flags.append('future_action_not_past_recall')
        if q['source_path']=='star/DHFA6.mp4':flags.append('weak_cross_split_visual_candidate_pending_adjudication')
        entries.append(dict(original=q,source_identity=m,source_time_audit=audit,previsual_flags=flags,
            image_review_planned=not any(f in flags for f in ['parent_recording_unresolved','negative_full_prefix_not_reviewed',
                'simulated_scene_supplement_not_natural_human_action_core','future_action_not_past_recall'])))
    OUT.mkdir(exist_ok=False);save(OUT/'original94.json',holdout);save(OUT/'previsual_metadata94.json',entries)
    media={q['original']['source_path'] for q in entries if q['image_review_planned']}
    # Two fixed source-level witnesses for the single pre-existing overlap flag.
    media.update(['star/DHFA6.mp4','star/X4DOF.mp4'])
    plan=[]
    for source in sorted(media):
        asset=assets[source];qs=[e for e in entries if e['original']['source_path']==source]
        maxquery=max((e['original']['query_time_s'] for e in qs),default=max(q['query_time_s'] for q in raw if q['source_path']==source))
        spans=[]
        for e in qs:
            spans.extend(e['original'].get('evidence_intervals') or [])
            if e['source_time_audit']:spans.extend(e['source_time_audit'].get('answer_intervals') or [])
        plan.append(dict(source_path=source,asset=asset,max_query_s=maxquery,
            question_ids=[e['original']['id'] for e in qs],author_intervals=sorted({tuple(x) for x in spans}),
            selection_rule='existing select_witnesses(limit24) plus uniform causal anchors if <24; source overlap pair uses uniform anchors',
            max_annotation_frames=24))
    save(OUT/'annotation_plan.json',dict(status='committed_before_first_holdout_image_review',sources=plan,
        source_count=len(plan),raw_questions=94,raw_videos=40,maximum_frames_per_source=24,
        image_canvas_hw=[192,256],disk_cap_bytes=80*2**20,prepare_wall_seconds=600,
        per_source_seconds=120,all_model_inputs_fps=1.,query_cutoff='pts_not_after_query',
        no_model_outputs_seen=True,no_model_runs=True,no_original_question_changed=True,
        metadata_sha256=sha(OUT/'previsual_metadata94.json'),original94_sha256=sha(OUT/'original94.json'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('audit_sampling.py'),Path(__file__).with_name('prepare_grounded_review.py')]},
        stop='one bounded visual pass; ambiguous remains excluded, no score-guided screenshots'))
    print(json.dumps(dict(status='annotation_plan_locked',sources_to_decode=len(plan),holdout_sources=40,questions=94,
        metadata_flags=dict(Counter(f for e in entries for f in e['previsual_flags'])))))


def render():
    start=time.monotonic();plan=read(OUT/'annotation_plan.json')
    assert sha(OUT/'previsual_metadata94.json')==plan['metadata_sha256']
    folder=OUT/'sheets';folder.mkdir(exist_ok=False)
    traces=[];receipts=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    for item in plan['sources']:
        if time.monotonic()-start>600:raise TimeoutError('bounded holdout preparation wall')
        asset=item['asset'];path=ROOT/asset['media_path'];assert sha(path)==asset['sha256']
        trace=sample(path,item['max_query_s'],1.,seconds=120)
        trace.update(source_path=item['source_path'],source_sha256=asset['sha256'],media_path=asset['media_path'],max_query_s=item['max_query_s'])
        traces.append(trace);frames=trace['frames'];spans=item['author_intervals']
        selected=select_witnesses(frames,spans,24) if spans else []
        seenids={f['frame_id'] for f in selected}
        for i in range(24):
            f=frames[round((len(frames)-1)*i/23)]
            if len(selected)>=24:break
            if f['frame_id'] not in seenids:selected.append(f);seenids.add(f['frame_id'])
        selected.sort(key=lambda f:f['pts_s'])
        # Selection is committed to disk before rendering/viewing each source.
        save(folder/('selection-'+item['source_path'].replace('/','_').replace('.mp4','')+'.json'),dict(source_path=item['source_path'],frames=selected,
            selected_before_viewing=True,annotation_only_not_model_input_selection=True))
        pages=[Image.new('RGB',(1024,890),(18,18,18)) for _ in range((len(selected)+15)//16)]
        for i,page in enumerate(pages):ImageDraw.Draw(page).text((4,4),item['source_path']+f' | native256x192 | p{i+1}',font=font,fill='white')
        wanted={f['frame_id']:(i,f) for i,f in enumerate(selected)};witnesses=[];origin=None;source_start=time.monotonic()
        with av.open(str(path)) as c:
            stream=c.streams.video[0];stream.codec_context.thread_count=1
            for fid,frame in enumerate(c.decode(stream)):
                if time.monotonic()-source_start>120:raise TimeoutError('source renderer bound')
                if origin is None:origin=float(frame.time)
                if fid not in wanted:continue
                i,expected=wanted[fid];pts=float(frame.time)-origin;rgb=canvas_bytes(frame)
                assert abs(pts-expected['pts_s'])<1e-7 and pts<=item['max_query_s']+1e-9 and hashlib.sha256(rgb).hexdigest()==expected['rgb_sha256']
                page,cell=i//16,i%16;x=cell%4*256;y=30+cell//4*214
                pages[page].paste(Image.frombytes('RGB',(256,192),rgb),(x,y))
                ImageDraw.Draw(pages[page]).text((x+2,y+194),f'id {fid} PTS {pts:.3f}',font=font,fill='white')
                witnesses.append(dict(**expected,page=page+1,cell=cell,hash_verified=True,source_path=item['source_path']))
                if len(witnesses)==len(wanted):break
        assert len(witnesses)==len(wanted)
        files=[];name=item['source_path'].replace('/','_').replace('.mp4','')
        for i,page in enumerate(pages):
            p=folder/f'{name}-p{i+1}.png';page.save(p);files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
        receipts.append(dict(source_path=item['source_path'],images=files,witnesses=witnesses))
        save(OUT/'sampling_1fps.json',traces);save(OUT/'image_receipts.json',receipts)
        assert sum(p.stat().st_size for p in folder.iterdir())<=plan['disk_cap_bytes']
        print(json.dumps(dict(source=item['source_path'],prefix_frames=len(frames),annotation_frames=len(witnesses),pages=len(files))),flush=True)
        del pages
    save(OUT/'render_completed.json',dict(status='rendered_not_yet_visually_reviewed',sources=len(receipts),
        images=sum(len(r['images']) for r in receipts),frames=sum(len(r['witnesses']) for r in receipts),
        seconds=time.monotonic()-start,new_media_bytes=0,models_loaded=0,images_disk_bytes=sum(p.stat().st_size for p in folder.iterdir()),
        plan_sha256=sha(OUT/'annotation_plan.json')))
    print(read(OUT/'render_completed.json'))


if __name__=='__main__':
    import sys
    {'prepare':prepare,'render':render}[sys.argv[1]]()
