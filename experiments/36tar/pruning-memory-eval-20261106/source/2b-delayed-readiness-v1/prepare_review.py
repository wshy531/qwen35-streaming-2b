"""Freeze a small source-stratified content review batch, never model-filtered."""
import argparse
import hashlib
import json
from pathlib import Path
import av
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT/'runs/delayed_training_candidates_20261104_01'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--media-root',type=Path,required=True)
    args = parser.parse_args()
    audit = [json.loads(x) for x in (PREP/'media_audit.jsonl').read_text().splitlines()]
    assert len(audit)==150
    sources = {r['video_id']:r for r in json.loads((PREP/'acquisition_plan.json').read_text())['files']}
    questions = json.loads((PREP/'questions_LOCAL_ONLY.json').read_text())
    selected = []
    for role in ('train_candidate','development_candidate'):
        for lo,hi in ((24,60),(60,10000)):
            candidates = [r for r in audit if r['media_check_passed'] and r['role']==role and lo<=r['duration_s']<hi]
            candidates.sort(key=lambda r:hashlib.sha256(('review-v1:'+r['video_id']).encode()).hexdigest())
            count = 4 if role=='train_candidate' else 2
            assert len(candidates)>=count
            selected.extend(candidates[:count])
    out = PREP/'review_batch01'
    out.mkdir(exist_ok=False)
    records = []
    for row in selected:
        source = sources[row['video_id']]
        samples = row['samples']
        overview = sorted(set(round(i*(len(samples)-1)/23) for i in range(24)))
        groups = [('overview',overview),('first24s_A',list(range(24))),('first24s_B',list(range(24,48)))]
        required = {samples[i]['frame_index']:i for _,indices in groups for i in indices if i<len(samples)}
        images = {}
        with av.open(str(args.media_root/source['mirror_path'])) as container:
            stream = container.streams.video[0];stream.thread_count=1
            for index,frame in enumerate(container.decode(stream)):
                if index not in required:
                    continue
                i = required[index]
                inner = ImageOps.contain(frame.to_image().convert('RGB'),(384,288),Image.Resampling.BICUBIC)
                canvas = Image.new('RGB',(384,288))
                canvas.paste(inner,((384-inner.width)//2,(288-inner.height)//2))
                assert hashlib.sha256(canvas.tobytes()).hexdigest()==samples[i]['rgb_sha256']
                images[i]=canvas
        sheets = []
        for name,indices in groups:
            sheet = Image.new('RGB',(384*4,312*6),'white')
            draw = ImageDraw.Draw(sheet)
            for slot,i in enumerate(indices):
                if i not in images:
                    continue
                x,y = (slot%4)*384,(slot//4)*312
                sheet.paste(images[i],(x,y))
                draw.text((x+4,y+290),f"{row['video_id']} | {samples[i]['pts_s']:.3f}s | sample {i}",fill='black')
            filename = row['video_id']+'_'+name+'.jpg'
            sheet.save(out/filename,quality=90)
            sheets.append(filename)
        records.append(dict(video_id=row['video_id'],role=row['role'],duration_s=row['duration_s'],
            media_sha256=source['mirror_lfs_sha256'],sheets=sheets,
            questions=[q for q in questions if q['video_id']==row['video_id']],
            content_review_completed=False,delayed_QA_admitted=False))
    with (out/'review_materials_LOCAL_ONLY.json').open('x',encoding='utf-8') as f:
        json.dump(dict(selection='fixed hash, role and duration strata; no model results',
                       source_count=len(records),sources=records,
                       caveat='Overview is sparse. First24s sheets show exact model RGB. Review alone does not prove lack of later cues.'),f,indent=2)
    print(json.dumps(dict(sources=len(records),sheets=len(records)*3,admitted=0)))


if __name__=='__main__':
    main()
