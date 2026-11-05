"""One bounded annotation pass for fixed OVO comparison sources, not holdout."""
import hashlib
import json
from pathlib import Path
import time

import av
from PIL import Image,ImageDraw,ImageFont,ImageOps

from audit_development_evidence import parse_crosstask,crosstask_question,normalize
from audit_sampling import canvas_bytes


def main():
    root=Path(__file__).resolve().parents[2];run=root/'runs/dataset_build_20261018_01'
    out=run/'ovo_comparison_sheets';out.mkdir(exist_ok=False)
    rows=json.loads((root/'experiments/2b-memory-eval/data/candidate_manifest.json').read_text(encoding='utf-8'))
    traces=json.loads((run/'ovo32_actual_index.json').read_text())['videos']
    cross=parse_crosstask(root/'runs/evidence_review_20261017_01/crosstask_release.zip')
    raw_coin=json.loads((root/'runs/development_audit_20261017_02/COIN.json').read_text())['database']
    coin={key:dict(member='official_COIN.json/'+key,steps=[dict(action=normalize(a['label']),
        start_s=a['segment'][0],end_s=a['segment'][1]) for a in value['annotation']]) for key,value in raw_coin.items()}
    plan=[];audits=[]
    for trace in traces:
        if trace['split']!='comparison':continue
        qs=[r for r in rows if r['dataset']=='ovo' and r['source_path']==trace['source_path']]
        frames=trace['frames_05fps'];priorities=[]
        for q in qs:
            prefix=[f for f in frames if f['pts_s']<=q['query_time_s']]
            priorities.append(prefix[-1])
            entry=crosstask_question(q,coin if q['source_path'].startswith('COIN/') else cross)
            audits.append(dict(id=q['id'],source_path=q['source_path'],**entry))
            if entry.get('answer_intervals'):
                for a,b in entry['answer_intervals']:
                    within=[f for f in prefix if a<=f['pts_s']<=b]
                    if within:priorities.extend([within[len(within)//2],within[0],within[-1]])
        priorities.extend(frames[round((len(frames)-1)*i/15)] for i in range(16))
        chosen={}
        for f in priorities:
            if len(chosen)==16:break
            chosen[f['frame_id']]=f
        plan.append(dict(source_path=trace['source_path'],local_path=trace['local_path'],
            source_sha256=trace['source_sha256'],questions=qs,
            frames=sorted(chosen.values(),key=lambda f:f['pts_s'])))
    assert len(plan)==16
    (out/'selection_plan.json').write_text(json.dumps(dict(purpose='annotation_only_not_model_input_selection',
        selected_before_viewing=True,sources=plan),indent=2)+'\n',encoding='utf-8')
    (run/'ovo_comparison_source_temporal_audit.json').write_text(json.dumps(audits,indent=2)+'\n',encoding='utf-8')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);receipts=[]
    for item in plan:
        sheet=Image.new('RGB',(1600,1080),(18,18,18));draw=ImageDraw.Draw(sheet)
        draw.text((8,3),item['source_path']+' | comparison annotation only | actual .5FPS',font=font,fill='white')
        wanted={f['frame_id']:(i,f) for i,f in enumerate(item['frames'])};seen=[];origin=None;start=time.monotonic()
        with av.open(item['local_path']) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=1
            for frame_id,frame in enumerate(container.decode(stream)):
                if time.monotonic()-start>120:raise TimeoutError('preview source budget')
                if origin is None:origin=float(frame.time)
                if frame_id not in wanted:continue
                i,expected=wanted[frame_id]
                if abs(float(frame.time)-origin-expected['pts_s'])>1e-7 or hashlib.sha256(canvas_bytes(frame)).hexdigest()!=expected['rgb_sha256']:
                    raise ValueError('preview differs from actual prefix')
                x=(i%4)*400;y=30+(i//4)*260;image=ImageOps.contain(frame.to_image(),(396,232))
                sheet.paste(image,(x+(400-image.width)//2,y))
                draw.text((x+4,y+234),f'id {frame_id} PTS {expected["pts_s"]:.3f}',font=font,fill='white')
                seen.append(dict(**expected,cell=i,hash_verified=True))
                if len(seen)==len(wanted):break
        if len(seen)!=len(wanted):raise ValueError('selected frame missing')
        name='ovo-'+Path(item['source_path']).stem+'.jpg';sheet.save(out/name,quality=90)
        receipts.append(dict(source_path=item['source_path'],image=name,frames=seen,elapsed_s=time.monotonic()-start))
        (out/'review_receipts.json').write_text(json.dumps(dict(status='generated_not_yet_reviewed',sources=receipts),indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(source=item['source_path'],frames=len(seen),hashes_verified=True)),flush=True)


if __name__=='__main__':main()
