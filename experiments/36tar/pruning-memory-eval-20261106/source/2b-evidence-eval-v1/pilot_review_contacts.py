"""Small reviewer sheets only, no alteration to evaluated images."""
import json
from pathlib import Path
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_eval_prepare_20261102_01'
materials=json.loads((R/'runs/evidence-nextqa-materials-20261102-01/materials.json').read_text())
cases=json.loads((P/'real_pilot_LOCAL_ONLY.json').read_text());sources={c['source_id'] for c in cases}
ids={m['source_id']:[f['frame_id'] for f in m['frames']] for m in materials if m['source_id'] in sources}
(P/'real_pilot_material_ids.json').write_text(json.dumps(ids,indent=2)+'\n')
for m in materials:
    if m['source_id'] not in ('9520717551','5002226482','5954391577'):continue
    frames=m['frames'];sheet=Image.new('RGB',(1536,312*((len(frames)+3)//4)),(35,35,35));draw=ImageDraw.Draw(sheet)
    for i,f in enumerate(frames):
        with Image.open(R/f['path']) as im:sheet.paste(im,((i%4)*384,(i//4)*312))
        draw.text(((i%4)*384+3,(i//4)*312+289),f'{f["frame_id"]} / {f["pts_s"]:.3f}s',fill='white')
    sheet.save(R/f'runs/evidence-real-judge-20261102-01/review-{m["source_id"]}.jpg',quality=85)
print('three review sheets and six source citation inventories written')
