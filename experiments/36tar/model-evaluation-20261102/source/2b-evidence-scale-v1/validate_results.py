"""Verify saved evidence and method coverage, without another model/API call."""
import argparse,hashlib,json,collections
from pathlib import Path
from PIL import Image
R=Path(__file__).resolve().parents[2];P=R/'runs/evidence_scale_prepare_20261102_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    panel={q['id']:q for q in read(P/'panel.json')};seen=set();files={};count=0;checked=[]
    for run in a.runs:
        done=read(run/'completed.json');assert done['status']=='complete_generation_only'
        rows=[json.loads(l) for l in (run/'answers.jsonl').read_text(encoding='utf-8').splitlines()];assert len(rows)==750
        materials={m['material_id']:m for m in read(run/'materials.json')};assert len(materials)==150
        by=collections.defaultdict(dict)
        for r in rows:
            key=(r['id'],r['arm']);assert key not in seen;seen.add(key);by[r['id']][r['arm']]=r
            assert len(r['generated_ids'])<=128 and r['input_tokens']+128<=4096
            assert r['source_id']==panel[r['id']]['source_id'];count+=1
        for qid,arms in by.items():
            assert set(arms)=={'full','fixed12','heuristic11','mask_ce','text_only'}
            n=len(arms['full']['visible_pairs']);assert n>0
            for arm,row in arms.items():
                assert row['query_time_s']==panel[qid]['query_time_s']
                assert row['visual_tokens']==(0 if arm=='text_only' else n*(108 if arm=='full' else 54))
                assert row['visible_pairs']==([] if arm=='text_only' else arms['full']['visible_pairs'])
            if panel[qid]['dataset']=='StreamingBench':assert n<=8
            m=materials[qid];assert m['source_id']==panel[qid]['source_id']
            assert m['frames'] and all(f['pts_s']<=panel[qid]['query_time_s']+1e-6 for f in m['frames'])
            for f in m['frames']:
                path=(R/f['path']).resolve();assert path.is_relative_to((R/'runs').resolve())
                if str(path) in files:assert files[str(path)]==f['rgb_sha256'];continue
                with Image.open(path) as im:
                    assert im.size==(384,288);assert hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()==f['rgb_sha256']
                files[str(path)]=f['rgb_sha256']
        checked.append(dict(run=run.as_posix(),dataset=done['dataset'],rows=750,questions=150,frame_contract_passed=True))
    result=dict(status='passed',datasets=checked,rows=count,unique_pngs=len(files),png_bytes=sum(Path(p).stat().st_size for p in files),
        all_five_arms=True,no_future_sampled_images=True,pixel_hashes_verified=True,paid_API=0,GPU=0)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
