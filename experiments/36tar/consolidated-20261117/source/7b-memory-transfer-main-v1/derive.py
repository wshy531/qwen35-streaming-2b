"""Generate a minimal, reviewable derivative of the successful GPU profile."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def build():
    source=ROOT/'experiments/7b-memory-transfer-v1/run_cost.py'
    text=source.read_text(encoding='utf-8')
    edits=[
      ('from readout import Memory',"import sys\nsys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/7b-memory-transfer-v1'))\nfrom readout import Memory"),
      ("MODEL=ROOT/'models/Qwen2.5-VL-7B-Instruct'", "MODEL=ROOT/'models/Qwen2.5-VL-7B-Instruct'\nFORMAL=ROOT/'runs/7b_formal_plan_20261107_01'\nRUN=ROOT/'runs/7b_formal_execution_20261107_01'"),
      ('def main(out):','def main(out,part):'),
      ("    assert read(PREP/'asset_completed.json')['files']==15", "    assert read(PREP/'asset_completed.json')['files']==15\n    for rel,digest in read(RUN/'execution_lock.json').items():assert sha(ROOT/rel)==digest,rel\n    assert read(RUN/'CPU_cluster.json')['passed'] and read(RUN/'CPU_local.json')['passed']\n    assert read(ROOT/'runs/7b-memory-profile-221557/qualification.json')['passed']\n    assert read(ROOT/'runs/7b-memory-profile-221557/completed.json')['answers']==20\n    budget=read(FORMAL/'plan.json')['parts'][part-1]\n    assert budget['part']==part\n    reference={(r['id'],r['arm']):r for r in [json.loads(line) for line in (ROOT/'runs/7b-memory-profile-221557/answers.jsonl').read_text(encoding='utf-8').splitlines()]}\n    reference_matches=0"),
      ("if time.monotonic()-start>57*60:raise TimeoutError('57-minute internal stop')", "if time.monotonic()-start>117*60:raise TimeoutError('117-minute internal stop')"),
      ("'7b-memory-slurm-'","'7b-transfer-main-'"),
      ("tasks=read(PREP/'profile.json');traces=read(PREP/'traces.json');groups={};qualified=False", "tasks=read(FORMAL/f'part-{part}.json');traces=read(PREP/'traces.json');groups={};qualified=True\n        assert len(tasks)==budget['questions']\n        expected_references=sum((q['id'],arm) in reference for q in tasks for arm in ARMS)"),
      ('profile_only=True','profile_only=False'),
      ("                            source_answers.append(dict(id=q['id']", "                            if (q['id'],arm) in reference:\n                                old=reference[(q['id'],arm)]\n                                for key in ('generated_ids','input_tokens','visual_tokens','nodes'):\n                                    assert result[key]==old[key],('profile regression',q['id'],arm,key)\n                                reference_matches+=1\n                            source_answers.append(dict(id=q['id']"),
      ("assert count['answers']==20 and count['tokens']<=2560 and count['pairs']==561 and count['vision']<=564", "assert count['answers']==budget['answers'] and count['tokens']<=128*budget['answers']\n        assert count['pairs']==count['vision']==budget['visual_pairs']\n        assert reference_matches==expected_references"),
      ("assert count['tokens']+count['qualification_tokens']<=2568", "assert count['qualification_tokens']==0"),
      ("save(out/'completed.json',dict(**count,seconds=time.monotonic()-start,profile_only=False,updates=0))", "save(out/'completed.json',dict(**count,seconds=time.monotonic()-start,profile_only=False,updates=0,part=part,reference_matches=reference_matches))"),
      ("p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)", "p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--part',type=int,choices=range(1,5),required=True)\n    args=p.parse_args();main(args.out,args.part)")]
    for old,new in edits:
        expected=2 if old=='profile_only=True' else 1
        assert text.count(old)==expected,(old,text.count(old))
        text=text.replace(old,new)
    compile(text,'run_main.py','exec')
    return text,dict(source=source.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),edits=edits)

if __name__=='__main__':
    text,receipt=build()
    with (HERE/'run_main.py').open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    receipt['target_sha256']=hashlib.sha256((HERE/'run_main.py').read_bytes()).hexdigest()
    (HERE/'derivation.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(receipt['target_sha256'])
