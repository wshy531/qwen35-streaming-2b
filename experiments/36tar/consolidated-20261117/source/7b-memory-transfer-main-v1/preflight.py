"""Check derived entry, frozen complete split and optionally exact remote media."""
import argparse,ast,hashlib,json,sys,time
from pathlib import Path
from derive import build
ROOT=Path(__file__).resolve().parents[2]
PREP=ROOT/'runs/7b_transfer_prepare_20261107_01'
PLAN=ROOT/'runs/7b_formal_plan_20261107_01'
OUT=ROOT/'runs/7b_formal_execution_20261107_01'
sys.path.insert(0,str(ROOT/'experiments/7b-memory-transfer-v1'))
from run_cost import sha,read,save
from readout import ARMS

def main(media):
    OUT.mkdir(exist_ok=True)
    actual=(Path(__file__).parent/'run_main.py').read_text(encoding='utf-8')
    expected,receipt=build();assert actual==expected
    original=ast.parse((ROOT/receipt['source']).read_text(encoding='utf-8'))
    derived=ast.parse(actual)
    for name in ('sha','read','safe','save','qualify'):
        a=next(n for n in original.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
        b=next(n for n in derived.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name
    for rel,digest in read(PREP/'execution_lock.json').items():assert sha(ROOT/rel)==digest,rel
    for rel,digest in read(PLAN/'input_lock.json').items():assert sha(ROOT/rel)==digest,rel
    assert read(ROOT/'runs/7b-memory-profile-221557/completed.json')['answers']==20
    assert read(ROOT/'runs/7b-memory-profile-221557/qualification.json')['passed']
    questions=read(PREP/'questions.json');traces=read(PREP/'traces.json');plan=read(PLAN/'plan.json')
    refs=[json.loads(x) for x in (ROOT/'runs/7b-memory-profile-221557/answers.jsonl').read_text(encoding='utf-8').splitlines()]
    ref_keys={(r['id'],r['arm']) for r in refs};seen=[];seen_sources=set();checks=[];media_rows={}
    for b in plan['parts']:
        qs=read(PLAN/f"part-{b['part']}.json");pairs={};r=0
        assert len(qs)==b['questions'] and len(qs)*len(ARMS)==b['answers']
        for q in qs:
            key=q['suite']+'|'+q['source_id'];t=traces[key]
            n=sum(x['pts_s']<=q['query_time_s']+1e-6 for x in t['sampled_trace'])//2 if q['suite']=='D18' else t['states'][q['id']]['complete_pairs']
            assert 1<=n<=502
            pairs[key]=max(pairs.get(key,0),n)
            path=Path(t['media_path']);path=path if path.is_absolute() else ROOT/path
            expected_sha=q['media_sha256']
            if str(path) in media_rows:assert media_rows[str(path)]['sha256']==expected_sha
            else:media_rows[str(path)]=dict(path=str(path),sha256=expected_sha)
            r+=sum((q['id'],arm) in ref_keys for arm in ARMS)
        assert not seen_sources.intersection(pairs)
        seen_sources.update(pairs);seen+=qs
        assert sum(pairs.values())==b['visual_pairs'] and len(pairs)==b['sources']
        checks.append(dict(part=b['part'],questions=len(qs),answers=len(qs)*5,pairs=sum(pairs.values()),profile_regressions=r))
    assert sorted(seen,key=lambda q:q['id'])==sorted(questions,key=lambda q:q['id'])
    assert len(seen)==len({q['id'] for q in seen})==228
    assert sum(r['profile_regressions'] for r in checks)==20
    assert sum(r['pairs'] for r in checks)==27394 and len(seen_sources)==201
    assert ARMS==('recent8frames','recent16frames','complete_history','LS','lexical')
    # Inherited real-processor contracts cover all 1140 token/position layouts.
    for name in ('CPU_local.json','CPU_cluster.json'):
        c=read(PREP/name);assert c['passed'] and c['layout_count']==1140
        assert c['layout_sha256']=='1523a123a820d2f277dfcda1a3166eb387a9aefd4279ab3e070438ff6feb8fbb'
    if media:
        assert sys.platform=='linux'
        for i,row in enumerate(media_rows.values(),1):
            p=Path(row['path']);assert p.is_file() and sha(p)==row['sha256'],str(p)
            row['bytes']=p.stat().st_size
            if i%25==0:print(json.dumps(dict(media_verified=i,total=len(media_rows))),flush=True)
        save(OUT/'media_verified.json',list(media_rows.values()))
    result=dict(passed=True,part_checks=checks,layout_count=1140,media_verified=len(media_rows) if media else 0,
        inherited_real_pixel_and_micro_QA=True,new_GPU=0,run_main_sha256=sha(Path(__file__).parent/'run_main.py'))
    save(OUT/('CPU_cluster.json' if media else 'CPU_local.json'),result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--media',action='store_true');main(p.parse_args().media)
