"""Freeze simple node controls; reuse only identical subsets from locked runs."""
import sys,json,hashlib,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/7b-head-stability-v1'))
import run_eval as prior
OUT=ROOT/'runs/7b_head_simple_prepare_20261112_01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def targets(n,c,s):
    return dict(first=[0] if n else [],last=[n-1] if n else [],
        control_first=list(range(len(c))),control_last=list(range(n-len(c),n)),
        small_first=list(range(len(s))),small_last=list(range(n-len(s),n)))
def main():
    prior.check_new();OUT.mkdir(exist_ok=True)
    tasks=read(prior.HERE/'questions.json');ar=ROOT/'runs/7b-pruned-head-adapt-223422/final';sr=ROOT/'runs/7b-head-stability-223607/final'
    old={r['id']:r for r in rows(ar/'routes.jsonl')};new={r['id']:r for r in rows(sr/'routes.jsonl') if r.get('kind')=='repair'}
    cached={};identities={}
    for root in (ar,sr):
        file=root/'answers.jsonl';identities[file.relative_to(ROOT).as_posix()]=sha(file)
        for r in rows(file):
            q=r['id'];n=(len(old[q]['predicted_gains'])).bit_length()-1
            mapping={'recent8':[],'LS':list(range(n)),'head':old[q]['old_choice'],'adapted':old[q]['choice'],
                'adapted_same_count_earliest':list(range(len(old[q]['choice']))),'control':new[q]['control_choice'],'small_step':new[q]['small_choice']}
            assert r['route'] in mapping
            choice=mapping[r['route']];key=(q,tuple(choice),r['shift'])
            if key in cached:prior.check_reference(r,cached[key]['row'])
            else:cached[key]=dict(row=r,choice=choice,source=file.relative_to(ROOT).as_posix(),sha=sha(file))
    controls=set(read(prior.NEW/'control_ids.json'));frozen=[];missing=0;reused=0
    for q in tasks:
        qid=q['id'];n=(len(old[qid]['predicted_gains'])).bit_length()-1
        m=targets(n,new[qid]['control_choice'],new[qid]['small_choice'])
        unique=sorted(set(map(tuple,m.values())));cache=[];todo=[]
        for c in unique:
            if all((qid,c,s) in cached for s in (0,2)):
                cache += [cached[qid,c,s] for s in (0,2)];reused+=2
            else:todo.append(list(c));missing+=2
        ref=cached[qid,tuple(range(n)),0] if qid in controls else None
        frozen.append(dict(question=q,n_history=n,methods=m,cached=cache,todo=todo,reference=ref))
    save(OUT/'tasks.json',frozen)
    plan=dict(questions=210,method_rows=2520,unique_new_scores=missing,reused_unique_scores=reused,reference_scores=len(controls),
        max_scores=missing+len(controls),vision_bound=26965,representations=0,GPUs=1,CPU=4,host_GiB=64,minutes=180,
        generated_tokens=0,training_updates=0,source_files=identities)
    save(OUT/'plan.json',plan);print(json.dumps(plan))
    files=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'run_eval.sh',OUT/'tasks.json',OUT/'plan.json']
    save(OUT/'lock.json',{p.relative_to(ROOT).as_posix():sha(p) for p in files})
    with tarfile.open(OUT/'transfer.tar','w') as t:
        for p in files+[OUT/'lock.json']:t.add(p,arcname=p.relative_to(ROOT).as_posix())
if __name__=='__main__':main()
