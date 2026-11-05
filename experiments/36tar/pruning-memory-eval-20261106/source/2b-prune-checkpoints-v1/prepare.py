"""Freeze a development-only checkpoint audit, without selecting examples by score."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
P=R/'runs/prune_checkpoints_prepare_20261102_01'
OLD=R/'runs/learned_prune_main_prepare_20261101_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def resolve(n):
    p=R/n
    if not p.exists() and n.startswith('repo/'):p=R/'publish/qwen35-streaming-2b'/n[5:]
    return p
def main():
    P.mkdir(exist_ok=False)
    q=[q for q in read(OLD/'eval.json') if q['role']=='development']
    vids={x['video_id'] for x in q};ids={x['id'] for x in q}
    assert len(q)==len(ids)==345 and len(vids)==40
    train=read(OLD/'train.json')
    assert not vids & {x['video_id'] for x in train}
    assert not ids & {x['id'] for x in train}
    sources=[s for s in read(OLD/'sources.json') if s['video_id'] in vids]
    rows=[json.loads(s) for s in (R/'runs/hard-mask-main-220062/answers.jsonl').read_text().splitlines()]
    ref=[r for r in rows if r['id'] in ids and r['shift']==0 and r['arm'] in ('mask_initial','mask_ce')]
    assert len(ref)==len({(r['id'],r['arm']) for r in ref})==690
    checkpoints={'step0':'runs/learned-prune-main-220024/scorer_untrained.pt'}
    checkpoints.update({f'step{k}':f'runs/hard-mask-main-220062/mask_ce_step{k}.pt' for k in (310,619,928,1237)})
    audit={r['video_id']:r for r in read(R/'runs/learned_prune_data_prepare_20261101_01/CPU_media_audit.json')['rows']}
    pairs=sum((len(audit[v]['samples'])+1)//2 for v in vids)
    spec=dict(questions=345,sources=40,arms=list(checkpoints),checkpoints=checkpoints,
        answers=1725,endpoint_replays=690,new_intermediate_answers=1035,validation_loss_forwards=1725,
        visual_calls=pairs*5,max_generated_tokens=1725*8,model_loads=1,updates=0,
        phases=[['step0','step1237'],['step310','step619','step928']],
        options='original order only',selection='all development questions; no result filtering',
        budget=dict(partition='g078t',comment='gx_llm',GPU=1,cpus=4,memory_GiB=24,minutes=60,output_MiB=64),
        limit='Posthoc development diagnostic, not unseen testing or independent cost. No automatic promotion.',
        decision='Compare all fixed points once. Development correct count primary, then CE, then earliest step for ties. If similar, stop training-duration exploration. Freeze any proposed choice before regression; never select on regression.')
    save(P/'questions.json',q);save(P/'sources.json',sources);save(P/'reference.json',ref);save(P/'protocol.json',spec)
    lock=read(R/'runs/hard_mask_main_prepare_20261102_01/execution_lock.json')
    for n,h in lock.items():assert sha(resolve(n))==h,n
    extra=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))+list(Path(__file__).parent.glob('*.md'))
    extra += [P/n for n in ('questions.json','sources.json','reference.json','protocol.json')]
    extra += [R/n for n in checkpoints.values()]
    for p in extra:lock[p.relative_to(R).as_posix()]=sha(p)
    save(P/'execution_lock.json',lock)
    print(json.dumps(dict(protocol=spec,locked_files=len(lock),lock_sha256=sha(P/'execution_lock.json'))))
if __name__=='__main__':main()
