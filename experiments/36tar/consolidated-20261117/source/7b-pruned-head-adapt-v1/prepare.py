"""Freeze unchanged sources and evaluation; never copy evaluation gold into worker."""
import math
from adapt import *

def main():
    HERE.mkdir(exist_ok=True)
    old=ROOT/'runs/7b_prune_head_prepare_20261111_01'
    groups=read(PREP/'fit_sources.json');tasks=read(old/'questions.json')
    train=[g for g in groups if g['role']=='train'];dev=[g for g in groups if g['role']=='development']
    assert (len(train),len(dev),len(tasks))==(72,16,210)
    eval_ids={q['source_id'] for q in tasks};eval_sha={q['media_sha256'] for q in tasks}
    assert not eval_ids & {g['source_id'] for g in groups}
    assert not eval_sha & {g['media_sha256'] for g in groups}
    assert not ({g['source_id'] for g in train}&{g['source_id'] for g in dev})
    assert not ({g['media_sha256'] for g in train}&{g['media_sha256'] for g in dev})
    assert not any(k in q for q in tasks for k in ('answer','answer_index','gold','correct'))
    assert {q['suite'] for q in tasks}=={'SB150','short60'}
    save(HERE/'fit_sources.json',groups);save(HERE/'questions.json',tasks)
    save(HERE/'traces.json',read(old/'traces.json'))
    refs=[json.loads(l) for l in (ROOT/'runs/7b-prune-head-223298/answers.jsonl').read_text().splitlines()]
    refs=[r for r in refs if r['mode']=='content' and r['route'] in ('recent8','LS','head')]
    assert len(refs)==1260
    save(HERE/'references.json',refs)
    routes=[json.loads(l) for l in (ROOT/'runs/7b-prune-head-223298/routes.jsonl').read_text().splitlines()]
    choices={r['id']:r['choice'] for r in routes if r['mode']=='content'};assert len(choices)==210
    save(HERE/'old_choices.json',choices)
    original=ROOT/'runs/7b_method_recovery_20261110_01/old_selector'
    fs=torch.load(original/'features.pt',weights_only=True,map_location='cpu')
    fs=[f for f in fs if f['role'] in ('train','development')];ids={f['id'] for f in fs}
    torch.save(fs,HERE/'recipe_features.pt')
    ss=[json.loads(l) for l in (original/'scores.jsonl').read_text().splitlines()]
    save(HERE/'recipe_scores.json',[r for r in ss if r['id'] in ids])
    bound=sum(math.ceil((max(q['query_time_s'] for q in g['questions'])+1)/2) for g in groups)
    eval_pairs=sum(t['pairs'] for t in read(HERE/'traces.json').values())
    plan=dict(hypothesis='same recipe, supervision measured on pruned bank',train_questions=144,development_questions=32,
        training_sources=72,development_sources=16,evaluation_questions=210,evaluation_sources=len(eval_ids),
        training_scoring_bound=176*32,evaluation_scoring=210*10,representations=386,
        training_vision_bound=bound,evaluation_vision=eval_pairs,vision_bound=bound+eval_pairs,
        trainable_parameters=713537,head_updates=180,epochs=20,base_updates=0,generated_tokens=0,
        GPUs=1,CPUs=4,memory_GiB=64,minutes=180,partition='g078t',comment='gx_llm',output_MiB=192,
        fresh_evaluation=False,automatic_retry=False,independent_end_to_end=False)
    save(HERE/'plan.json',plan)
    files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.sh'))
    files += [HERE/f for f in ('fit_sources.json','questions.json','traces.json','references.json','old_choices.json','plan.json','recipe_features.pt','recipe_scores.json')]
    save(HERE/'lock.json',{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)})
    print(json.dumps(plan))
if __name__=='__main__':main()
