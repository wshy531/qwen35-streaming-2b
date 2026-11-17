"""Freeze by source/type/duration/hash, never by method correctness."""
from collections import defaultdict
from common import *

def stratified(rows,n):
    buckets=defaultdict(list)
    for q in rows:
        t=q['query_time_s'];b='short' if t<180 else 'medium' if t<600 else 'long'
        buckets[(q['type'],b)].append(q)
    for k in buckets:buckets[k].sort(key=lambda q:hashlib.sha256(q['id'].encode()).hexdigest())
    out=[];used=set()
    while len(out)<n:
        progress=False
        for k in sorted(buckets):
            while buckets[k] and buckets[k][0]['source_id'] in used:buckets[k].pop(0)
            if buckets[k] and len(out)<n:
                q=buckets[k].pop(0);out.append(q);used.add(q['source_id']);progress=True
        if not progress:raise ValueError('insufficient distinct sources')
    return out

def main():
    PREP.mkdir(exist_ok=True)
    original=ROOT/'runs/7b_transfer_prepare_20261107_01'
    qs=read(original/'questions.json')
    selected=[q for q in qs if q['suite'] in ('D18','SB150')]
    seprep=ROOT/'runs/7b_sember_value_prepare_20261108_01'
    selected+=read(seprep/'questions.json')
    assert len(selected)==288
    keys=read(ROOT/'datasets/streaming-evaluation-v1/answer_keys_LOCAL_ONLY.json')
    keys.update({r['question_id']:r['correct_index'] for r in read(ROOT/'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json')})
    save(PREP/'evaluation_keys_LOCAL_ONLY.json',{q['id']:keys[q['id']] for q in selected})
    traces=read(original/'traces.json')
    for q in selected:
        if q['suite']=='S-EMBER-early-candidate':
            a=read(seprep/'traces'/f"{q['source_id']}.json")
            traces[q['suite']+'|'+q['source_id']]=dict(media_path=q['media_path'],pairs=a['pairs'])
    traces={q['suite']+'|'+q['source_id']:traces[q['suite']+'|'+q['source_id']] for q in selected}
    # The legacy D18 question list has a partner-relative path; the frozen trace
    # has its explicitly authorized, read-only absolute media path.
    for q in selected:
        q['media_path']=traces[q['suite']+'|'+q['source_id']]['media_path']
    save(PREP/'evaluation.json',selected);save(PREP/'traces.json',traces)
    refs={}
    for p in sorted((ROOT/'runs').glob('7b-transfer-main-p*/answers.jsonl')):
        for line in p.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['id'] in {q['id'] for q in selected} and r['arm'] in ('recent8frames','LS','complete_history'):
                refs[r['id']+'|'+r['arm']]=r
    for p in sorted((ROOT/'runs').glob('7b-sember-value-p*/answers.jsonl')):
        for line in p.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['id'] in {q['id'] for q in selected} and r['arm'] in ('recent16frames','LS','complete_history'):
                refs[r['id']+'|'+r['arm']]=r
    # Latest protocol: one visual block per node, without invented time labels.
    for p in [*sorted((ROOT/'runs').glob('7b-protocol-followup-p*/answers.jsonl')),ROOT/'runs/7b-time-protocol-221665/answers.jsonl']:
        for line in p.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['id'] in {q['id'] for q in selected} and (r.get('mode')=='segmented' or r['arm']=='complete_history' and r.get('mode')=='timestamped'):
                refs[r['id']+'|'+r['arm']]=r
    auditrefs={r['id']:r for r in refs.values() if r['arm']=='complete_history' and r.get('mode') in ('segmented','timestamped')}
    save(PREP/'audit_references.json',auditrefs)
    refs={k:r for k,r in refs.items() if r.get('mode')=='segmented'}
    save(PREP/'references.json',refs)
    # Spread audit lengths across available <= 16k-token full histories. Selection
    # does not consult whether any previous answer was correct.
    audit=[]
    for suite,n in [('SB150',6),('D18',2),('S-EMBER-early-candidate',4)]:
        eligible=[q for q in selected if q['suite']==suite and q['id'] in auditrefs and auditrefs[q['id']]['input_tokens']<=16384]
        eligible.sort(key=lambda q:(auditrefs[q['id']]['input_tokens'],q['id']))
        assert len(eligible)>=n,(suite,len(eligible))
        audit += [eligible[round(i*(len(eligible)-1)/(n-1))]['id'] for i in range(n)]
    assert len(set(audit))==12
    save(PREP/'audit_ids.json',audit)
    eval_sha={q['media_sha256'] for q in selected};eval_sources={q['source_id'] for q in selected}
    groups=[]
    for role,file,n in [('train','remaining_train.json',72),('development','remaining_train.json',16)]:
        source=read(ROOT/'runs/history_utility_main_prepare_20261104_01'/file)
        source=sorted(source,key=lambda g:hashlib.sha256(g['video_id'].encode()).hexdigest())
        source=source[:72] if role=='train' else source[72:88]
        assert len(source)==n
        for g in source:
            assert g['sha256'] not in eval_sha and g['video_id'] not in eval_sources
            items=sorted(g['questions'],key=lambda q:hashlib.sha256(q['id'].encode()).hexdigest())[:2]
            assert len(items)==2
            cutoff=g['audit']['samples'][-1]['pts_s']
            items=[dict(q,source_id=g['video_id'],suite=role,media_sha256=g['sha256'],media_path=g['path'],query_time_s=cutoff) for q in items]
            groups.append(dict(role=role,source_id=g['video_id'],media_path=g['path'],media_sha256=g['sha256'],questions=items))
    assert not ({g['media_sha256'] for g in groups if g['role']=='train'} & {g['media_sha256'] for g in groups if g['role']=='development'})
    save(PREP/'fit_sources.json',groups)
    pairs={}
    for q in selected:
        key=q['suite']+'|'+q['source_id'];s=traces[key]
        if isinstance(s.get('pairs'),list):n=sum(p['second']['pts_s']<=q['query_time_s']+1e-9 for p in s['pairs'])
        elif s.get('sampled_trace'):n=sum(p['pts_s']<=q['query_time_s']+1e-9 for p in s['sampled_trace'])//2
        else:n=s['states'][q['id']]['complete_pairs']
        pairs[key]=max(pairs.get(key,0),n)
    plan=dict(evaluation_questions=288,evaluation_sources=len(pairs),suites={'SB150':150,'D18':18,'S-EMBER-early-candidate':120},
        primary_questions=270,diagnostic_questions=18,
        prune_modes=['full','fixed','content'],routes=['recent8','LS'],prune_classifications=3456,audit_questions=12,
        visual_calls_prune=3*sum(pairs.values())+36+1,generated_tokens_max=12*128+12*384,
        train_sources=72,train_questions=144,development_sources=16,development_questions=32,
        history_scoring_max=464*16*2,history_representations=464,
        history_recipe='refit original utility regression with 7B descriptors; fixed epoch20; no evaluation-based selection',
        comparison='same frame pair coverage; full bank1120 and half bank560; no claim of equal token budget',
        audit_limit_tokens=16384,audit_thresholds={'max_abs':1e-3,'total_variation':1e-4,'generated_ids_equal':True},
        evaluation_protocol='two fixed option permutations, normalized single-letter logits; auxiliary generated audit only; no API',
        prune_minutes=360,history_minutes=240,gpus_per_job=1,cpus=4,memory_GiB=64,partition='g078t',comment='gx_llm',
        automatic_retry=False,downloads=False,API=False,base_updates=0)
    save(PREP/'plan.json',plan);print(json.dumps(plan,ensure_ascii=False))
if __name__=='__main__':main()

