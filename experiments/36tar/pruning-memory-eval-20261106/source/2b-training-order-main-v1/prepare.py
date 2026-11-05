"""Freeze official EOF supervision; no answer-dependent admission or early labels."""
import collections, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/training_order_main_prepare_20261104_01'
OLD=ROOT/'runs/learned_prune_main_prepare_20261101_01'
NEW=ROOT/'runs/delayed_training_candidates_20261104_01'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rank(x): return hashlib.sha256(('order-main-v1:'+x).encode()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True)
    oldsrc={s['video_id']:s for s in read(OLD/'sources.json')}
    newsrc={s['video_id']:s for s in read(NEW/'acquisition_plan.json')['files']}
    oldaudit={r['video_id']:r for r in read(ROOT/'runs/learned_prune_data_prepare_20261101_01/CPU_media_audit.json')['rows']}
    newaudit={r['video_id']:r for r in map(json.loads,(NEW/'media_audit.jsonl').read_text().splitlines())}
    trainold=read(OLD/'train.json'); devold=[q for q in read(OLD/'eval.json') if q['role']=='development']
    assert len(trainold)==1237 and len(devold)==345
    groups={}; excluded=[]
    for pool,qs,srcs,audits,folder in [
        ('short_train',trainold,oldsrc,oldaudit,'learned-prune-nextqa-v1'),
        ('old_development',devold,oldsrc,oldaudit,'learned-prune-nextqa-v1'),
        ('new',read(NEW/'questions_LOCAL_ONLY.json'),newsrc,newaudit,'delayed-training-nextqa-v1')]:
        for q in qs:
            a=q['answer_index']; opts=q['options']
            if not isinstance(a,int) or not 0<=a<5 or len(opts)!=5 or not all(isinstance(o,str) and o.strip() for o in opts) or sum(o.strip().lower()==opts[a].strip().lower() for o in opts)!=1:
                excluded.append(dict(id=q['id'],pool=pool,reason='invalid or duplicated correct option'));continue
            vid=q['video_id']; s=srcs[vid]; audit=audits[vid]
            role=pool if pool!='new' else ('long_train' if s['role']=='train_candidate' else 'long_development')
            if vid not in groups:
                samples=audit['samples']; n=(len(samples)+1)//2
                groups[vid]=dict(video_id=vid,role=role,path='datasets/'+folder+'/'+s['mirror_path'],sha256=s['mirror_lfs_sha256'],
                    audit=dict(samples=samples,encoding_pairs=n),duration_s=audit['duration_s'],questions=[])
            assert groups[vid]['role']==role
            groups[vid]['questions'].append(dict(q,type=q.get('type',q.get('official_type')),query_policy='official_full_clip_EOF'))
    ordered={role:sorted([g for g in groups.values() if g['role']==role],key=lambda g:rank(g['video_id']))
             for role in ('short_train','long_train','old_development','long_development')}
    train=ordered['short_train']+ordered['long_train']; dev=ordered['old_development']+ordered['long_development']
    assert set(g['video_id'] for g in train).isdisjoint(g['video_id'] for g in dev)
    assert set(g['sha256'] for g in train).isdisjoint(g['sha256'] for g in dev)
    assert len(set(g['sha256'] for g in groups.values()))==len(groups)
    seq=[g['video_id'] for g in train]
    # Each source is an identical batch in both arms. Proportional interleaving,
    # preserving within-domain order; no repeated examples or padding.
    merged=sorted([( (i+.5)/len(ordered[r]),r,i,g['video_id']) for r in ('short_train','long_train') for i,g in enumerate(ordered[r])])
    mix=[x[-1] for x in merged]
    assert collections.Counter(seq)==collections.Counter(mix)
    summary={r:dict(videos=len(gs),questions=sum(len(g['questions']) for g in gs),pairs=sum(g['audit']['encoding_pairs'] for g in gs),
                   duration_min=min(g['duration_s'] for g in gs),duration_max=max(g['duration_s'] for g in gs),
                   types=dict(collections.Counter(q['type'][0] for g in gs for q in g['questions']))) for r,gs in ordered.items()}
    protocol=dict(seed=20261104,lr=1e-4,clip=1.,epochs=1,loss='mean official answer CE per source batch',
        base_frozen=True,scorer_parameters=69697,short_read='all original short clip pairs',long_read='LS864 recent432 fixed event every3 pairs',
        training_questions=sum(len(g['questions']) for g in train),updates_per_arm=len(train),
        evaluation_questions=sum(len(g['questions']) for g in dev),evaluation_shifts=[0,2],
        no_early_cutoff_labels=True,no_model_based_filter=True,seed_count=1,
        claim_limit='curriculum of short EOF vs bounded longer EOF, not paired same-question memory necessity',summary=summary)
    save(OUT/'panel_LOCAL_ONLY.json',dict(groups=groups,train=seq,development=[g['video_id'] for g in dev],schedules=dict(sequential=seq,interleaved=mix)))
    save(OUT/'protocol.json',protocol);save(OUT/'exclusions.json',excluded)
    print(json.dumps(dict(protocol=protocol,exclusions=excluded),ensure_ascii=False))
if __name__=='__main__':main()
