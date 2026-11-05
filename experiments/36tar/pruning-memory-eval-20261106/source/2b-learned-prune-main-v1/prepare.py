import json,hashlib
from pathlib import Path
from protocol import ROOT,PREP,DATA,read,ARMS,train_shift

def main():
    PREP.mkdir(exist_ok=True)
    sources=read(DATA/'admitted_sources.json');questions=read(DATA/'admitted_questions.json')
    files={r['video_id']:r for r in read(DATA/'acquisition_plan.json')['files']}
    sources=[dict(s,mirror_lfs_sha256=files[s['video_id']]['mirror_lfs_sha256']) for s in sources]
    train=[dict(q,shift=train_shift(q)) for q in questions if q['role']=='train']
    train.sort(key=lambda q:hashlib.sha256(('P1-one-epoch-order:'+q['id']).encode()).hexdigest())
    evaluation=[q for q in questions if q['role'] in ['development','confirmation']]
    assert len(train)==1237 and len(evaluation)==652
    assert not {q['video_id'] for q in train}&{q['video_id'] for q in evaluation}
    pair_counts={s['video_id']:s['encoding_pairs'] for s in sources}
    spec=dict(seed=20261101,epochs=1,batch_size=1,optimizer='AdamW',lr=1e-4,weight_decay=0.,clip_gradient_norm=1.,
        objective='official_answer_letter_CE_only',auxiliary_loss=None,training_questions=1237,
        training_vision_calls=sum(pair_counts[q['video_id']] for q in train),
        eval_questions=652,eval_arms=list(ARMS),eval_rotations=[0,2],eval_answers=652*6*2,
        eval_vision_calls=sum(pair_counts[s['video_id']] for s in sources if s['role']!='train')*6,
        max_new_tokens=8,max_generated_tokens=652*6*2*8,model_loads=1,base_updates=0,
        checkpoint_policy='final_one_epoch_no_selection',trainable_parameters=69697,
        no_model_result_based_exclusions=True,deadline_minutes=118,
        quality_only_shared_decode_not_independent_cost=True,
        note='No full teacher consistency loss in first round; no source/video label is synthesized by student.')
    for n,x in [('sources.json',sources),('train.json',train),('eval.json',evaluation),('protocol.json',spec)]:
        (PREP/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(spec,indent=2))

if __name__=='__main__':main()
