"""Freeze recipe, source separation, content reference and executable dependency closure."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
OUT=ROOT/'runs/7b_content_residual_main_prepare_20261113_01'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def main():
    OUT.mkdir(exist_ok=True)
    panel=read('runs/training_order_main_prepare_20261104_02/panel_LOCAL_ONLY.json')
    qs=read('runs/7b_prune_head_prepare_20261111_01/questions.json')
    order=panel['schedules']['sequential'];train=[panel['groups'][v] for v in order]
    assert len(set(order))==270 and sum(len(g['questions']) for g in train)==2351 and len(qs)==210
    protected=[panel['groups'][v] for v in panel['development']]
    protected=[(str(g['video_id']),g['sha256']) for g in protected]+[(str(q['source_id']),q['media_sha256']) for q in qs]
    for path in ('runs/head_data_coverage_20261112_01/STAR_reserve_inputs.json','runs/head_data_coverage_20261112_01/MLVU_eval_inputs.json'):
        protected.extend((str(q['source_id']),q['media_sha256']) for q in read(path))
    assert not set(map(str,order))&{i for i,h in protected}
    assert not {g['sha256'] for g in train}&{h for i,h in protected}
    refs=[];wanted={(q['id'],r,s) for q in qs for r in ('recent8','LS') for s in (0,2)}
    for line in (ROOT/'runs/7b-prune-head-223298/answers.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['mode']=='content' and (r['id'],r['route'],r['shift']) in wanted:
            refs.append({k:r[k] for k in ('id','suite','source_id','route','shift','selected_original','visual_tokens','option_logits')})
    assert len(refs)==840 and {(r['id'],r['route'],r['shift']) for r in refs}==wanted
    media={g['path']:g['sha256'] for g in train}
    for q in qs:
        assert q['media_path'] not in media or media[q['media_path']]==q['media_sha256']
        media[q['media_path']]=q['media_sha256']
    plan=dict(hypothesis='Exact content/coverage start plus learned residual improves fixed final readouts',
        train_sources=270,train_questions=2351,proposals=270,scorer_parameters=87105,
        training_order='short150/1237 then longer120/1114; unchanged',
        objective='gold-letter full-vocabulary CE; frozen base; same-source hard loss acceptance with Adam rollback',
        evaluation_questions=210,evaluation_suites={'StreamingBench':150,'short':60},evaluation_classifications=1680,
        evaluation_routes=['recent8','LS'],option_shifts=[0,2],initial_content_regression=840,
        training_vision_bound=14517,evaluation_vision_bound=53930,training_language=7053,
        model_loads=2,generated_tokens=0,train_minutes=180,evaluation_minutes=180,total_GPU_hours_limit=6,
        GPU_per_job=1,CPU_per_job=4,memory_GiB_per_job=64,partition='g078t',comment='gx_llm',
        dependency='evaluation after training succeeds; failure stops chain; no retry/requeue',
        checkpoint_rule='fixed initial/final; intermediate checkpoints for traceability only',
        stop_rule='one pass; no benefit versus content start means close this recipe without sweep',
        limitations=['open regression, not unseen generalization','long_train is longer NExT-QA, not proven history dependency',
                    'source ID and byte SHA separation cannot guarantee distinct filming sessions',
                    'initialization and mandatory spatial coverage are a bundled recipe, not isolated causal factors'],
        source_isolation=True,media_files=len(media))
    for name,data in [('plan.json',plan),('content_references.json',refs),('media.json',[dict(path=p,sha256=h) for p,h in sorted(media.items())])]:
        (OUT/name).write_text(json.dumps(data,indent=2)+'\n')
    files=set(read('runs/7b_trained_prune_evaluation_prepare_20261113_01/lock.json'))
    files.update(read('runs/7b_content_residual_prepare_20261113_01/lock.json'))
    files.update(p.relative_to(ROOT).as_posix() for p in HERE.iterdir() if p.suffix in ('.py','.sh'))
    files.update((OUT/n).relative_to(ROOT).as_posix() for n in ('plan.json','content_references.json','media.json'))
    files.update(['runs/7b_content_residual_prepare_20261113_01/lock.json','runs/7b_trained_prune_completion_prepare_20261113_01/lock.json'])
    for p in OUT.glob('imports-*.json'):files.update(json.loads(p.read_text()))
    lock={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(files)}
    (OUT/'lock.json').write_text(json.dumps(lock,indent=2)+'\n')
    bundle=OUT/'prepare.tar'
    with tarfile.open(bundle,'w') as t:
        for p in sorted(HERE.iterdir()):
            if p.suffix in ('.py','.sh'):t.add(p,arcname=p.relative_to(ROOT).as_posix())
        for p in sorted(OUT.glob('*.json')):t.add(p,arcname=p.relative_to(ROOT).as_posix())
    print(json.dumps(dict(plan=plan,execution_files=len(lock),bundle_sha=hashlib.sha256(bundle.read_bytes()).hexdigest())))
if __name__=='__main__':main()
