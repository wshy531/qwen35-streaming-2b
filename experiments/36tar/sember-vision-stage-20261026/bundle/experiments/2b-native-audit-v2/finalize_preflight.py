"""Lock the ready CPU checks and source package identities, without granting GPU permission."""
import ast
import hashlib
from pathlib import Path
import numpy as np
from audit_common import ROOT,OUT,read,save


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def consistent_save(path,value):
    if path.exists():assert read(path)==value
    else:save(path,value)


def main():
    stages=('official530','official540','stream')
    rows={s:read(OUT/f'CPU_matched_{s}.json') for s in stages}
    assert len({r['state_dict_sha256'] for r in rows.values()})==1
    assert all(rows[s]['layouts']==rows['stream']['layouts'] for s in stages)
    assert len({tuple(r['generated_ids']) for r in rows.values()})==1
    reference=np.load(OUT/'CPU_matched_stream.npy')
    errors={s:float(np.abs(np.load(OUT/f'CPU_matched_{s}.npy')-reference).max()) for s in stages}
    assert max(errors.values())<1e-6
    assert read(OUT/'CPU_normal_real6s_attempt02.json')['status']=='passed'
    assert all(r['status']=='passed' for r in read(OUT/'CPU_official_540_probe.json')['results'])
    consistent_save(OUT/'CPU_comparison.json',dict(status='passed',same_weight_SHA=True,all24_layouts_identical=True,
        max_logit_errors=errors,generated_IDs_identical=True,official540_default_real6s=True,
        real_weights=0,GPU_jobs=0,no_long_video_capability_claim=True))
    design=read(OUT/'inputs/design.json')
    design.update(attention='official sdpa forced MATH; matches current C numerical backend',
        max_vision_encodings=75,resource_budget=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,RAM_GiB=24,minutes=60),
        log_limit_bytes=128*2**20,wait_for_218537_to_finish=True)
    # Preparation metadata is finalized once, before the execution lock exists.
    (OUT/'inputs/design.json').write_text(__import__('json').dumps(design,ensure_ascii=False,indent=2),encoding='utf-8')
    dependencies={}
    for path in Path(__file__).parent.iterdir():
        if path.suffix not in ('.py','.sh','.md'):continue
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
        dependencies[path.relative_to(ROOT).as_posix()]=digest(path)
    for name in ('sember_source.py','sember_protocol.py'):
        path=ROOT/'experiments/2b-sember-v1'/name
        dependencies[path.relative_to(ROOT).as_posix()]=digest(path)
    for name in ('config.json','tokenizer_config.json','video_preprocessor_config.json','preprocessor_config.json','chat_template.jinja'):
        path=ROOT/'models/Qwen3.5-2B'/name
        if path.exists():dependencies[path.relative_to(ROOT).as_posix()]=digest(path)
    lock=dict(runtime_sha256=dependencies,
        input_sha256={p.name:digest(p) for p in (OUT/'inputs').iterdir() if p.is_file()},
        official_package=read(OUT/'official_environment.json'),
        original_official_source_SHA=rows['official530']['identity']['model_source_sha256'],
        new_official_source_SHA=rows['official540']['identity']['model_source_sha256'],
        explicit_new_GPU_authorization_required=True)
    save(OUT/'inputs/execution_lock.json',lock)
    originals=['runs/sember_main_prepare_20261025_01/GPU_results_218526/reference/answers.jsonl',
        'runs/sember_prepare_20261025_01/selection/labels_LOCAL_ONLY.json']
    consistent_save(OUT/'local_analysis_identity.json',dict(frozen_inputs={p:digest(ROOT/p) for p in originals},
        labels_and_existing_answers_not_uploaded=True))
    print(dict(status='frozen',runtime_files=len(dependencies),execution_lock_sha256=digest(OUT/'inputs/execution_lock.json')),flush=True)


if __name__=='__main__':main()
