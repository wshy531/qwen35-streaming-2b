"""Verify and freeze data plus the reused0.8B interface identity, no inference."""
import ast
import copy
import csv
import json
from pathlib import Path
import subprocess

from evaluation_gate import official_identity,require_frozen,freeze_issues
from prepare_natural_v2 import ROOT,OUT,BASE,DATA,read,save,sha
from score_memory_eval import digest


def main():
    raw=read(OUT/'original149.json');rows=read(OUT/'admitted74.json');requests=read(OUT/'model_requests74.json')
    raw235=read(DATA/'candidate_manifest.json');old={r['id']:r for r in raw235}
    extended=raw235+read(OUT/'additional78.json')
    assert len(extended)==313 and len({q['id'] for q in extended})==313
    assert extended[:235]==raw235
    save(OUT/'candidate_manifest_v2_313.json',extended)
    csvraw={f'nextgqa:val:{r["video_id"]}:{r["qid"]}':r for r in csv.DictReader((DATA/'upstream/nextgqa/val.csv').open(encoding='utf-8'))}
    for q in raw:
        upstream=csvraw[q['id']]
        assert q['question']==upstream['question'] and q['task']==upstream['type']
        assert q['options']==[upstream[f'a{i}'] for i in range(5)] and q['options'][q['answer_index']]==upstream['answer']
        if q['id'] in old:assert official_identity(q)==official_identity(old[q['id']])
    assert len(rows)==74 and len({r['id'] for r in rows})==74
    assert {r['id'] for r in rows}=={r['id'] for r in requests}
    assert sha(DATA/'candidate_manifest.json')=='cd1d888eb0a0c00f300649f41800e60fd02b002e91f41f154b42e041fcc28161'
    summary=read(OUT/'summary.json');proof=read(OUT/'cpu_input_preflight.json')
    assert proof['status']=='passed_actual_08b_tokenizer_and_template_no_model' and proof['pretrained_loads']==0
    assert all(sha(ROOT/p)==h for p,h in proof['source_sha256'].items())
    prior=ROOT/'runs/08b_memory_compare_20261019_01'
    prior_audit=read(ROOT/'runs/08b_stage2_analysis_20261019_01/delivery_receipt.json')
    assert prior_audit['status']=='stage2_consistency_audit_passed'
    for p,h in prior_audit['source_sha256'].items():assert sha(ROOT/p)==h
    assert read(prior/'completed.json')['records']==72 and read(prior/'completed.json')['model_loads']==1
    loaded=read(prior/'loaded.json');assert sha(Path(loaded['source']))==loaded['source_sha256']
    basis=dict(status='reused_real_08b_interface_plus_new_CPU_input_check',
        earlier_real_interface_sampling_fps=.5,earlier_real_interface_questions=24,
        earlier_real_interface_scope='FP32 same pair encoder, memory policies, readout and model source; source hashes unchanged',
        earlier_completed_sha256=sha(prior/'completed.json'),earlier_loaded_sha256=sha(prior/'loaded.json'),
        earlier_source_audit_sha256=sha(ROOT/'runs/08b_stage2_analysis_20261019_01/delivery_receipt.json'),
        new_cpu_input_preflight_sha256=sha(OUT/'cpu_input_preflight.json'),new_1fps_real_weight_outputs=0,
        new_input_max_pairs=proof['max_pairs'],new_input_max_visual_tokens=proof['max_full_visual_tokens'],
        limitation='This binds an already exercised interface to new causal requests; it is not a new1FPS numerical equivalence or accuracy result.')
    save(OUT/'08b_interface_basis.json',basis)
    bundle=copy.deepcopy(read(OUT/'execution_bundle_draft.json'))
    bundle['status']='frozen_for_evaluation'
    bundle['input_protocol'].update(model_revision=proof['processor_revision'],processor_revision=proof['processor_revision'],
        real_interface_receipt_sha256=sha(OUT/'08b_interface_basis.json'))
    assert bundle['input_protocol']['prompt_template_sha256']==proof['prompt_template_sha256']
    for q in bundle['rows']:q['formal_evaluation_eligible']=True
    require_frozen(bundle)
    save(OUT/'frozen_bundle_08b.json',bundle)
    repo=ROOT/'publish/qwen35-streaming'
    assert subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()==''
    source_files=[Path(__file__).with_name(n) for n in ['prepare_natural_v2.py','prepare_additional78_sheets.py',
        'review_additional78.py','assemble_natural_v2.py','prepare_model_canvas_review.py','lock_natural_v2.py',
        'preflight_natural_v2.py','freeze_natural_v2.py']]
    for p in source_files:ast.parse(p.read_text(encoding='utf-8'))
    summary.update(status='curated_data_and_08b_protocol_frozen_no_new_inference',formal_model_bundle_frozen=True,
        frozen_model_scope='0.8B only; 2B still needs its own true interface receipt and preflight',
        execution_admission_issues=[],evaluation_sha256=digest(bundle),frozen_bundle_sha256=sha(OUT/'frozen_bundle_08b.json'))
    save(OUT/'summary.json',summary)
    files=['candidate_manifest_v2_313.json','original149.json','additional78.json','additional78_review.json','sampling_1fps.json',
        'all149_final_dispositions.json','admitted74.json','model_requests74.json','scorer_only_key74.json','strata74.json',
        'analysis_rules.json','cpu_input_preflight.json','08b_interface_basis.json','frozen_bundle_08b.json','summary.json',
        '主比较74题清单.csv','全部149题处置.csv']
    save(OUT/'DATA_LOCK.json',dict(status='locked',scope='curated natural comparison v2; no inference performed',
        questions=74,canonical_video_sources=21,sampling_fps=1.,canvas_hw=[192,256],
        evaluation_sha256=digest(bundle),model_visible_trace_sha256=bundle['input_protocol']['input_trace_sha256'],
        file_sha256={name:sha(OUT/name) for name in files},source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source_files},
        source_registry_sha256=sha(BASE/'content_release_v1/source_registry.json'),
        images_sha256={im['path']:sha(ROOT/im['path']) for path in ['additional78_sheets/receipts.json','model_canvas_sheets/receipts.json']
            for s in read(OUT/path)['sources'] for im in s['images']},
        original235_unchanged=True,holdout_viewed=False,pretrained_models_loaded_this_phase=0,
        real1fps_outputs=0,old_repository_clean=True,cluster_operations=0,
        execution_note='Passing data/interface admission does not constitute a model result or a new cluster GPU authorization.'))
    print(dict(status='locked_and_gate_passed_for_08b',questions=74,canonical_videos=21,
        evaluation_sha256=digest(bundle),new_model_loads=0,true1fps_answers=0))


if __name__=='__main__':main()
