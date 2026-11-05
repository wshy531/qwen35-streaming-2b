"""Independent structural and contact-sheet pixel audit; no model imports."""
import copy
import hashlib
from pathlib import Path

from PIL import Image

from prepare_holdout_review import ROOT,OUT,read,save,sha
from evaluation_gate import official_identity,freeze_issues
from prefix_request import make_request


def main():
    lock=read(OUT/'DATA_ROSTER_LOCK.json')
    for p,h in lock['file_sha256'].items():assert sha(OUT/p)==h,p
    for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h,p
    ledger=read(OUT/'all94_final_dispositions.json');original=read(OUT/'original94.json')
    admitted=read(OUT/'admitted.json');requests=read(OUT/'model_requests.json')
    assert len(ledger)==len(original)==94 and len({q['id'] for q in ledger})==94
    assert [official_identity(q) for q in ledger]==[official_identity(q) for q in original]
    accepted={q['id'] for q in ledger if q['admission']['include_in_roster']}
    assert accepted=={q['id'] for q in admitted}=={q['id'] for q in requests}
    assert len(accepted)==51 and len(ledger)-len(accepted)==43
    traces={t['source_path']:t for t in read(OUT/'sampling_1fps.json')}
    pixels=0;pages=0
    for receipt in read(OUT/'image_receipts.json'):
        for page,im in enumerate(receipt['images'],1):
            assert sha(ROOT/im['path'])==im['sha256']==lock['image_sha256'][im['path']]
            with Image.open(ROOT/im['path']) as image:
                assert image.size==(1024,890) and image.mode=='RGB'
                for w in receipt['witnesses']:
                    if w['page']!=page:continue
                    x=w['cell']%4*256;y=30+w['cell']//4*214
                    crop=image.crop((x,y,x+256,y+192))
                    assert hashlib.sha256(crop.tobytes()).hexdigest()==w['rgb_sha256']
                    pixels+=1
            pages+=1
    assert (pixels,pages)==(674,59)
    amap={q['id']:q for q in admitted}
    for r in requests:
        assert set(r)=={'id','media_path','media_sha256','input_protocol','model_visible','input_request_sha256'}
        v=r['model_visible'];assert set(v)=={'question','options','query_time_s','frames','completed_real_pairs','snapshot_tail_pair'}
        q=amap[r['id']];assert q['review']['model_outputs_seen'] is False and not q['review']['flags_pending']
        rebuilt=make_request(q,dict(source_path=q['source_path'],media_path=q['media_path'],sha256=q['media_sha256']),traces[q['source_path']])
        assert rebuilt['model_visible']==v and rebuilt['input_request_sha256']==r['input_request_sha256']
        altered=copy.deepcopy(q)
        altered.update(answer_index=(q['answer_index']+1)%len(q['options']),evidence_intervals=[[9999,10000]],task='changed')
        altered['review']={'note':'THIS MUST NEVER ENTER THE MODEL'}
        blind=make_request(altered,dict(source_path=q['source_path'],media_path=q['media_path'],sha256=q['media_sha256']),traces[q['source_path']])
        assert blind['model_visible']==v and blind['input_request_sha256']==r['input_request_sha256']
        assert all(f['pts_s']<=q['query_time_s'] for f in v['frames'])
        assert len(v['completed_real_pairs'])*2+bool(v['snapshot_tail_pair'])==len(v['frames'])
        if v['snapshot_tail_pair']:
            tail=v['snapshot_tail_pair'];assert tail['synthetic_second_frame'] and tail['commit_to_main_stream'] is False
            assert tail['frame_ids']==[v['frames'][-1]['frame_id']]*2
        assert all(any(w['frame_id']==f['frame_id'] and w['rgb_sha256']==f['rgb_sha256'] for f in v['frames']) for w in q['review']['witnesses'])
    proof=read(OUT/'cpu_input_preflight.json');assert proof['model_instances']==proof['pretrained_loads']==0
    assert proof['data_lock_sha256']==sha(OUT/'DATA_ROSTER_LOCK.json')
    for p,h in proof['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert len(proof['details'])==51 and {r['id'] for r in proof['details']}==accepted
    assert proof['max_suffix_tokens']==79 and proof['max_full_input_tokens']==5711
    assert set(proof['exceeds_prior_native_matched_3072_guard'])=={'ovo:546','ovo:564','ovo:558'}
    bundle=read(OUT/'execution_bundle_draft.json');assert freeze_issues(bundle)==read(OUT/'summary.json')['execution_admission_issues']
    assert 'bundle_not_frozen' in freeze_issues(bundle)
    summary=read(OUT/'summary.json')
    assert summary['natural_core']['questions']==47 and summary['instructional_supplement']['questions']==4
    assert summary['primary_FT']['questions']==33
    result=dict(status='all94_data_and_pixels_audit_passed_execution_pending',all_questions=94,qualified=51,not_admitted=43,
        natural_core=47,primary_natural_FT=33,contextual_natural_C=14,instructional_supplement=4,
        exact_sheet_crops_verified=pixels,exact_sheet_pages_verified=pages,answer_blind_metamorphic_requests_verified=len(requests),
        causal_and_tail_checks=len(requests),original_fields_and_splits_unchanged=True,models_loaded=0,
        cuda_used=False,cluster_operations=0,formal_model_bundle_frozen=False,
        data_lock_sha256=sha(OUT/'DATA_ROSTER_LOCK.json'),preflight_sha256=sha(OUT/'cpu_input_preflight.json'),
        auditor_sha256=sha(Path(__file__)),remaining='Independent human review absent; model-specific native/position preflight and GPU budget not yet completed.2B remains unvalidated.')
    save(OUT/'delivery_receipt.json',result);print(result)


if __name__=='__main__':main()
