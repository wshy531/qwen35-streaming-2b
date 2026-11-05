"""Separate model-facing draft requests from scorer-only GT/annotations."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from prefix_request import make_request
from score_memory_eval import digest


def main():
    root=Path(__file__).resolve().parents[2];run=root/'runs/dataset_build_20261018_01'
    release=run/'content_release_v1'
    hashes=json.loads((release/'content_hashes.json').read_text(encoding='utf-8'))
    for name,expected in hashes.items():
        if hashlib.sha256((release/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('content snapshot changed: '+name)
    media={m['source_path']:m for m in json.loads((release/'media_inventory.json').read_text(encoding='utf-8'))['files']}
    traces={t['source_path']:t for t in json.loads((release/'actual_sampling_traces.json').read_text(encoding='utf-8'))['videos']}
    annotations=json.loads((release/'question_annotation_overlay.json').read_text(encoding='utf-8'))['questions']
    requests=[];keys=[]
    for item in annotations:
        row=item['original'];request=make_request(row,media[row['source_path']],traces[row['source_path']])
        requests.append(request)
        keys.append(dict(id=row['id'],answer_index=row['answer_index'],task=row['task'],split=row['split'],
            source_group_id=item['source_group_id'],original_row_sha256=item['original_row_sha256'],
            input_request_sha256=request['input_request_sha256'],provisional_role=item['provisional_role']))
    output=run/'draft_input_export';output.mkdir(exist_ok=False)
    for name,values in [('model_requests.jsonl',requests),('scorer_only_answer_key.jsonl',keys)]:
        (output/name).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values),encoding='utf-8')
    report=dict(status='cpu_draft_request_export_not_model_evaluation',requests=len(requests),
        per_split=dict(Counter(k['split'] for k in keys)),
        per_question_future_frame_count=sum(any(f['pts_s']>r['model_visible']['query_time_s'] for f in r['model_visible']['frames']) for r in requests),
        odd_snapshot_tail_requests=sum(r['model_visible']['snapshot_tail_pair'] is not None for r in requests),
        real_source_frames=sum(len(t['frames']) for t in traces.values()),
        model_visible_payload_sha256=digest([r['model_visible'] for r in requests]),
        model_visible_fields=['question','options','query_time_s','frames','completed_real_pairs','snapshot_tail_pair'],
        excluded_from_model=['answer_index','task','dataset','split','source_group_id','evidence_intervals',
            'evidence_gap_s','review','witnesses','provisional_role','source_timing_audit'],
        interface_adapter_integrated=False,true_2b_outputs=0,
        file_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
