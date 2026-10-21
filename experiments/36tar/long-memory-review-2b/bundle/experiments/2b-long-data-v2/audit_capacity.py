"""Reject a long-duration label as a substitute for actual context pressure."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/2b_memory_value_design_20261021_01'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    cfg=ROOT/'models/Qwen3.5-2B/config.json'
    layouts=ROOT/'runs/2b_long_stream_prepare_20261020_01/runtime_inputs_v2/expected_layout34.json'
    inventory=ROOT/'runs/2b_long_stream_prepare_20261020_01/local_long_video_inventory.json'
    limit=read(cfg)['text_config']['max_position_embeddings'];rows=read(layouts)
    assert limit==262144 and len(rows)==34
    records=[dict(id=r['id'],input_tokens=r['input_tokens'],maximum_position=r['maximum_position'],
        fraction_of_configured_context=r['input_tokens']/limit,
        exceeds_configured_context=r['input_tokens']+8>limit or r['maximum_position']>=limit,
        exceeds_previous_engineering_guard=r['input_tokens']+8>32768) for r in rows]
    assert not any(r['exceeds_configured_context'] for r in records)
    inv=read(inventory)['summary']
    report=dict(status='verified_existing_set_NOT_context_capacity_stress',questions=34,configured_context=limit,
        max_input_tokens=max(r['input_tokens'] for r in records),
        max_fraction=max(r['fraction_of_configured_context'] for r in records),
        over_configured_context=sum(r['exceeds_configured_context'] for r in records),
        available_inventory_max_video_seconds=inv['maximum_duration_s'],records=records,
        planned_4h_visual_only_lower_bound_tokens=4*3600//2*48,
        planned_4h_note='conditional on actual1FPS complete pairs at48 visual tokens; not an acquired asset or actual run',
        no_guard_lowering=True,no_new_model_load=True,no_GPU=True,
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (cfg,layouts,inventory)})
    with (OUT/'existing_capacity_audit.json').open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print({k:v for k,v in report.items() if k not in ('records','input_sha256')})


if __name__=='__main__':main()
