"""Reuse all fixed R2 questions, not results, to define the next quality comparison."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/content_quality_prepare_20261101_01'
def main():
    source=ROOT/'runs/midprune_quality_prepare_20261026_01/inputs'
    requests=json.loads((source/'requests.json').read_text(encoding='utf-8'))
    videos=json.loads((source/'videos.json').read_text(encoding='utf-8'))
    assert len(requests)==478 and len(videos)==128
    OUT.mkdir(exist_ok=False)
    for name in ('requests.json','videos.json','expected_prefixes.json'):
        (OUT/name).write_bytes((source/name).read_bytes())
    protocol=dict(status='quality_inputs_prepared_not_GPU_run',questions=478,videos=128,
        arms=['full','post_fixed','mid_fixed','post_content','mid_content'],
        formal_answers=2390,max_new_tokens=16,warmup_answers=5,maximum_generated_tokens=38320,
        original_option_order=True,no_new_questions=True,frame_inputs='unchanged original R2 expected_prefixes',
        same_recent_pair_coverage=8,full_tokens_per_pair=108,pruned_tokens_per_pair=54,
        no_model_results_used_for_sampling=True,candidate_informed_by_old_R2_development=True,
        old_478_is_opened_development_not_fresh_test=True,
        first_gate='new component timing must retain useful savings before GPU quality run',
        future_new_source_confirmation='StreamingBench frozen protocol, original timing issues retained',
        analysis=['all content correct','correct-to-wrong','wrong-to-correct','task type','source grouping','format separate'],
        precision='FP32',native_pure_text_reference='not included in this compression ablation',
        resource_budget_not_yet_frozen=True,GPU_authorized=False,
        input_SHA256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('requests.json','videos.json','expected_prefixes.json')})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('Prepared all 128 sources / 478 questions / 5 modes, no answers or labels read')
if __name__=='__main__':main()
