"""Quality and isolated-process costs; no conversion into edge-device claims."""
import argparse,statistics
from pathlib import Path
from collections import Counter
from common import OUT,ARMS,read,save
from analyze import analyze


def main(result):
    complete=read(result/'completed.json');assert complete['status']=='complete'
    analyze(result,OUT/'answer_results.json')
    quality=read(OUT/'answer_results.json')
    replay=quality['historical_replay']
    assert replay['expected']==909
    # Any mismatch remains visible; costs cannot be accepted as the old workload silently.
    equivalent=not any(replay[k] for k in ('generated_ID_mismatches','input_ID_mismatches','node_mismatches'))
    records=complete['results'];costs={}
    for arm in ARMS:
        runs=[r for r in records if r['completed']['arm']==arm]
        assert len(runs)==3
        profiles=[p for r in runs for p in r['completed']['profiles']]
        assert len(profiles)==len({p['group'] for p in profiles})==112
        qa=[q for p in profiles for q in p['QA']];assert len(qa)==303
        costs[arm]=dict(
            block_wall_seconds=sum(r['completed']['measured_block_wall_seconds'] for r in runs),
            source_pipeline_wall_seconds=sum(p['pipeline_wall_seconds'] for p in profiles),
            model_load_seconds=sum(r['startup']['model_load_seconds'] for r in runs),
            warmup_seconds=sum(r['startup']['warmup_seconds'] for r in runs),
            source_components={key:sum(p['source'].get(key,0.) for p in profiles) for key in
                ('decode_seconds','transform_seconds','media_integrity_sha_seconds','open_seconds')},
            encoder_components={key:sum(p['encoding'][key] for p in profiles) for key in
                ('preprocess_seconds','vision_seconds','selection_banks_seconds')},
            nested_bank_components={key:sum(p[key] for p in profiles) for key in
                ('transfer_validate_seconds','selection_seconds','bank_update_seconds')},
            audit_seconds=sum(p['audit_seconds'] for p in profiles),
            nested_log_write_seconds=sum(p['log_write_seconds'] for p in profiles),
            QA_components={key:sum(q[key] for q in qa) for key in
                ('context_build_seconds','prefill_seconds','decode_seconds','QA_total_seconds')},
            mean_input_tokens=statistics.mean(q['input_tokens'] for q in qa),
            max_QA_cache_bytes=max(q['QA_cache_bytes'] for q in qa),
            max_persistent_bank_selector_tensor_bytes=max(p['bank']['standalone_persistent_tensor_bytes'] for p in profiles),
            max_measured_GPU_allocated_bytes=max(r['completed']['measured_peak_GPU_allocated_bytes'] for r in runs),
            max_measured_GPU_reserved_bytes=max(r['completed']['measured_peak_GPU_reserved_bytes'] for r in runs),
            max_process_lifetime_RSS_bytes=max(r['completed']['process_lifetime_peak_RSS_bytes'] for r in runs),
            RSS_includes_load_and_warmup=True,
            baseline_GPU_allocated_range=[min(r['completed']['baseline']['GPU_allocated_bytes'] for r in runs),max(r['completed']['baseline']['GPU_allocated_bytes'] for r in runs)],
            actual_decoded_frames=sum(p['source']['decoded_frames'] for p in profiles),
            actual_encoded_pairs=sum(p['vision_encodings'] for p in profiles))
    assert {v['actual_encoded_pairs'] for v in costs.values()}=={3981}
    assert len({v['actual_decoded_frames'] for v in costs.values()})==1
    save(OUT/'standalone_cost_results.json',dict(actual_device='A800 FP32',quality_replay_equivalent=equivalent,costs=costs,
        complete_budget=dict(measured_answers=909,warmup_answers=9,measured_encodings=11943,warmup_encodings=72,pretrained_loads=9),
        limitations=['Causal file replay, not paced1x or edge hardware.',
          'One workload pass per method; blocks rotate order, not full repeated independent trials.',
          'Wall times include integrity, audits, logging and cleanup; source pipeline and block wall are distinct.',
          'Bank timing and log writing are nested in other metrics; never add all columns together.',
          'RSS is process lifetime high-water including load/warmup; CUDA measurement peak reset after warmup.',
          'Large hashing audits are timed separately, but returned-ID/node digest work is also in wall time.',
          'Long source open cost remains in wall time, not separately resolved.']))
    print(dict(quality_equivalent=equivalent,arms=3,groups_each=112))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,required=True)
    main(parser.parse_args().results)
