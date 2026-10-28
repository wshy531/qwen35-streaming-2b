"""Freeze all original formal questions, no selection based on development scores."""
from collections import Counter
from main_common import ROOT,OUT,PREP,ARMS,read,save,sha


def main():
    inputs=OUT/'inputs';inputs.mkdir(exist_ok=False)
    questions=[q for q in read(PREP/'selection/requests.json') if q['split']=='formal']
    videos=[v for v in read(PREP/'selection/videos.json') if v['split']=='formal']
    lock=read(PREP/'selection/selection_lock.json')
    assert len(questions)==478 and len(videos)==128
    assert set(q['question_id'] for q in questions)==set(lock['formal_question_ids'])
    forbidden={'correct_letter','correct_index','ground_truths','answer_start_time','answer_end_time','official_memory_recency','end_gap_proxy_s'}
    assert all(not forbidden.intersection(q) for q in questions)
    prepared=read(ROOT/'runs/sember_adapter_prepare_20261025_01/input_preflight_all.json')
    records=[r for r in prepared['records'] if r['split']=='formal']
    expected={q['question_id']:q for r in records for q in r['queries']}
    assert set(expected)=={q['question_id'] for q in questions}
    pairs=sum(len(r['pair_records']) for r in records);assert pairs==14678
    save(inputs/'requests.json',questions);save(inputs/'videos.json',videos);save(inputs/'expected_prefixes.json',expected)
    first_pairs={r['video_id']:r['pair_records'][0] for r in records}
    save(inputs/'first_pairs.json',first_pairs)
    save(inputs/'protocol.json',dict(stage='fixed_formal_128source_478question_comparison',questions=478,sources=128,
        bounded_arms=list(ARMS),reference_arm='R_all_history_stream',bounded_answers=5258,reference_answers=478,
        measured_answers=5736,warmup_answers=12,answers_max=5748,generation_tokens_max=91968,
        pairs_per_process=pairs,visual_encodings_including_warmup=29358,pretrained_loads=2,
        dtype='FP32',canvas=[288,384],sampling_FPS=.5,max_new_tokens=16,prefill_chunk=512,
        warmup=dict(video_id=questions[0]['video_id'],query_time=first_pairs[questions[0]['video_id']]['second']['pts_s']),
        script='experiments/2b-sember-main-v1/run_main.sh',
        resource_budget=dict(GPUs=1,CPUs=4,system_RAM_GiB=24,partition='g078t',comment='gx_llm',minutes=300),
        log_limit_bytes=128*2**20,worker_RSS_limit_bytes=20*2**30,GPU_allocated_limit_bytes=48*2**30,
        stop_before_allocation_end_seconds=180,automatic_retry=False,training=False,download=False,
        GPU_authorized=False,authorization_basis='specific script/resource confirmation recorded separately',
        frozen_original_selection_SHA=sha(PREP/'selection/selection_lock.json'),
        original_development_execution_SHA=sha(ROOT/'runs/sember_adapter_prepare_20261025_01/inputs/execution_lock.json'),
        category_counts=dict(Counter(q['question_category'] for q in questions)),
        labels_available_to_worker=False,no_score_based_filtering=True,
        primary_metric='official parsed-choice content correct / all 478; format separate',
        all_low_scores_and_failures_preserved=True,not_wallclock_1x=True,not_independent_method_cost=True))
    print(dict(status='formal_inputs_frozen',sources=128,questions=478,answers_with_warmup=5748,
        pairs_per_process=pairs,visual_calls=29358,pretrained_loads=2,GPU_authorized=False),flush=True)


if __name__=='__main__':main()
