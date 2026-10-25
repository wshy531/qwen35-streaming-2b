"""Freeze the already selected eight development queries and actual input prefixes."""
from sember_common import OUT,PREP,ARMS,read,save,sha


def main():
    inputs=OUT/'inputs';inputs.mkdir(exist_ok=False)
    queries=read(PREP/'profiler_requests.json');all_input=read(OUT/'input_preflight_all.json')
    assert all_input['status']=='passed'
    expected={q['question_id']:q for record in all_input['records'] for q in record['queries']}
    groups={}
    for q in queries:
        assert q['split']=='development18'
        groups.setdefault(q['video_id'],[]).append(q)
    ids=set(groups);videos=[v for v in read(PREP/'selection/videos.json') if v['video_id'] in ids]
    prefixes={qid:expected[qid] for qid in [q['question_id'] for q in queries]}
    pairs=sum(max(prefixes[q['question_id']]['pairs'] for q in qs) for qs in groups.values())
    save(inputs/'requests.json',queries);save(inputs/'videos.json',videos);save(inputs/'expected_prefixes.json',prefixes)
    save(inputs/'expected_RGB_prefixes.json',all_input['RGB_identity_prefixes'])
    save(inputs/'profile_protocol.json',dict(stage='development_interface_and_cost_profile',formal_test=False,
        queries=8,source_videos=len(videos),bounded_arms=list(ARMS),separate_reference='R_all_history_stream',
        bounded_answers=88,reference_answers=8,warmup_answers=12,answers_upper=108,generation_tokens_upper=1728,
        pairs_per_process=pairs,total_vision_encodings_including_warmup=2*pairs+2,
        pretrained_loads=2,canvas=[288,384],sampling_FPS=.5,greedy_max_new_tokens=16,
        GPU_budget=dict(partition='g078t',comment='gx_llm',GPUs=1,CPUs=4,system_RAM_GiB=24,minutes=30),
        logs_limit_bytes=64*2**20,automatic_retries=False,GPU_authorized=False,
        labels_in_worker=False,official_selection_lock_SHA=sha(PREP/'selection/selection_lock.json'),
        standalone_cost_claim=False,not_wallclock_1x=True))
    print(dict(status='profile_inputs_frozen',queries=8,sources=len(videos),pairs_per_process=pairs,
        vision_calls_including_warmup=2*pairs+2,answers_upper=108,GPU_authorized=False),flush=True)


if __name__=='__main__':main()
