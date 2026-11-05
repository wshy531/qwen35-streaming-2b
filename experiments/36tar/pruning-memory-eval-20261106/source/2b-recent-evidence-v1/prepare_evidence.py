"""All previous tasks retained; labels and cached outcomes never enter requests."""
from pathlib import Path
import sys
from gate_common import ROOT, PREP, INPUTS, read, save, sha
from evidence_policy import VERSION

def main():
    old = ROOT / 'runs/memory_collab_prepare_20261026_01'
    trans = ROOT / 'runs/perception_transition_prepare_20261027_01'
    router = ROOT / 'runs/temporal_router_prepare_20261027_01/inputs'
    tasks = read(router / 'evaluation_tasks_LOCAL_ONLY.json')
    originals = {r['id']: r for r in read(old / 'partner_readonly/requests-no-labels.json')}
    originals.update({r['id']: r for r in read(trans / 'inputs/tasks_LOCAL_ONLY.json')})
    cards = {r['query']['id']: r['current_review_card'] for r in read(old / 'case_review_index_LOCAL_ONLY.json')}
    source_traces = {r['name']: r['sampled_trace'] for r in read(trans / 'inputs/sources_LOCAL_ONLY.json')}
    manifests = {r['name']: r for r in read(ROOT / 'runs/sember_prepare_20261025_01/available18_video_manifest.json')['videos']}
    sys.path.insert(0, str(ROOT / 'experiments/2b-perception-transition-v1'))
    from canonical_prompt import clean_options
    requests = []
    for task in tasks:
        q = originals[task['original_task_id']]
        if task['cohort'] == 'official102_reference':
            card = cards[q['id']]
            frames = card['recent4_frames']
            count = card['actual_committed_prefix']['committed_pairs']
        else:
            prefix = [f for f in source_traces[task['source_name']] if f['pts_s'] <= task['query_time_s']]
            assert len(prefix) % 2 == 0
            count, frames = len(prefix)//2, prefix[-8:]
        assert len(frames) == 8 and frames[-1]['pts_s'] <= task['query_time_s']
        assert all(a['frame_id'] < b['frame_id'] and a['pts_s'] < b['pts_s'] for a,b in zip(frames, frames[1:]))
        requests.append(dict(id=task['id'], question=q['question'], options=clean_options(q),
            source_name=task['source_name'], query_time_s=task['query_time_s'],
            committed_pairs=count, recent_frames=frames))
    assert len(requests) == 120 and len({r['source_name'] for r in requests}) == 18
    sources = []
    for name in sorted({r['source_name'] for r in requests}):
        pairs = {}
        for q in requests:
            if q['source_name'] != name: continue
            for i in range(4):
                frames = q['recent_frames'][i*2:i*2+2]
                index = q['committed_pairs']-4+i
                if index in pairs: assert pairs[index] == frames
                pairs[index] = frames
        sources.append(dict(**manifests[name], pairs=[dict(index=i, frames=f) for i,f in sorted(pairs.items())]))
    budget = dict(GPUs=1, CPUs=4, RAM_GiB=24, minutes=20, worker_seconds=1050,
        pretrained_loads=1, vision_classifier_forwards=1200, text_classifier_forwards=1200,
        warmup_classifier_forwards=4, visual_forwards=sum(len(s['pairs']) for s in sources),
        new_QA_answers=0, generated_tokens=0, training_steps=0, downloads=0,
        maximum_output_MiB=32, maximum_single_file_MiB=16, requeue=False, automatic_retry=False)
    save(INPUTS/'requests.json', requests)
    save(INPUTS/'sources.json', sources)
    save(INPUTS/'evaluation_tasks_LOCAL_ONLY.json', tasks)
    save(INPUTS/'frozen_branch_outputs_LOCAL_ONLY.json', read(router/'frozen_branch_outputs_LOCAL_ONLY.json'))
    save(INPUTS/'lexical_decisions.json', read(router/'lexical_decisions.json'))
    save(INPUTS/'protocol.json', dict(schema='recent-evidence-policy-v1', policy_version=VERSION,
        budget=budget, tasks=120, option_rotations=5, distinct_question_texts=113, sources=18,
        recent_real_frames=8, recent_projected_tokens=432, gate_context_limit=2048,
        visual_feature_archive=False, historical_visual_input_to_gate=False,
        bank_algorithms_unchanged=True, answers_are_frozen_selections_not_new_QA=True,
        gate_input='recent frames, their timestamps, question and displayed options; no cached answer',
        no_video_control='same question, options and recent timestamp text; video tokens omitted',
        decision='two cyclic A/B views averaged after semantic log-softmax mapping; exact tie full',
        teacher_or_sufficiency_ground_truth_available=False, failures_not_filtered=True,
        held_out=False, official102_independent_review=False, development6_assistant_review_only=True,
        cost_scope='selector component plus targeted recent encoding, not continuous streaming end-to-end',
        end_condition='One frozen development round. All seven previous routing quality gates plus superiority over same-prompt text-only control required before independent validation; otherwise stop without prompt/threshold sweep.',
        sources_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [
            old/'partner_readonly/requests-no-labels.json',old/'case_review_index_LOCAL_ONLY.json',
            trans/'inputs/tasks_LOCAL_ONLY.json',trans/'inputs/sources_LOCAL_ONLY.json',
            router/'evaluation_tasks_LOCAL_ONLY.json',router/'frozen_branch_outputs_LOCAL_ONLY.json']}))
    print(dict(status='prepared_not_authorized', tasks=120, budget=budget))

if __name__ == '__main__': main()
