"""Fixed, answer-independent offline evidence diagnostic; not a streaming method."""
from bisect import bisect_right
import hashlib
import json
import math

from ..sember_102_2b.readout import PROMPT_VERSION, question_prompt

ARMS = ('native_prefix', 'native_evidence', 'current_evidence', 'text_only_clean')
PROFILE = dict(schema='sember-readout-diagnostic-v1', questions=102, sources=18,
    arms=list(ARMS), maximum_answers=408, max_new_tokens=8, maximum_generated_tokens=3264,
    prompt_version=PROMPT_VERSION, canvas_hw=[192, 256], sampling_fps=1,
    model_revision='15852e8c16360a2fea060d615a32b45270f8a8fc', precision='float32',
    context_tokens=20000, maximum_pairs=300, pretrained_loads=2,
    cpus=4, RAM_GiB=48, minimum_free_GPU_GiB=64, maximum_GPU_allocated_GiB=72,
    maximum_RSS_GiB=40, slurm_minutes=120, worker_seconds=6900,
    automatic_retry=False, query_padding=False, held_out=False,
    evidence_source='official author interval; sufficiency not independently established',
    native_model='repository root qwen35_local original implementation',
    native_position_adapter='existing per-temporal-group get_rope_index compatibility adapter',
    native_scope='original full forward, no memory bank; not unmodified HF generate',
    evidence_selection='all committed original pairs with >=1 sample inside author interval',
    model_input='question/options and selected pixels/timestamps; no correct answers',
    status='prepared_not_executed')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def requests(queries, labels):
    """Annotations select oracle pixels offline; gold is never copied to requests."""
    by_id = {r['id']: r for r in labels}
    if len(by_id) != len(labels) or set(by_id) != {q['id'] for q in queries}:
        raise ValueError('question/annotation identity mismatch')
    result = []
    for q in queries:
        label = by_id[q['id']]
        start, end = label['answer_start_time'], label['answer_end_time']
        if (not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end))
                or not 0 <= start <= end <= q['query_time_s']
                or label['video_id'] != q['source_path'].split('/')[-1][:-4]
                or label['query_time_s'] != q['query_time_s']):
            raise ValueError('invalid evidence/source/query binding')
        result.append(dict(query=dict(q), evidence_interval=[start, end],
                           prompt_sha256=hashlib.sha256(question_prompt(q).encode()).hexdigest()))
    return result


def select(frames, query_time, interval):
    """Keep global pairing and all evidence hits. No thinning, padding, or future."""
    times = [r['pts_s'] for r in frames]
    if not times or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError('strictly chronological samples required')
    end = bisect_right(times, query_time)
    committed = end - end % 2
    if committed < 2 or committed // 2 > PROFILE['maximum_pairs']:
        raise ValueError('prefix outside preregistered pair budget; no truncation')
    start, stop = interval
    if not 0 <= start <= stop <= query_time:
        raise ValueError('evidence after query or invalid interval')
    pairs = [i for i in range(committed // 2)
             if any(start <= frames[j]['pts_s'] <= stop for j in (2*i, 2*i+1))]
    prefix = list(frames[:committed])
    evidence = [frames[j] for i in pairs for j in (2*i, 2*i+1)]
    return dict(prefix=prefix, evidence=evidence, evidence_pair_indices=pairs,
        pending_frame=frames[end-1] if end % 2 else None,
        evidence_status='selected' if pairs else 'no_committed_evidence_sample',
        prefix_sha256=digest(prefix), evidence_sha256=digest(evidence))
