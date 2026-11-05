"""Outcome supervision is a policy target, never visual-sufficiency truth."""
from collections import Counter
import hashlib
import re
import unicodedata

SEED = 'read-benefit-20261028-v1'

def normalized_question(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())

def parent_id(name):
    match = re.fullmatch(r'([0-9]+)_start_[0-9.]+_end_[0-9.]+\.mp4', name)
    if not match:
        raise ValueError('unknown S-EMBER source naming; explicit parent map needed')
    return match.group(1)

def supervision(rows):
    if len(rows) != 5 or {r['shift'] for r in rows} != set(range(5)):
        raise ValueError('five unique rotations per task required')
    counts = Counter()
    for row in rows:
        a, b = row['recent']['correct'], row['full']['correct']
        if type(a) is not bool or type(b) is not bool:
            raise ValueError('content correctness must be boolean')
        counts['recent_correct'] += a
        counts['full_correct'] += b
        counts['recent_only'] += a and not b
        counts['full_only'] += b and not a
        counts['both_correct'] += a and b
        counts['both_wrong'] += not a and not b
    net = counts['full_correct'] - counts['recent_correct']
    any_success = counts['recent_correct'] + counts['full_correct'] > 0
    return dict(**counts, target_gain=net / 5, fit_weight=int(any_success),
                preference=('full' if net > 0 else 'recent' if net < 0 else 'tie'),
                opposing_rotation_preferences=bool(counts['recent_only'] and counts['full_only']),
                visual_sufficiency_label=None,
                label_kind='fixed_model_fixed_paths_outcome_weak_supervision')

def source_groups(requests, source_metadata):
    """Union parent IDs, identical media SHA, and identical normalized questions."""
    parents = {r['source_name']: parent_id(r['source_name']) for r in requests}
    representatives = {p: p for p in set(parents.values())}
    def root(p):
        while representatives[p] != p:
            representatives[p] = representatives[representatives[p]]
            p = representatives[p]
        return p
    def union(a, b):
        a, b = root(a), root(b)
        representatives[max(a, b)] = min(a, b)
    seen_sha = {}
    for name in sorted(parents):
        sha = source_metadata[name]['sha256']
        if sha in seen_sha:
            union(parents[name], seen_sha[sha])
        seen_sha[sha] = parents[name]
    seen_question = {}
    for r in sorted(requests, key=lambda x: x['id']):
        question = normalized_question(r['question'])
        p = parents[r['source_name']]
        if question in seen_question:
            union(p, seen_question[question])
        seen_question[question] = p
    components = {}
    for p in sorted(representatives):
        components.setdefault(root(p), []).append(p)
    group_for_parent = {}
    for members in components.values():
        key = 'g_' + hashlib.sha256('|'.join(members).encode()).hexdigest()[:16]
        for p in members:
            group_for_parent[p] = key
    # Hash ordering and round robin use identities only, not outcomes or labels.
    groups = sorted(set(group_for_parent.values()),
                    key=lambda g: hashlib.sha256((SEED + g).encode()).hexdigest())
    folds = {g: i % 3 for i, g in enumerate(groups)}
    return {r['id']: dict(parent_id=parents[r['source_name']],
                          leakage_group=group_for_parent[parents[r['source_name']]],
                          fold=folds[group_for_parent[parents[r['source_name']]]])
            for r in requests}

def runtime_request(request):
    """Allowlist; evaluator labels, roles, source IDs and old outputs never render."""
    return dict(id=request['id'], question=request['question'],
                recent_frames=request['recent_frames'], query_time_s=request['query_time_s'])
