"""Label-free query validation and explicit pending evidence admission."""
from collections import defaultdict
import copy
import math
from pathlib import PurePosixPath

ARMS = ('M0', 'M2', 'M2_A06', 'LS_shared384_v1', 'reservoir4', 'text_only')
QUERY_FIELDS = {'id', 'source_path', 'query_time_s', 'question', 'options', 'cohort'}


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_queries(requests, *, canonical=False):
    if not isinstance(requests, list) or not requests:
        raise ValueError('nonempty query list required')
    groups, seen = defaultdict(list), set()
    for row in requests:
        if not isinstance(row, dict) or set(row) != QUERY_FIELDS:
            raise ValueError('label-free query field allowlist violated')
        qid, path = row['id'], row['source_path']
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError('unique nonempty query ID required')
        if (not isinstance(path, str) or not path or '\\' in path or ':' in path
                or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts
                or str(PurePosixPath(path)) != path):
            raise ValueError('normalized relative POSIX source path required')
        query = row['query_time_s']
        if not finite_number(query) or query < 0:
            raise ValueError('finite nonnegative query time required')
        if (not isinstance(row['question'], str) or not row['question'].strip()
                or not isinstance(row['options'], list) or not 2 <= len(row['options']) <= 5
                or any(not isinstance(x, str) or not x.strip() for x in row['options'])
                or not isinstance(row['cohort'], str) or not row['cohort'].strip()):
            raise ValueError('complete question, options and cohort required')
        if canonical and not ((qid.startswith('ovo:') and row['cohort'] == 'official_OVO_unchanged')
                              or (qid.startswith('local_longstream_v1:') and row['cohort'] == 'manual_development')):
            raise ValueError('canonical cohort/ID mismatch')
        seen.add(qid)
        groups[path].append(copy.deepcopy(row))
    for group in groups.values():
        group.sort(key=lambda q: q['query_time_s'])  # stable original order for ties
    if canonical and (len(seen) != 34 or len(groups) != 6
                      or sum(q.startswith('ovo:') for q in seen) != 28
                      or sum(q.startswith('local_longstream_v1:') for q in seen) != 6):
        raise ValueError('canonical 28 original + 6 manual / 6 sources required')
    return dict(groups)


def bind_official_labels(requests, records):
    """Require exact request identity; never infer labels from predictions."""
    validate_queries(requests)
    byid = {q['id']: q for q in requests}
    result = {}
    for item in records:
        qid, raw = item['request_id'], item['record']
        if qid not in byid or qid in result:
            raise ValueError('duplicate or unbound official label')
        row = byid[qid]
        if (qid != 'ovo:'+str(raw['id']) or raw['video'] != row['source_path']
                or raw['realtime'] != row['query_time_s'] or raw['question'] != row['question']
                or raw['options'] != row['options']):
            raise ValueError('official label/request identity drift')
        gt = raw['gt']
        if type(gt) is not int or not 0 <= gt < len(row['options']) or raw['answer'] != row['options'][gt]:
            raise ValueError('invalid official label')
        result[qid] = gt
    return result


def pending_catalog(requests, verified_labels):
    validate_queries(requests)
    if not set(verified_labels).issubset(q['id'] for q in requests):
        raise ValueError('unbound verified labels')
    catalog = []
    for row in requests:
        label = verified_labels.get(row['id'])
        if label is not None and (type(label) is not int or not 0 <= label < len(row['options'])):
            raise ValueError('invalid verified label')
        catalog.append(dict(id=row['id'], original_label_index=label,
            original_label_verified=row['id'] in verified_labels,
            admission_group='pending', evidence_intervals=None, sampled_evidence_present=None,
            later_repetition_or_context_cues=None, manual_label=None,
            human_semantic_review='pending', eligible_for_confirmed_accuracy=False))
    return catalog
