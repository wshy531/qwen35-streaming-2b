"""Read-only evidence assembly for one bounded history-interference diagnostic."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREP = ROOT / 'runs/perception_interference_prepare_20261027_01'
INPUTS = PREP / 'inputs'
OLD = ROOT / 'runs/memory_collab_prepare_20261026_01'
ARMS = ('recent', 'full', 'history_value_swap')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def displayed(task, shift):
    if type(shift) is not int or shift not in range(5):
        raise ValueError('one of the five frozen option rotations required')
    options = [re.sub(r'^[A-E]\.\s*', '', text) for text in task['options']]
    if len(options) != len(set(options)) or len(options) != 5:
        raise ValueError('five distinct complete options required')
    mapping = [(i + shift) % 5 for i in range(5)]
    return dict(question=task['question'], options=[options[i] for i in mapping]), mapping


def require_admission(manifest, reviews):
    """Independent-review gate; assistant development review never passes it."""
    byid = {r['pair_id']: r for r in reviews}
    if len(byid) != len(reviews):
        raise ValueError('duplicate review record')
    for pair in manifest['pairs']:
        review = byid.get(pair['pair_id'], {})
        if (review.get('decision') != 'admit'
                or not review.get('reviewer_identity')
                or review.get('reviewer_role') not in ('user', 'collaborator')
                or review.get('current_sufficient_in_committed_input') is not True
                or review.get('old_answer_supported_before_query') is not True
                or review.get('donor_unrelated_to_current_answer') is not True
                or not review.get('old_evidence_intervals_s')
                or review.get('manifest_sha256') != sha(INPUTS / 'manifest_LOCAL_ONLY.json')):
            raise ValueError('independent content admission pending: ' + pair['pair_id'])
    return True


def require_development_review(manifest, review):
    """User-delegated assistant review permits development diagnosis only."""
    if (review.get('scope') != 'assistant_reviewed_development_only'
            or review.get('reviewer_role') != 'assistant'
            or review.get('independent_review') is not False
            or review.get('formal_benchmark_admitted') is not False
            or review.get('user_delegation') != '你能自己做吗，我不太想审核'
            or review.get('manifest_sha256') != sha(INPUTS / 'manifest_LOCAL_ONLY.json')):
        raise ValueError('explicit development-only review and user delegation required')
    tasks = {t['id']: t for t in read(INPUTS / 'tasks_LOCAL_ONLY.json')}
    rows = {r['task_id']: r for r in review.get('tasks', [])}
    if len(rows) != len(review.get('tasks', [])) or set(rows) != set(tasks):
        raise ValueError('complete, unique per-task review required')
    for key, row in rows.items():
        task = tasks[key]
        if row.get('decision') not in ('include_development', 'uncertain_reference_only'):
            raise ValueError('explicit content disposition required')
        if row['decision'] == 'include_development':
            if row.get('reviewed_correct_index') != task['correct_index'] or row.get('evidence_sufficient') is not True:
                raise ValueError('reviewed answer/evidence mismatch')
        elif row.get('primary_content_scored') is not False:
            raise ValueError('uncertain evidence cannot enter primary content score')
        evidence = row.get('evidence_images', [])
        if not evidence:
            raise ValueError('actual reviewed images required')
        for item in evidence:
            path = (ROOT / item['relative_path']).resolve()
            if not path.is_relative_to(ROOT.resolve()) or sha(path) != item['sha256']:
                raise ValueError('review evidence changed')
    for pair in manifest['pairs']:
        if rows[pair['pair_id'] + ':current']['decision'] != 'include_development':
            raise ValueError('current evidence insufficient; revise panel before running')
    return True


def dependencies():
    sources = sorted(p for p in HERE.glob('*') if p.suffix in ('.py', '.md', '.sh'))
    sources += [ROOT / 'experiments/2b-sember-v1' / name for name in (
        'sember_common.py', 'sember_nodes.py', 'sember_banks.py',
        'sember_runtime.py', 'sember_protocol.py')]
    # A byte-pinned, read-only partner snapshot is used only for CPU scheduling parity.
    sources += [OLD / 'partner_readonly/policy.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
