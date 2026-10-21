"""Standard-library integrity and raw-answer rescoring; no media/model/network/GPU."""
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
RUNS = {
    'T': '2b_validation_T_prepare_20261020_01',
    'long': '2b_long_stream_prepare_20261020_01',
    'readability': '2b_memory_readability_prepare_20261020_01',
    'A06': '2b_closed_balance_prepare_20261020_01',
    'A04': '2b_spatial_time_prepare_20261021_01',
}


class Package:
    def __init__(self):
        self.manifest = json.loads((HERE / 'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))

    def open(self, relative):
        item = self.manifest['files'][relative]
        path = HERE / item['stored_path']
        return gzip.open(path, 'rt', encoding='utf-8') if item['gzip'] else path.open(encoding='utf-8')

    def read(self, relative):
        with self.open(relative) as f:
            return json.load(f)

    def rows(self, relative):
        with self.open(relative) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def path(self, tag, name):
        return 'bundle/runs/' + RUNS[tag] + '/' + name

    def get(self, tag, name):
        return self.read(self.path(tag, name))

    def check_integrity(self):
        banned = {'.mp4', '.avi', '.mov', '.png', '.jpg', '.jpeg', '.npz', '.npy', '.pt', '.pth', '.safetensors'}
        for relative, item in self.manifest['files'].items():
            assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
            assert not Path(item['stored_path']).is_absolute() and '..' not in Path(item['stored_path']).parts
            assert Path(relative).suffix.lower() not in banned, relative
            assert Path(relative).name != 'GPU_authorization.json', relative
            path = HERE / item['stored_path']
            assert path.stat().st_size == item['bytes'], relative
            with path.open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == item['sha256'], relative
            opener = gzip.open if item['gzip'] else open
            with opener(path, 'rb') as f:
                h = hashlib.sha256()
                size = 0
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
                    size += len(chunk)
            assert size == item['original_bytes'] and h.hexdigest() == item['original_sha256'], relative
            if Path(relative).suffix == '.py':
                with self.open(relative) as f:
                    ast.parse(f.read(), filename=relative)
        repo = HERE.parents[2]
        base = HERE.parent / 'memory-eval-2b/bundle'
        checked = 0
        for relative, expected in self.manifest['external_dependencies'].items():
            if relative.startswith('repo/'):
                path = repo / relative[5:]
            elif relative.startswith('previous_bundle/'):
                path = base / relative[len('previous_bundle/'):]
            else:
                continue  # Asset identity only: weights/tokenizer not required to rescore.
            with path.open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == expected, relative
            checked += 1
        return checked


def choice(text, n):
    match = re.fullmatch(r'\s*([A-Ea-e])\s*[.!]?\s*', text or '')
    value = ord(match[1].upper()) - 65 if match else -1
    return value if 0 <= value < n else None


def streamed(package, tag, arms):
    rows = {q['id']: q for q in package.get('long', 'stream_requests34.json')}
    labels = {q['id']: q for q in package.get('long', 'stream_labels34_LOCAL_ONLY.json')}
    snapshots = {q['query_id']: q for q in package.get('long', 'stream_query_snapshots34.json')}
    files = ['gpu_217776/native/events.jsonl', 'gpu_217776/memory/events.jsonl'] if tag == 'long' else [
        'gpu_217786/results/events.jsonl' if tag == 'A06' else 'gpu_217797/results/events.jsonl']
    answers = {}
    for name in files:
        for e in package.rows(package.path(tag, name)):
            if e['event']['kind'] != 'query':
                continue
            uid = e['event']['query_id']
            assert uid in rows
            assert e['committed_frames_sha256'] == snapshots[uid]['committed_frames_sha256']
            assert e['event']['committed_through_s'] <= rows[uid]['query_time_s']
            values = {e['arm']: e['answer']} if tag == 'long' else e['answer']['arms']
            for arm, answer in values.items():
                assert arm in arms and (uid, arm) not in answers
                answers[uid, arm] = answer
    assert set(answers) == {(uid, arm) for uid in rows for arm in arms}
    groups = {
        'all28': [uid for uid, q in rows.items() if q['cohort'] == 'official_OVO_unchanged'],
        'supported16': [uid for uid, q in rows.items() if q['cohort'] == 'official_OVO_unchanged' and labels[uid]['verdict'] == 'supported'],
        'manual6': [uid for uid, q in rows.items() if q['cohort'] == 'manual_development'],
    }
    table = {name: {arm: sum(choice(answers[uid, arm].get('text'), len(rows[uid]['options'])) == labels[uid]['answer_index']
                             for uid in ids) for arm in arms} for name, ids in groups.items()}
    reported = package.get(tag, 'strict_results.json')
    names = ['official_all28', 'official_supported16', 'manual_development6'] if tag == 'long' else [
        'original_all28', 'original_supported16', 'manual_development6']
    for group, old in zip(groups, names):
        assert table[group] == {arm: value['correct'] for arm, value in reported['summary'][old]['scores'].items()}
    assert reported['actual_records'] == len(answers)
    return table, answers


def rescore(package):
    result = {}
    trows = {q['id']: q for q in package.get('T', 'scorer_rows_LOCAL_ONLY.json')}
    texpected = package.get('T', 'gpu_217215/result/analysis.json')['tables']['all']
    tanswers = {}
    for stage in ('native', 'memory'):
        for p in package.rows(package.path('T', f'gpu_217215/result/{stage}/predictions.jsonl')):
            key = p['id'], p['arm']
            assert key not in tanswers and key[0] in trows and key[1] in texpected
            tanswers[key] = p
    assert set(tanswers) == {(uid, arm) for uid in trows for arm in texpected}
    totals = Counter()
    for (uid, arm), p in tanswers.items():
        q = trows[uid]['original']
        totals[arm] += p['status'] == 'ok' and choice(p.get('text'), len(q['options'])) == q['answer_index']
    assert dict(totals) == {a: v['strict_correct'] for a, v in texpected.items()}
    assert all(tanswers[uid, 'native_original']['generated_ids'] == tanswers[uid, 'full_pairwise_reference']['generated_ids'] for uid in trows)
    result['T_98_strict'] = dict(totals)
    tables, old = streamed(package, 'long', ['native', 'main_only', 'M0', 'M2', 'text_only'])
    result['long_34'] = tables
    for tag, arms, aliases in [
        ('A06', ['M0', 'M2', 'A06'], {'M0': 'M0', 'M2': 'M2'}),
        ('A04', ['M0_384', 'M2_384', 'M0_768', 'M2_768', 'S24_384'], {'M0_384': 'M0', 'M2_384': 'M2'}),
    ]:
        tables, answers = streamed(package, tag, arms)
        assert all(answer['generated_ids'] == old[uid, aliases[arm]]['generated_ids']
                   for (uid, arm), answer in answers.items() if arm in aliases)
        result[tag + '_34'] = tables
    labels = {q['id']: q for q in package.get('readability', 'labels6_LOCAL_ONLY.json')}
    rows = {q['id']: q for q in package.get('readability', 'requests6_LOCAL.json')}
    answers = {}
    totals = Counter()
    for p in package.rows(package.path('readability', 'gpu_217784/results/predictions.jsonl')):
        key = p['id'], p['arm']
        assert key not in answers and key[0] in labels
        answers[key] = p
        totals[p['arm']] += p['status'] == 'ok' and choice(p['text'], len(rows[p['id']]['options'])) == labels[p['id']]['answer_index']
    reported = package.get('readability', 'readability_strict_results.json')
    assert dict(totals) == reported['strict_correct'] and len(answers) == 36
    assert set(answers) == {(uid, arm) for uid in labels for arm in totals}
    assert all(answers[uid, 'F_bank_reproduction']['generated_ids'] == old[uid, 'M2']['generated_ids'] for uid in labels)
    result['readability_6'] = dict(totals)
    ledger = package.read('bundle/runs/2b_memory_value_prepare_20261021_01/stage_GPU_budget_ledger.json')
    download = package.read('bundle/runs/2b_memory_value_prepare_20261021_01/download_plan.json')
    assert ledger['active_GPU_submission_allowed'] is False and ledger['automatic_resume'] is False
    assert download['download_allowed'] is False and download['automatic_resume'] is False
    result['status'] = 'paused_by_user_no_new_download_or_GPU'
    result['raw_answers_rescored'] = 1262
    return result


def main():
    package = Package()
    dependencies = package.check_integrity()
    result = rescore(package)
    print(json.dumps(dict(status='passed', files=len(package.manifest['files']),
        repository_dependencies_checked=dependencies, results=result, pretrained_loads=0,
        GPU_submissions=0, note='Historical answers rescored; not a new model experiment'), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
