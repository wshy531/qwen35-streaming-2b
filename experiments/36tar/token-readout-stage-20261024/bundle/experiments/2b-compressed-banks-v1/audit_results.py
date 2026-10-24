"""Independent post-run integrity checks; no model load or submission."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / 'runs/2b_compressed_banks_prepare_20261023_01'
RAW = BASE / 'results-218058'
EXPECTED = {
    'answers.jsonl': '61dc3893d95cb4c476aa02f63fc212bae729d8f2139db4c2665961d971b3d797',
    'events.jsonl': '99f354ccadabb3320178f36315acd3bcc4a9332d628549599f383888efb1d7cb',
    'tokens.jsonl': 'e783fd7343f356265bccff64ce7840b974ed25edc190170587603334b106a877',
    'loaded.json': 'ee570aa1f2af400fb3bab7c706fed5247287f15811af7bff52d467fd997ad409',
    'completed.json': 'd7b9ec457efa240f6de762e403acc800bd844a9cda3c366e2fcb438a909ae362',
}


def load(name):
    return json.loads((BASE / name).read_text(encoding='utf-8'))


def lines(name):
    with (RAW / name).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def main():
    hashes = {}
    for name, expected in EXPECTED.items():
        with (RAW / name).open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert hashes[name] == expected, name
    complete = json.loads((RAW / 'completed.json').read_text(encoding='utf-8'))
    scores = load('answer_results.json')
    expected_arms = set(scores['tables']['primary269']['scores'])
    answers = {}
    per_arm = Counter()
    visual_max, input_max = Counter(), Counter()
    for row in lines('answers.jsonl'):
        key = row['id'], row['arm']
        assert key not in answers
        answers[key] = row
        assert row['status'] == 'ok'
        assert row['banks_unchanged'] and row['branch_released'] and row['no_main_language_cache']
        assert row['visual_tokens'] <= 384
        assert sum(n['tokens'] for n in row['nodes']) == row['visual_tokens']
        assert len(row['generated_ids']) <= 8
        assert row['input_tokens'] <= 2048
        per_arm[row['arm']] += 1
        visual_max[row['arm']] = max(visual_max[row['arm']], row['visual_tokens'])
        input_max[row['arm']] = max(input_max[row['arm']], row['input_tokens'])
    assert len(answers) == complete['answers'] == 2727
    assert set(per_arm) == expected_arms and set(per_arm.values()) == {303}
    token_rows = defaultdict(list)
    for row in lines('tokens.jsonl'):
        token_rows[row['query_id'], row['arm']].append(row)
    assert set(token_rows) == set(answers)
    for key, row in answers.items():
        observed = token_rows[key]
        assert [r['token_index'] for r in observed] == list(range(len(observed)))
        assert [r['token_id'] for r in observed] == row['generated_ids']
        assert all(a['monotonic_s'] <= b['monotonic_s'] for a,b in zip(observed, observed[1:]))
    assert sum(map(len, token_rows.values())) == complete['generated_tokens'] == 5541
    group_ends = []
    selection_count = 0
    for row in lines('events.jsonl'):
        if row.get('kind') == 'group_end':
            group_ends.append(row)
            assert not row['state']['invalid']
            for bank in row['state']['banks']['banks'].values():
                assert bank['visual_tokens'] <= 384
        if row.get('kind') in ('selection', 'query_only_selection'):
            trace = row['trace']
        elif 'selection_trace' in row:
            trace = row['selection_trace']
        elif 'selection_trace' in row.get('state', {}):
            trace = row['state']['selection_trace']
        else:
            continue
        selection_count += 1
        for selected in trace['selected_indices'].values():
            assert len(selected) == 24 and len(set(selected)) == 24
            assert sorted(selected) == selected and min(selected) >= 0 and max(selected) < 48
    assert len(group_ends) == len({r['group'] for r in group_ends}) == 112
    assert selection_count == 3981
    assert sum(r['state']['vision_encodings'] for r in group_ends) == complete['vision_encodings'] == 3981
    replay = scores['historical_replay']
    assert replay['expected'] == 1212
    assert not replay['generated_ID_mismatches'] and not replay['input_ID_mismatches'] and not replay['node_mismatches']
    report = dict(status='pass', SHA256=hashes, answers=len(answers), generated_tokens=5541,
                  per_arm=dict(per_arm), visual_max=dict(visual_max), input_max=dict(input_max),
                  group_ends=len(group_ends), selection_traces_checked=selection_count,
                  historical_replay_rows=1212, complete=complete,
                  limitations=['Structural integrity does not establish correctness or speed benefit.',
                               'Selection indices checked structurally; no re-encoding or score recomputation.',
                               'Shared-process GPU peak and runtime are not standalone per-arm measurements.'])
    destination = BASE / 'result_audit.json'
    with destination.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
