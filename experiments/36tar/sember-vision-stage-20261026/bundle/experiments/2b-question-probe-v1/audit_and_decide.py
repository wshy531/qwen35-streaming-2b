"""Audit causal evidence and controls, then the predeclared quality/cost gates."""
import argparse
import json
from pathlib import Path
from common import OUT, ARMS, read, save
from analyze import analyze


def main(result):
    done = read(result/'completed.json')
    assert done['status'] == 'complete' and done['answers'] == 909 and done['vision_encodings'] == 3981
    rows = [json.loads(line) for line in (result/'answers.jsonl').read_text(encoding='utf-8').splitlines()]
    tokens = [json.loads(line) for line in (result/'tokens.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len({(r['id'], r['arm']) for r in rows}) == 909
    keyed = {(r['id'], r['arm']): r for r in rows}
    token_map = {}
    for token in tokens:token_map.setdefault((token['query_id'], token['arm']), []).append(token)
    query_ids = {q['id'] for q in read(OUT/'inputs/questions.json')}
    probes = 0
    for uid in query_ids:
        original = keyed[uid, 'LS_full48']
        for arm in ARMS:
            row = keyed[uid, arm];stream = token_map[uid, arm]
            assert row['generated_ids'] == [t['token_id'] for t in stream]
            assert [t['token_index'] for t in stream] == list(range(len(stream)))
            assert row['banks_unchanged'] and row['branch_released'] and row['QA_includes_full_probe_and_rebuild']
            assert row['storage_node_digest'] == original['storage_node_digest']
            assert [(n['first'], n['last'], n['begin'], n['end'], n['depth']) for n in row['nodes']] == [
                (n['first'], n['last'], n['begin'], n['end'], n['depth']) for n in original['nodes']]
            if arm != 'LS_full48':assert row['visual_tokens']*2 == original['visual_tokens']
            if arm == 'LS_probe24':
                probes += 1;probe = row['probe']
                assert probe['completed_decoder_layers'] == 3 and probe['scored_attention_layer_one_based'] == 4
                assert not probe['probe_cache_created'] and not probe['full_N_by_N_attention_materialized']
                assert not probe['probe_logits_computed'] and probe['query_text_tokens_only']
                assert 0 <= probe['probe_seconds'] <= row['context_build_and_probe_seconds'] <= row['QA_total_seconds']
                assert 0 < probe['query_rows'] <= 128 and probe['key_tokens'] == original['input_tokens']
                assert len(row['selected_indices']) == len(original['nodes'])
                for n, indices in zip(original['nodes'], row['selected_indices']):
                    assert indices == sorted(set(indices)) and len(indices)*2 == n['tokens']
                    assert min(indices) >= 0 and max(indices) < n['tokens']
            else:assert row['probe'] is None and row['selected_indices'] is None
    assert probes == done['measured_probes'] == 303
    assert all(r['committed_frames_sha256'] for r in rows if r['cohort'] == 'stream34')
    assert len(done['profiles']) == 112
    assert all(p['source']['completed'] and p['source']['codec']['requested_threads'] == 4 for p in done['profiles'])
    analyze(result, OUT/'answer_results.json')
    scored = read(OUT/'answer_results.json');replay = scored['historical_replay'];tables = scored['tables']
    assert replay['expected'] == 606
    assert not any(replay[k] for k in ('generated_ID_mismatches', 'input_ID_mismatches', 'node_mismatches'))
    costs = {a: sum(r['QA_total_seconds'] for r in rows if r['arm'] == a) for a in ARMS}
    reduction = 1-costs['LS_probe24']/costs['LS_full48']
    subsets = ('primary269', 'stream_supported16')
    quality = all(tables[s]['scores']['LS_probe24']['correct'] >= tables[s]['scores']['LS_full48']['correct'] for s in subsets)
    recovery = any(tables[s]['scores']['LS_probe24']['correct'] > tables[s]['scores']['LS_read24']['correct'] for s in subsets)
    eligible = quality and recovery and reduction >= .15
    save(OUT/'integrity_and_decision.json', dict(status='passed', formal_answers=909, probes=303,
        streamed_tokens=len(tokens), historical_replay=606, QA_seconds=costs,
        probe_seconds=sum(r['probe']['probe_seconds'] for r in rows if r['arm'] == 'LS_probe24'),
        quality_gate_vs_full=quality, net_recovery_vs_odd_gate=recovery,
        QA_reduction_including_probe=reduction, engineering_gate=reduction >= .15, eligible=eligible,
        default_enabled=False, independent_test_performed=False,
        decision='separately approve M2/A06 and frozen new-source confirmation' if eligible else
                 'close first version; no layer/formula/ratio sweep',
        limitations=['all303 are opened development, not new-source proof',
                     'no net loss does not mean per-question losslessness or statistical noninferiority',
                     'shared vision costs do not establish independent end-to-end speed',
                     'bank storage remains384; no persistent observation-main KV',
                     'ranking cannot recover information lost during bank storage/merging']))
    print(dict(status='passed', eligible=eligible, QA_reduction_including_probe=reduction))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--results', type=Path, required=True)
    main(parser.parse_args().results)
