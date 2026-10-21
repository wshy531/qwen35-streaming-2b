"""Record one actually viewed full-prefix source; never infer pixel verdicts."""
from collections import Counter
import json
import sys

from prepare import OUT, ROOT, read, save, sha


def record(value):
    traces = read(OUT/'sampling_all40_1fps.json')['videos']
    source = value['source_path']
    trace = next(t for t in traces if t['source_path'] == source)
    originals = [q for q in read(OUT/'candidate_questions119.json') if q['source_path'] == source]
    assert set(value['viewed_pages']) == {p['path'] for p in trace['pages']}
    assert value['all_pages_actually_viewed'] is True
    assert {str(q['suffix']) for q in value['questions']} == {q['id'].split(':')[-1] for q in originals}
    assert len(value['questions']) == len(originals)
    assert value['source_note'] and value['source_group_id']
    assert value['parent_status'] in ['canonical_ID_only_no_observed_overlap',
                                      'related_to_existing_source','related_new_sources','uncertain']
    for page in trace['pages']:
        assert sha(ROOT/page['path']) == page['sha256']
    rows = []
    for item in value['questions']:
        original = next(q for q in originals if q['id'].split(':')[-1] == str(item['suffix']))
        assert item['verdict'] in ['supported','ambiguous','unsupported']
        assert item['stratum'] in ['F','T','C']
        assert item['note'] and item['late_cues_note']
        assert item['memory'] in ['past_not_repeated_in_sample','past_with_late_cue','recent','mixed','uncertain']
        witnesses = []
        for target in item['times']:
            frame = min(trace['frames'],key=lambda f:abs(f['pts_s']-target))
            assert abs(frame['pts_s']-target) < .15
            assert frame['pts_s'] <= original['query_time_s']+1e-9
            witnesses.append(dict(frame_id=frame['frame_id'],pts_s=frame['pts_s'],rgb_sha256=frame['rgb_sha256']))
        if item['verdict'] == 'supported':
            assert witnesses, 'visible evidence required for supported label'
        snapshot = next(q['snapshot'] for q in read(OUT/'sampling_all40_1fps.json')['questions'] if q['id'] == original['id'])
        rows.append(dict(id=original['id'],source_path=source,source_group_id=value['source_group_id'],
            original=original,content_verdict=item['verdict'],stratum=item['stratum'],
            witnesses=witnesses,memory_character=item['memory'],note=item['note'],
            late_repetition_or_context_cues=item['late_cues_note'],actual_sampling_snapshot=snapshot,
            parent_status=value['parent_status'],old_evidence_necessity_proven=False,
            reviewer='single_assistant_all_fixed1FPS_pages',independent_second_review='not_performed',
            model_outputs_used=False,original_labels_unchanged=True,
            provisional_content_eligible=item['verdict']=='supported' and value['parent_status'] not in [
                'related_to_existing_source','uncertain']))
    folder = OUT/'content_reviews'
    folder.mkdir(exist_ok=True)
    save(folder/(trace['source_group_id'].split(':')[-1]+'.json'),dict(
        source_path=source,source_note=value['source_note'],source_group_id=value['source_group_id'],
        parent_status=value['parent_status'],all_pages_actually_viewed=True,
        pages=[dict(path=p['path'],sha256=p['sha256']) for p in trace['pages']],questions=rows,
        model_outputs_used=False,second_review='not_performed'))
    done = [q for p in folder.glob('*.json') for q in read(p)['questions']]
    counts = Counter(q['content_verdict'] for q in done)
    with (ROOT/'实验总纲.md').open('a',encoding='utf-8') as stream:
        stream.write(f"\n1.47 T内容审核：{source}全部{len(trace['pages'])}页已实际查看，累计{len(done)}/119题，处置{dict(counts)}；单助手、未看模型输出、必要记忆未证明，原标签未改。\n")
    print(dict(recorded=source,total_reviewed=len(done),verdicts=dict(counts)),flush=True)


def finalize():
    originals = read(OUT/'candidate_questions119.json')
    sources = [read(p) for p in sorted((OUT/'content_reviews').glob('*.json'))]
    rows = [q for s in sources for q in s['questions']]
    assert len(sources) == 40 and len(rows) == len({q['id'] for q in rows}) == 119
    assert {q['id'] for q in rows} == {q['id'] for q in originals}
    assert all(s['all_pages_actually_viewed'] for s in sources)
    eligible = [q for q in rows if q['provisional_content_eligible']]
    save(OUT/'content_dispositions119.json',rows)
    save(OUT/'content_eligible_questions.json',eligible)
    result = dict(status='single_assistant_content_review_complete_not_independent_human_adjudication',
        sources=40,questions=119,pages=sum(len(s['pages']) for s in sources),
        verdicts=dict(Counter(q['content_verdict'] for q in rows)),
        eligible_questions=len(eligible),eligible_source_groups=len({q['source_group_id'] for q in eligible}),
        independent_second_review_completed=0,old_evidence_necessity_proven=0,
        model_outputs_used=False,original_labels_unchanged=True,
        analysis_protocol_sha256=sha(OUT/'analysis_protocol.json'),
        source_reviews_sha256={p.name:sha(p) for p in sorted((OUT/'content_reviews').glob('*.json'))})
    save(OUT/'content_review_receipt.json',result)
    print(result)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'finalize':
        finalize()
    else:
        record(json.load(sys.stdin))
