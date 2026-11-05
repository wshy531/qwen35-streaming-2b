"""Aggregate complete paired quality and independent synchronized pipeline costs."""
import argparse
import collections
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT/'runs/stream_prune_cost_prepare_20261102_01'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    assert read(args.run/'completed.json')['status'] == 'complete'
    protocol = read(PREP/'protocol.json')
    labels = read(ROOT/'runs/evidence_scale_prepare_20261102_01/labels_LOCAL_ONLY.json')
    rows, startup = [], []
    for job in read(PREP/'jobs.json'):
        directory = args.run/f'block{job["block"]}_{job["arm"]}'
        assert read(directory/'completed.json')['questions'] == len(job['question_ids'])
        startup.append(read(directory/'startup.json'))
        rows.extend(json.loads(line) for line in (directory/'answers.jsonl').read_text(encoding='utf-8').splitlines())
    assert len(rows) == len({(r['id'],r['arm']) for r in rows}) == 60
    assert all(all(r['reference_match'].values()) for r in rows)
    assert sum(r['vision_calls'] for r in rows) == protocol['formal_visual_calls']
    generated = sum(len(r['generated_ids']) for r in rows)+sum(len(s['warmup']['generated_ids']) for s in startup)
    assert generated <= protocol['max_generated_tokens']
    assert sum(s['warmup']['vision_calls'] for s in startup) <= protocol['warmup_visual_call_upper']
    summary = {}
    for arm in protocol['arms']:
        subset = [r for r in rows if r['arm']==arm]
        assert len(subset) == 20
        summary[arm] = dict(n=20, option_correct=sum(r['original_selected_index']==labels[r['id']] for r in subset),
                            pipeline_seconds=sum(r['pipeline_wall_seconds'] for r in subset),
                            decode_sampling_seconds=sum(r['decode_sampling_driver_seconds'] for r in subset),
                            preprocessing_seconds=sum(r['encoding']['preprocess_transfer_seconds'] for r in subset),
                            vision_seconds=sum(r['encoding']['vision_seconds'] for r in subset),
                            QA_seconds=sum(r['timing']['QA_total_seconds'] for r in subset),
                            prefill_seconds=sum(r['timing']['prefill_seconds'] for r in subset),
                            generation_seconds=sum(r['timing']['generation_seconds'] for r in subset),
                            generated_tokens=sum(len(r['generated_ids']) for r in subset),
                            input_tokens=sum(r['input_tokens'] for r in subset), visual_tokens=sum(r['visual_tokens'] for r in subset),
                            peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in subset),
                            peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in subset),
                            max_selector_bytes=max(r['selector_bytes'] for r in subset))
    lookup = {(r['id'],r['arm']):r for r in rows}
    comparisons = []
    for a,b in [('full','fixed12'),('full','heuristic11'),('fixed12','heuristic11')]:
        ratios = [lookup[(q['id'],b)]['pipeline_wall_seconds']/lookup[(q['id'],a)]['pipeline_wall_seconds'] for q in read(PREP/'panel.json')]
        reduction = 1-summary[b]['pipeline_seconds']/summary[a]['pipeline_seconds']
        loss = summary[a]['option_correct']-summary[b]['option_correct']
        comparisons.append(dict(baseline=a, candidate=b, total_pipeline_reduction_fraction=reduction,
                                median_source_runtime_ratio=statistics.median(ratios), faster_sources=sum(r<1 for r in ratios),
                                net_option_loss=loss,
                                meets_engineering_screen=reduction>=.05 and loss<=0,
                                interpretation='Single balanced-order run per source. Passing is not statistical noninferiority or edge validation.'))
    output = dict(status='complete', summary=summary, paired=comparisons, model_loads=len(startup),
                  startup_load_seconds=sum(s['load_seconds'] for s in startup), generated_tokens=generated,
                  formal_visual_calls=protocol['formal_visual_calls'], reference_answers_all_matched=True,
                  judged_explanation_quality='Pending separate blind review', hardware='A800', wallclock_1x=False)
    with (args.run/'analysis.json').open('x',encoding='utf-8') as handle:
        json.dump(output,handle,ensure_ascii=False,indent=2)
    print(json.dumps(output,ensure_ascii=False))


if __name__ == '__main__':
    main()
