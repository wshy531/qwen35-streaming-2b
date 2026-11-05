"""Paired accuracy summaries for frozen scorer endpoints on the old regression panel."""
import argparse
import json
from collections import Counter
from pathlib import Path

from analyze_screen import paired, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    receipt = json.loads((args.run/'completed.json').read_text())
    assert receipt['status']=='completed' and not receipt['cpu_tiny']
    rows = [json.loads(line) for line in (args.run/'answers.jsonl').read_text().splitlines()]
    assert len(rows)==receipt['answers']==240
    groups = {arm:[row for row in rows if row['readout']==arm]
              for arm in ('initial','short_then_delayed','mixed','text')}
    for group in groups.values():
        assert len(group)==60 and len({r['video_id'] for r in group})==30
        assert Counter(r['cue_group'] for r in group)=={'C':20,'T':20,'D':20}
    metrics = {arm:summarize(group) for arm,group in groups.items()}
    family = {arm:{kind:summarize([r for r in group if r['cue_group']==kind]) for kind in ('C','T','D')}
              for arm,group in groups.items()}
    comparisons = {arm:paired(groups['initial'],groups[arm]) for arm in ('short_then_delayed','mixed')}
    comparisons['sequence_vs_mixed'] = paired(groups['mixed'],groups['short_then_delayed'])
    with (args.run/'analysis.json').open('x',encoding='utf-8') as f:
        json.dump(dict(metrics=metrics, family=family, comparisons=comparisons),f,indent=2)
    print(json.dumps(dict(metrics=metrics, paired_counts={
        arm:{k:len(v) for k,v in changes.items()} for arm,changes in comparisons.items()}),indent=2))


if __name__ == '__main__':
    main()
