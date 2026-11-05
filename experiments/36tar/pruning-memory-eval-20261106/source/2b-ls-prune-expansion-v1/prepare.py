"""Freeze existing panels without selecting on results; no GPU or media reads."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/ls_prune_expansion_prepare_20261103_01'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    sources = {
        'regression300': ROOT / 'runs/anchor_merge_panels_20261103_01/comprehensive300.json',
        'memory478': ROOT / 'runs/isolated_recent_prepare_20261103_01/requests.json',
    }
    result = {}
    for name, path in sources.items():
        rows = json.loads(path.read_text(encoding='utf-8'))
        keys = [r.get('question_id', r.get('id')) for r in rows]
        assert len(keys) == len(set(keys)) == (300 if name == 'regression300' else 478)
        assert all(r['question'] and len(r['options']) >= 2 for r in rows)
        types = Counter(r.get('question_category', r.get('type')) for r in rows)
        groups = Counter(r.get('dataset', 'S-EMBER') for r in rows)
        times = [float(r.get('question_time', r.get('query_time_s'))) for r in rows]
        # Original role/split fields are provenance, not a claim of unseen testing.
        clean = [{**r, 'evaluation_role': 'opened_development_regression'} for r in rows]
        target = OUT / (name + '.json')
        with target.open('x', encoding='utf-8') as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        result[name] = dict(questions=len(rows), datasets=dict(groups), types=dict(types),
            video_files=len({(r.get('dataset', 'S-EMBER'), r.get('video_id', r.get('source_id'))) for r in rows}),
            query_seconds=dict(min=min(times), max=max(times),
                under180=sum(t < 180 for t in times), from180to600=sum(180 <= t < 600 for t in times),
                at_least600=sum(t >= 600 for t in times)),
            source=path.relative_to(ROOT).as_posix(), source_sha256=sha(path), manifest_sha256=sha(target))
    spec = dict(panels=result, arms=['full', 'fixed12', 'heuristic11', 'mask_ce'],
        bank='LS', budget_tokens=864, recent_tokens=432,
        scope='Autonomous feature-dependent schedules; practical combined effect, not pure pruning attribution',
        stage1_answers=1200, stage2_answers=1912,
        stage2_gate='Decide after complete stage1 paired results; no per-question result filtering',
        score='Option content correctness; formatting separate; no API evidence score',
        protocols='Preserve per-panel sampling and prompts. NExT-QA EOF is not midstream evidence.',
        not_unseen=True, execution_ready=False, gpu_submitted=False,
        outstanding=['runtime worker and exact prompt locks', 'media and training-source overlap checks',
                     'independent environment', 'CPU layouts and endpoint controls', 'GPU cost estimate and specific authorization'])
    with (OUT / 'panel_contract.json').open('x', encoding='utf-8') as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
