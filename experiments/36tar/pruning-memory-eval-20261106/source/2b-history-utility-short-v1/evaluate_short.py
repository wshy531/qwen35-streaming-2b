"""Apply frozen final heads to short development, without optimizer updates."""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/2b-history-utility-v1'))
import fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and args.out.resolve().is_relative_to((ROOT/'runs').resolve())
    assert fit.read(args.heads/'completed.json')['status'] == 'completed'
    manifest = fit.read(ROOT/'runs/history_utility_short_prepare_20261104_01/short_manifest.json')
    train = fit.read(fit.PREP/'train_manifest_LOCAL_ONLY.json')
    for key in ('video_id', 'sha256'):
        assert not ({g[key] for g in manifest} & {g[key] for g in train})
    data = fit.load([args.run], manifest)
    assert len(data['ids']) == 345
    torch.set_num_threads(1)
    args.out.mkdir()
    results = {}
    for blind in (False, True):
        name = 'blind_history' if blind else 'content_visible'
        checkpoint = torch.load(args.heads/(name+'.pt'), map_location='cpu', weights_only=True)
        assert checkpoint['epoch'] == 20 and checkpoint['train_questions'] == 1114
        assert checkpoint['blind_history'] == blind
        model = fit.HistoryUtilityHead()
        model.load_state_dict(checkpoint['state_dict'])
        model.eval().requires_grad_(False)
        report, rows = fit.evaluate(model, data, blind)
        results[name] = report
        with (args.out/(name+'_paired.jsonl')).open('x', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')
    original = [json.loads(line) for line in (args.run/'answers.jsonl').read_text().splitlines()]
    complete = [row for row in original if row['phase'] == 'full_short']
    assert len({(r['id'], r['shift']) for r in complete}) == 690
    pruned = [json.loads(line) for line in (ROOT/'runs/repair-guard-220499/answers.jsonl').read_text().splitlines()]
    pruned = [r for r in pruned if r['stage'] == 'final' and r['view'] == 'full_short']
    assert {(r['id'], r['shift']) for r in pruned} == {(r['id'], r['shift']) for r in complete}
    fit.save(args.out/'completed.json', dict(status='completed', questions=345, training_updates=0,
        full_short=dict(unpruned_correct=sum(r['correct'] for r in complete),
                        repaired_prune_correct=sum(r['correct'] for r in pruned), answers=690),
        bounded_LS_heads=results, note='separate full-short pruning comparison and bounded-LS head transfer'))


if __name__ == '__main__':
    main()
