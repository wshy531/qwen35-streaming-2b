"""Audit frozen CPU fits and replay only cached train/development readouts."""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment as exp

torch = exp.torch
ROOT = exp.ROOT
OUT = exp.OUT


def pairwise(pred, target, valid, counts):
    delta = target[:, :, None] - target[:, None, :]
    diff = pred[:, :, None] - pred[:, None, :]
    upper = torch.triu(torch.ones(valid.shape[1], valid.shape[1], dtype=torch.bool), diagonal=1)[None]
    eligible = valid[:, :, None] & valid[:, None, :] & upper & (delta.abs() > 1e-4)
    right = delta * diff > 0
    out = {}
    for key, use in (
        ('all', eligible),
        ('same_cardinality', eligible & (counts[:, :, None] == counts[:, None, :])),
    ):
        rows = [float(right[j, mask].float().mean()) for j, mask in enumerate(use) if bool(mask.any())]
        out[key] = dict(pairs=int(use.sum()), questions=len(rows),
                        pooled=float(right[use].float().mean()) if bool(use.any()) else None,
                        question_equal_mean=sum(rows)/len(rows) if rows else None)
    return out


def baselines(packed):
    star = ROOT/'runs/7b-star-utility-223721'
    states = {row['id']: row for row in map(json.loads, (star/'states.jsonl').read_text().splitlines())}
    questions = {q['id']: q for role in ('train', 'development')
                 for q in exp.previous.read(ROOT/f'runs/head_data_coverage_20261112_01/STAR_{role}_inputs.json')}
    report = {}
    indices = {}
    for key, p in packed.items():
        name = '/'.join(key)
        choices = {k: [] for k in ('recent', 'LS', 'last')}
        if key[0] == 'STAR': choices['time_overlap'] = []
        for j, meta in enumerate(p['meta']):
            n = int(p['x'][4][j].sum())
            choices['recent'].append(0)
            choices['LS'].append(len(meta['choices'])-1)
            choices['last'].append(meta['choices'].index([n-1]) if n else 0)
            if key[0] == 'STAR':
                q = questions[meta['id']]
                match = re.search(r'video interval from ([0-9.]+) to ([0-9.]+) seconds', q['question'])
                assert match is not None
                a, b = map(float, match.groups())
                assert [a, b] == q['author_interval_s']
                history = states[meta['id']]['bank']['nodes'][:n]
                included = [k for k, node in enumerate(history) if node['end'] >= a and node['begin'] <= b]
                choices['time_overlap'].append(meta['choices'].index(included))
        report[name] = {}
        for method, selections in choices.items():
            ix = torch.tensor(selections)
            row = torch.arange(len(ix))
            gain = p['target'][row, ix]
            best = p['target'].masked_fill(~p['valid'], -torch.inf).max(1).values
            report[name][method] = dict(
                correct=float(p['correct'][row, ix].sum()), questions=len(ix), judgments=2*len(ix),
                mean_gain=float(gain.mean()), mean_regret=float((best-gain).mean()),
                mean_visual_tokens=float(p['tokens'][row, ix].mean()),
            )
        indices[name] = choices
    return report, indices


def audit_metrics(head, p, blind, reference):
    metrics, details = exp.evaluate(head, p, blind)
    with torch.no_grad(): pred = head(*p['x'], blind_history=blind)
    metrics['pairwise'] = pairwise(pred, p['target'], p['valid'], p['x'][-1].sum(-1))
    count = len(details)
    comparison = {}
    for method, chosen in reference.items():
        same = 0; positive = negative = tied = 0; gain = loss = 0.; delta = 0.
        for j, d in enumerate(details):
            same += d['selected'] == chosen[j]
            diff = d['correct'] - float(p['correct'][j, chosen[j]])
            positive += diff > 0; negative += diff < 0; tied += diff == 0
            gain += max(diff, 0.); loss += max(-diff, 0.)
            delta += d['gain'] - float(p['target'][j, chosen[j]])
        comparison[method] = dict(
            same_selection=same, questions=count, selection_agreement=same/count,
            positive_question_net=positive, negative_question_net=negative, tied_question_net=tied,
            correct_net=gain-loss, correct_gain_from_positive_question_nets=gain,
            correct_loss_from_negative_question_nets=loss, mean_gain_difference=delta/count,
        )
    metrics['vs_baselines'] = comparison
    return metrics


def main():
    torch.set_num_threads(1)
    inputs = exp.previous.read(OUT/'inputs.json')
    summary = exp.previous.read(OUT/'summary.json')
    completed = exp.previous.read(OUT/'completed.json')
    assert completed['fits'] == 9 and completed['updates'] == 1620
    for rel, digest in {**inputs['execution_sha'], **inputs['input_sha']}.items():
        assert exp.previous.sha(ROOT/rel) == digest, rel
    packed, _ = exp.previous.load()
    base, indices = baselines(packed)
    result = dict(
        purpose='Post-fit CPU audit; cached train/development only; no GPU, new QA, training or heldout readout',
        definitions=dict(
            correctness='Original option content correctness summed over two correlated option permutations',
            regret='Per-query maximum cached probability gain minus gain of selected subset',
            pairwise='Unordered legal candidate pairs; truth gaps <= 1e-4 excluded; predicted ties count incorrect',
            gain_loss='Per-question net change over two permutations, not per-permutation transition counts',
            blind='History descriptor alone zeroed; question, recent, metadata and bank structure retained',
            time_overlap='Only question-visible STAR interval; no hidden interval supplied to the learner',
        ),
        completed=completed, baselines=base, fits={}, checks={},
    )
    for seed in inputs['plan']['seeds']:
        arms = [summary[f'{arm}-{seed}'] for arm in inputs['plan']['arms']]
        assert len({a['initial_sha'] for a in arms}) == 1
        assert len({a['batch_order_sha'] for a in arms}) == 1
        old_curve = exp.previous.read(ROOT/f'runs/7b_star_head_fit_20261113_01/mixed-{seed}/curve.json')
        new_curve = exp.previous.read(OUT/f'huber-{seed}/curve.json')
        assert len(old_curve) == len(new_curve) == 18
        for old, new in zip(old_curve, new_curve):
            assert old['step'] == new['step']
            assert old['online_loss'] == new['online_loss']
            for domain in ('old', 'STAR'):
                assert old['metrics'][domain]['correct'] == new['metrics'][domain]['correct']
                assert old['metrics'][domain]['huber'] == new['metrics'][domain]['huber']
        result['checks'][str(seed)] = dict(initial_identical=True, batch_order_identical=True,
                                           huber_all_18_logged_points_reproduce_old_mixed=True)
    for name, frozen in summary.items():
        curve = exp.previous.read(OUT/name/'curve.json')
        assert frozen['selected_step'] == min(curve, key=lambda r: r['equal_domain_dev_regret'])['step']
        entry = dict(selected_step=frozen['selected_step'], clipped_steps=frozen['clipped_steps'],
                     clipping_fraction=frozen['clipped_steps']/180, selected={}, final={})
        blind = name.startswith('rank_blind_history-')
        for endpoint in ('selected', 'final'):
            path = OUT/name/f'{endpoint}.pt'
            assert exp.previous.sha(path) == frozen[f'{endpoint}_sha']
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
            head = exp.previous.Head()
            head.load_state_dict(ckpt['state_dict']);head.eval().requires_grad_(False)
            assert ckpt['step'] == (frozen['selected_step'] if endpoint == 'selected' else 180)
            for key, p in packed.items():
                label = '/'.join(key)
                ordinary, _ = exp.evaluate(head, p, blind)
                assert ordinary == frozen[f'{endpoint}_metrics'][label], (name, endpoint, label)
                entry[endpoint][label] = audit_metrics(head, p, blind, indices[label])
        result['fits'][name] = entry
    result['rank_vs_same_seed_huber'] = {}
    for seed in inputs['plan']['seeds']:
        for arm in ('rank', 'rank_blind_history'):
            name = f'{arm}-{seed}'
            delta = {}
            for endpoint in ('selected', 'final'):
                delta[endpoint] = {}
                for split, metric in result['fits'][name][endpoint].items():
                    huber = result['fits'][f'huber-{seed}'][endpoint][split]
                    delta[endpoint][split] = dict(
                        correct_net=metric['correct']-huber['correct'],
                        mean_regret_change=metric['mean_regret']-huber['mean_regret'],
                        pairwise_question_equal_change=metric['pairwise']['all']['question_equal_mean']-huber['pairwise']['all']['question_equal_mean'],
                        same_cardinality_question_equal_change=metric['pairwise']['same_cardinality']['question_equal_mean']-huber['pairwise']['same_cardinality']['question_equal_mean'],
                    )
            result['rank_vs_same_seed_huber'][name] = delta
    result['checks']['all_execution_and_input_sha_match'] = True
    result['checks']['all_nine_selected_and_nine_final_checkpoints_reload_exact_metrics'] = True
    result['checks']['all_selected_checkpoints_follow_first_minimum_equal_domain_regret'] = True
    result['limitations'] = [
        'All data here are development or training; checkpoints use development regret and are not unbiased test results.',
        'More clipping under rank than Huber is recorded; this objective comparison is not a loss-scale ablation.',
        'Both datasets keep original source weighting; no-history rows have zero training loss but remain in the denominator.',
        'An identical policy to LS cannot establish learned content-selection gain or reduced runtime.',
    ]
    result['analysis_code_sha'] = exp.previous.sha(Path(__file__))
    exp.previous.save(OUT/'analysis.json', result)
    print(json.dumps(dict(checks=result['checks'], baselines=base), ensure_ascii=False))


if __name__ == '__main__': main()
