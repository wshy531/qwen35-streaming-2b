"""Fixed CPU comparison: original utility Huber versus within-query ranking."""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'experiments/7b-star-head-fit-v1'))
import fit as previous

torch = previous.torch
F = torch.nn.functional
OUT = ROOT / 'runs/7b_head_ranking_repair_20261113_01'


def rank_loss(pred, target, valid, temperature=.1, tie=1e-4):
    """Every unordered legal pair once, including NULL; no target gradients."""
    assert pred.shape == target.shape == valid.shape and valid.dtype == torch.bool
    assert temperature > 0
    truth = target.detach()
    delta = truth[:, :, None] - truth[:, None, :]
    diff = pred[:, :, None] - pred[:, None, :]
    upper = torch.triu(torch.ones(valid.shape[1], valid.shape[1], dtype=torch.bool, device=pred.device), diagonal=1)
    active = valid[:, :, None] & valid[:, None, :] & upper & (delta.abs() > tie)
    weight = torch.where(active, delta.abs(), torch.zeros_like(delta))
    terms = F.softplus(-delta.sign() * diff / temperature)
    denominator = weight.sum((1, 2))
    rows = (weight * terms).sum((1, 2)) / denominator.clamp_min(1e-12)
    return rows.mean()


@torch.no_grad()
def evaluate(head, packed, blind=False):
    pred = head(*packed['x'], blind_history=blind)
    assert torch.isfinite(pred).all() and bool((pred[:, 0] == 0).all())
    details = []
    for j, meta in enumerate(packed['meta']):
        choices = meta['choices']
        chosen = previous.choose(pred[j, :len(choices)], choices)
        gain = float(packed['target'][j, chosen])
        maximum = float(packed['target'][j, :len(choices)].max())
        nodes = len(choices[chosen])
        details.append(dict(**meta, selected=chosen, selected_nodes=nodes,
                            correct=float(packed['correct'][j, chosen]), gain=gain,
                            regret=maximum-gain,
                            tokens=float(packed['tokens'][j, chosen]),
                            has_history=len(choices) > 1,
                            prediction=pred[j, :len(choices)].tolist()))
    n = len(details)
    metrics = dict(questions=n, judgments=2*n,
                   correct=sum(d['correct'] for d in details),
                   mean_gain=sum(d['gain'] for d in details)/n,
                   mean_regret=sum(d['regret'] for d in details)/n,
                   voluntary_null=sum(d['has_history'] and d['selected_nodes'] == 0 for d in details),
                   questions_with_history=sum(d['has_history'] for d in details),
                   mean_nodes=sum(d['selected_nodes'] for d in details)/n,
                   mean_visual_tokens=sum(d['tokens'] for d in details)/n,
                   count_histogram={str(k):sum(d['selected_nodes'] == k for d in details) for k in range(5)},
                   huber=float(previous.objective(pred, packed['target'], packed['valid'], torch.ones(n))),
                   rank_loss=float(rank_loss(pred, packed['target'], packed['valid'])))
    return metrics, details


def execution_files():
    paths = {HERE/'plan.json', HERE/'experiment.py', HERE/'rank_checks.py'}
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            path = Path(path).resolve()
            if path.is_relative_to(ROOT/'experiments') and path.suffix == '.py':
                paths.add(path)
    for rel in (
        'runs/7b_pruned_head_adapt_prepare_20261111_01/fit_sources.json',
        'runs/head_data_coverage_20261112_01/STAR_train_inputs.json',
        'runs/head_data_coverage_20261112_01/STAR_development_inputs.json',
        'runs/head_data_coverage_20261112_01/STAR_train_keys_LOCAL_ONLY.json',
        'runs/head_data_coverage_20261112_01/STAR_development_keys_LOCAL_ONLY.json',
    ):
        paths.add(ROOT/rel)
    return {str(p.relative_to(ROOT)): previous.sha(p) for p in sorted(paths)}


def baselines(p):
    totals = {k:dict(correct=0., gain=0., tokens=0., questions=0) for k in ('recent','LS','last','time_overlap')}
    for j,m in enumerate(p['meta']):
        n = int(p['x'][4][j].sum())
        indices = dict(recent=0, LS=len(m['choices'])-1, last=m['choices'].index([n-1]) if n else 0)
        # STAR time-overlap choice is already computed from question-visible
        # bounds by the frozen loader; read states to reconstruct it separately
        # in analysis. No ambiguous choice among the loader's allowed set here.
        for name,i in indices.items():
            t=totals[name];t['correct']+=float(p['correct'][j,i]);t['gain']+=float(p['target'][j,i]);t['tokens']+=float(p['tokens'][j,i]);t['questions']+=1
    return {k:dict(v,mean_gain=v['gain']/v['questions'],mean_visual_tokens=v['tokens']/v['questions']) for k,v in totals.items() if v['questions']}


def main():
    assert not OUT.exists(), 'Never overwrite a previous result'
    torch.set_num_threads(1)
    plan = previous.read(HERE/'plan.json')
    from rank_checks import run_checks
    checks = run_checks()
    packed, data_hashes = previous.load()
    locks = execution_files()
    OUT.mkdir(parents=True)
    previous.save(OUT/'inputs.json', dict(plan=plan, execution_sha=locks, input_sha=data_hashes, checks=checks))
    previous.save(OUT/'baselines.json', {'/'.join(k):baselines(p) for k,p in packed.items()})
    start=time.monotonic();summary={};initial_hashes={};batch_hashes={}
    for seed in plan['seeds']:
        for arm in plan['arms']:
            torch.manual_seed(seed)
            head=previous.Head();assert all(v.device.type == 'cpu' for v in head.parameters())
            assert sum(v.numel() for v in head.parameters()) == 713537
            initial = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in head.state_dict().values())).hexdigest()
            if seed in initial_hashes: assert initial == initial_hashes[seed]
            initial_hashes[seed]=initial
            opt=torch.optim.AdamW(head.parameters(), lr=plan['optimizer']['lr'], weight_decay=plan['optimizer']['weight_decay'])
            gen=torch.Generator().manual_seed(seed);queues={};best=float('inf');curve=[]
            folder=OUT/f'{arm}-{seed}';folder.mkdir();visited=hashlib.sha256()
            blind=arm == 'rank_blind_history';clipped_steps=0
            def draw(domain,n):
                p=packed[domain,'train'];total=len(p['meta']);buf=queues.setdefault(domain,[])
                while len(buf)<n:buf.extend(torch.randperm(total,generator=gen).tolist())
                ix=torch.tensor(buf[:n]);del buf[:n]
                visited.update(domain.encode()+ix.numpy().tobytes())
                return p,ix
            for step in range(1,plan['updates']+1):
                pieces=[draw('old',8),draw('STAR',8)]
                x=[torch.cat([p['x'][k][ix] for p,ix in pieces]) for k in range(len(previous.FIELDS)+1)]
                valid=torch.cat([p['valid'][ix] for p,ix in pieces]);target=torch.cat([p['target'][ix] for p,ix in pieces])
                opt.zero_grad(set_to_none=True)
                pred=head(*x,blind_history=blind)
                loss=previous.objective(pred,target,valid,torch.ones(len(target))) if arm=='huber' else rank_loss(pred,target,valid,plan['rank']['temperature'],plan['rank']['target_tie_tolerance'])
                assert torch.isfinite(loss)
                loss.backward();gradient_norm=float(torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True));clipped_steps+=gradient_norm>1.;opt.step()
                if step%10==0:
                    dev={d:evaluate(head,packed[d,'development'],blind)[0] for d in ('old','STAR')}
                    regret=sum(m['mean_regret'] for m in dev.values())/2
                    curve.append(dict(step=step,online_loss=float(loss.detach()),last_gradient_norm=gradient_norm,clipped_steps=clipped_steps,equal_domain_dev_regret=regret,metrics=dev))
                    if regret < best:
                        best=regret;selected_step=step;selected=copy.deepcopy(head.state_dict())
            visit_sha=visited.hexdigest()
            if seed in batch_hashes: assert visit_sha == batch_hashes[seed]
            batch_hashes[seed]=visit_sha
            final_state=copy.deepcopy(head.state_dict())
            final_metrics={'/'.join(k):evaluate(head,p,blind)[0] for k,p in packed.items()}
            torch.save(dict(state_dict=final_state,arm=arm,seed=seed,step=plan['updates'],width=3584),folder/'final.pt')
            head.load_state_dict(selected);head.eval().requires_grad_(False)
            torch.save(dict(state_dict=selected,arm=arm,seed=seed,step=selected_step,width=3584),folder/'selected.pt')
            selected_metrics={};details=[]
            for k,p in packed.items():
                selected_metrics['/'.join(k)],rows=evaluate(head,p,blind);details.extend(rows)
            previous.save(folder/'curve.json',curve);previous.save(folder/'selections.json',details)
            summary[f'{arm}-{seed}']=dict(initial_sha=initial,batch_order_sha=visit_sha,selected_step=selected_step,clipped_steps=clipped_steps,
                selected_metrics=selected_metrics,final_metrics=final_metrics,
                selected_sha=previous.sha(folder/'selected.pt'),final_sha=previous.sha(folder/'final.pt'))
            previous.save(OUT/'summary.json',summary)
            print(json.dumps(dict(arm=arm,seed=seed,step=selected_step,development={k:v for k,v in selected_metrics.items() if 'development' in k})),flush=True)
    for rel,digest in {**locks,**data_hashes}.items(): assert previous.sha(ROOT/rel)==digest,rel
    previous.save(OUT/'completed.json',dict(fits=9,updates=1620,seconds=time.monotonic()-start,cpu=True,gpu=0,new_qa=0,base_updates=0,
        reserve_opened=False,initial_sha=initial_hashes,batch_order_sha=batch_hashes))


if __name__=='__main__':main()
