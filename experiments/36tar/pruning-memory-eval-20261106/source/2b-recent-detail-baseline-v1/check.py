"""CPU contracts: finite budget, causal provenance, old-bank parity and isolation."""
import json
import os
from pathlib import Path
from guard import *


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    torch.manual_seed(20261103)
    arms = {a: RecentDetail(a) for a in ARMS}
    references = {b: Bank('LS', 54, budget=b, recent=432, closed=324) for b in (810, 864)}
    previous = None
    checks = parity = 0
    for i in range(160):
        # Deliberate event transitions and long runs force repeated history merges.
        x = torch.randn(108, 16) + (2 if (i // 12) % 2 else -2)
        node = pair_node(x, i, 2. * i, 2. * i + 1, (9, 12), torch.arange(1, 108, 2))
        desc = node.features.mean(0)
        start = previous is None or distance(previous, desc) > THRESHOLD
        previous = desc
        for bank in references.values():
            bank.append(node, starts_event=start)
        for arm, memory in arms.items():
            assert memory.append(x, i, 2. * i, 2. * i + 1) == start
            budget = 864 if arm == 'uniform864' else 810
            assert digest(memory.bank.nodes()) == digest(references[budget].nodes())
            parity += 1
            out = memory.export()
            s = memory.status()
            assert sum(len(n.features) for n in out) == s['stored_feature_tokens'] <= s['total_visual_budget']
            assert all(n.last <= i and n.end <= 2. * i + 1 for n in out)
            assert [e.node.first for e in memory.bank.recent] == list(range(max(0, i-7), i+1))
            before = memory.fingerprint()
            if arm == 'protect864':
                assert torch.equal(out[-1].features, x)
                assert digest(out[:-1]) == digest(arms['uniform810'].export()[:-1])
                assert memory.extra.shape == (54, 16)
                assert s['resident_tensor_bytes'] == (arms['uniform810'].status()['resident_tensor_bytes'] + 54*16*4)
            out[-1].features.zero_()
            assert memory.fingerprint() == before
            checks += 1
    rejected = 0
    m = arms['protect864']
    for bad, index, begin, end in [(torch.zeros(54,16),160,320.,321.),
                                    (torch.full((108,16),float('nan')),160,320.,321.),
                                    (torch.zeros(108,16),162,324.,325.)]:
        before = m.fingerprint()
        try:
            m.append(bad,index,begin,end)
        except ValueError:
            rejected += 1
        assert m.fingerprint() == before
    assert rejected == 3
    final = {a:m.status() for a,m in arms.items()}
    assert final['uniform864']['stored_feature_tokens'] == 864
    assert final['uniform810']['stored_feature_tokens'] == 810
    assert final['protect864']['stored_feature_tokens'] == 864
    for m in arms.values():
        m.close()
        assert not m.bank.nodes() and m.extra is None and m.previous is None
        try:
            m.export()
        except RuntimeError:
            pass
        else:
            raise AssertionError('closed export accepted')
    result = dict(passed=True, steps=160, legacy_bank_parity=parity,
                  causal_budget_export_checks=checks, rejected_invalid_inputs=rejected,
                  latest_full_exact=True, histories_matched_to_uniform810=True,
                  extra_stored_feature_tokens=54, full_encoder_required=True,
                  GPU=0, model_loads=0, capability_evidence=False, final=final)
    target = ROOT/'runs/recent_detail_prepare_20261103_01/CPU_local.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='final'}))


if __name__ == '__main__':
    main()
