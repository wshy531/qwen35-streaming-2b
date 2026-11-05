"""Keep LS differentiable replay on the original model device.

Only projected leaves are temporarily copied to GPU; large visual activation
graphs remain absent until the per-pair backward. The original v1 is unchanged.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'experiments/2b-delayed-readiness-v1'))
from split_backward import capture_pairs, visual_backward, cache_bytes
from delayed_qa import training_context, last_logits
from objective import loss_parts, option_ids
import torch


def qa_leaf_gradients(model, builder, captured, question, answer_index):
    device = model.get_input_embeddings().weight.device
    # v1 replayed merges on CPU. Match whole-graph arithmetic placement here.
    leaves = {i:p.features.to(device).detach().clone().requires_grad_(True)
              for i,p in captured.pairs.items()}
    assert all(leaf.device == device for leaf in leaves.values())
    emb, pos, ids, meta = training_context(model, builder, captured.nodes, question,
                                          captured.replay, leaves)
    logits = last_logits(model, emb, pos)
    loss, _, _, _ = loss_parts(logits, option_ids(builder, device), answer_index)
    loss.backward()
    gradients = {i:leaf.grad.detach().cpu().clone() for i,leaf in leaves.items()
                 if leaf.grad is not None}
    assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
    return dict(loss=float(loss.detach()), logits=logits.detach().cpu(),
                ids=ids.detach().cpu(), positions=pos.detach().cpu(),metadata=meta,
                gradients=gradients, replay_device=str(device),
                projected_leaf_bytes=sum(x.numel()*x.element_size() for x in leaves.values()))
