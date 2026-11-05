"""Three readouts of the same bank. Distinct name avoids the legacy readout module."""
from dataclasses import replace
import sys
import torch
from common import ROOT, ARMS

sys.path.insert(0, str(ROOT / 'experiments/2b-sember-v1'))
from sember_runtime import context
from sember_nodes import digest, validate


def split(bank):
    if bank.policy != 'LS' or bank.recent_cap != 432 or bank.budget != 864:
        raise ValueError('frozen LS4_4 bank required')
    history = [entry.node for entry in bank.history]
    recent = [entry.node for entry in bank.recent]
    if not history or len(recent) != 4 or any(len(n.features) != 108 for n in history + recent):
        raise ValueError('complete four-pair recent suffix and history required')
    return history, recent


def swapped_history(history, donor_features):
    if len(donor_features) != 4:
        raise ValueError('four frozen donor pairs required')
    result = []
    for i, node in enumerate(history):
        value = donor_features[i % 4]
        if value.shape != node.features.shape or value.device.type != 'cpu' or value.dtype != torch.float32:
            raise ValueError('donor feature shape/device/dtype mismatch')
        swapped = replace(node.copy(), features=value.clone())
        validate(swapped)
        result.append(swapped)
    return result


@torch.inference_mode()
def compose(model, builder, bank, question, arm, donor_features):
    if arm not in ARMS:
        raise ValueError('unregistered diagnostic condition')
    before = digest(bank.nodes())
    history, recent = split(bank)
    full = context(model, builder, history + recent, question, visual_limit=864,
                   context_limit=8192, max_new_tokens=8)
    if arm == 'full':
        result = full
    elif arm == 'recent':
        result = context(model, builder, recent, question, visual_limit=864,
                         context_limit=8192, max_new_tokens=8)
        emb, pos, ids, meta = result
        prefix = builder.prefix_ids.shape[1]
        removed = full[2].shape[1] - ids.shape[1]
        if removed <= 0 or not torch.equal(ids[:, prefix:], full[2][:, prefix + removed:]):
            raise ValueError('recent IDs are not an exact full suffix')
        if not torch.equal(emb[:, prefix:], full[0][:, prefix + removed:]):
            raise ValueError('recent embeddings changed')
        pos = pos.clone()
        pos[1:, :, prefix:] = full[1][1:, :, prefix + removed:]
        result = emb, pos, ids, dict(meta, removed_history_input_tokens=removed)
    else:
        result = context(model, builder, swapped_history(history, donor_features) + recent,
                         question, visual_limit=864, context_limit=8192, max_new_tokens=8)
        if not torch.equal(result[1], full[1]) or not torch.equal(result[2], full[2]):
            raise ValueError('history intervention changed positions or text IDs')
        visual = (full[2] == builder.processor.video_token_id)[0]
        historical_columns = torch.where(visual)[0][:sum(len(n.features) for n in history)]
        stable = torch.ones(full[2].shape[1], dtype=torch.bool, device=full[0].device)
        stable[historical_columns.to(stable.device)] = False
        if not torch.equal(result[0][:, stable], full[0][:, stable]):
            raise ValueError('intervention escaped historical visual columns')
    if digest(bank.nodes()) != before:
        raise RuntimeError('readout mutated bank')
    emb, pos, ids, meta = result
    return emb, pos, ids, dict(meta, arm=arm, bank_digest=before,
        source_substitution_is_artificial=arm == 'history_value_swap',
        no_history_in_recent_language_forward=arm == 'recent',
        recent_coordinates_anchored_to_full=True, bank_unchanged=True)
