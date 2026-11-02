"""Gold answer CE, optionally training-correct full-path option preservation."""
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/2b-learned-prune-profile-v1'))
from training import context, question_text


def option_ids(builder, device):
    ids = [builder.tokenize(c) for c in 'ABCDE']
    assert all(x.shape == (1, 1) for x in ids)
    return torch.tensor([x.item() for x in ids], device=device)


def last_logits(model, builder, records, q):
    emb, positions, ids = context(model, builder, records, question_text(q))
    h = model.model.language_model(inputs_embeds=emb, position_ids=positions,
        attention_mask=torch.ones(emb.shape[:2], device=emb.device, dtype=torch.long), use_cache=False)
    return model.lm_head(h.last_hidden_state[:, -1]).float(), ids.shape[1]


def loss_parts(logits, options, answer_index, teacher_options=None):
    target = options[answer_index].reshape(1)
    ce = F.cross_entropy(logits, target)
    kl = logits.new_zeros(())
    teacher_correct = teacher_options is not None and int(teacher_options.argmax(-1)) == answer_index
    if teacher_correct:
        kl = F.kl_div(logits[:, options].log_softmax(-1), teacher_options.detach().softmax(-1),
                      reduction='batchmean')
    return ce + 0.5 * kl, ce, kl, bool(teacher_correct)
