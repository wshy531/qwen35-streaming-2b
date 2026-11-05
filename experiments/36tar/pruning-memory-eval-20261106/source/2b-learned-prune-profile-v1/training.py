"""Differentiable fresh QA context, preserving the existing bank readout layout."""
import itertools
import torch


def context(model, builder, records, question):
    offset = builder.prefix_ids.shape[1]
    ids_list = [builder.prefix_ids]
    positions = [torch.arange(offset).expand(3, -1)]
    values = []
    for index, begin, end, features, selected in records:
        assert len(features) == len(selected) and 1 <= len(selected) <= 108
        ids = builder.tokenize(f'<{(begin+end)/2:.1f} seconds>' + builder.processor.vision_start_token +
                               builder.processor.video_token*len(features) + builder.processor.vision_end_token)
        coords = []
        for modality, group in itertools.groupby((ids[0] == builder.processor.video_token_id).tolist()):
            count = len(list(group))
            if modality:
                assert count == len(selected)
                coords.extend((offset, offset+int(i)//12, offset+int(i)%12) for i in selected)
                offset += 12
            else:
                coords.extend((offset+i,)*3 for i in range(count)); offset += count
        ids_list.append(ids); positions.append(torch.tensor(coords).T); values.append(features)
    suffix = builder.question_ids(question)
    ids_list.append(suffix); positions.append(torch.arange(offset, offset+suffix.shape[1]).expand(3, -1))
    ids = torch.cat(ids_list, 1)
    assert len(records) <= 20 and ids.shape[1] <= 4096
    pos = torch.cat((torch.arange(ids.shape[1])[None], torch.cat(positions, 1)), 0)[:, None]
    device = model.get_input_embeddings().weight.device
    embeddings = model.get_input_embeddings()(ids.to(device))
    visual = torch.cat(values).to(embeddings)
    mask = (ids.to(device) == builder.processor.video_token_id).unsqueeze(-1).expand_as(embeddings)
    assert int(mask.sum()) == visual.numel()
    return embeddings.masked_scatter(mask, visual), pos.to(device), ids


def task_loss(model, builder, records, question, answer):
    embeddings, positions, ids = context(model, builder, records, question)
    targets = builder.tokenize(answer).to(embeddings.device)
    n = targets.shape[1]; assert 1 <= n <= 4
    # Teacher forcing through all answer tokens; only answer positions are scored.
    extra = model.get_input_embeddings()(targets[:, :-1])
    if n > 1:
        p = positions[:, :, -1:] + torch.arange(1, n, device=positions.device)[None, None]
        positions = torch.cat((positions, p), -1)
    embeddings = torch.cat((embeddings, extra), 1)
    output = model.model.language_model(inputs_embeds=embeddings, position_ids=positions,
              attention_mask=torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long), use_cache=False)
    logits = model.lm_head(output.last_hidden_state[:, -n:])
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)), ids.shape[1]


def question_text(q):
    return q['question'] + '\n' + '\n'.join(f'{chr(65+i)}. {x}' for i, x in enumerate(q['options'])) + '\nAnswer with one option letter.'
