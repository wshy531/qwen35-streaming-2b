"""Use the inference bank's layout with differentiable replayed visual values."""
import torch

from resolution_runtime import context


def training_context(model, builder, nodes, question, replay, leaves):
    if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
        raise RuntimeError("Delayed QA requires autograd")
    emb, positions, ids, metadata = context(model, builder, nodes, question)
    # The existing context supplies token IDs, time labels and MRoPE coordinates.
    # Replace only the visual slots outside inference mode.
    visual = torch.cat(replay.values(nodes, leaves)).to(emb)
    mask = (ids.to(emb.device) == builder.processor.video_token_id)
    mask = mask.unsqueeze(-1).expand_as(emb)
    if int(mask.sum()) != visual.numel():
        raise ValueError("Replayed visual embedding size differs from bank layout")
    return emb.clone().masked_scatter(mask, visual), positions.clone(), ids.clone(), metadata


def last_logits(model, embeddings, positions):
    hidden = model.model.language_model(
        inputs_embeds=embeddings,
        position_ids=positions,
        attention_mask=torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long),
        use_cache=False,
    )
    return model.lm_head(hidden.last_hidden_state[:, -1]).float()
