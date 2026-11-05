"""Single-pair reduction and equal-label readout; no cross-time merge or question selection."""
import itertools
import time

import torch
from common import ARMS
from features import raw_block, reduce
from forest import Node, raw, validate


def variants(features, index, begin, end):
    block = raw_block(features, index, begin, end)
    nodes = {'full48': raw(block)}
    for arm, method in (('odd24', 'odd'), ('tome24', 'tome')):
        selected = reduce(block, method)
        coords = torch.cat((torch.full((24, 1), index, dtype=torch.long), selected.xy), dim=1)
        nodes[arm] = Node(selected.features, selected.mass, coords, index, index, begin, end, (begin + end) / 2)
        validate(nodes[arm])
    assert torch.equal(nodes['odd24'].coords, nodes['tome24'].coords)
    assert torch.equal(nodes['odd24'].features, nodes['full48'].features[1::2])
    return nodes


class Archive:
    """Finite per-clip diagnostic archive, NOT the deployed bounded memory bank."""
    def __init__(self, pair_cap=320):
        self.pair_cap = pair_cap
        self.nodes = {arm: [] for arm in ARMS}
        self.reduction_seconds = {arm: 0. for arm in ARMS}

    @torch.inference_mode()
    def append(self, features, index, begin, end):
        if index != len(self.nodes['full48']) or index >= self.pair_cap:
            raise ValueError('causal contiguous finite clip archive required')
        if index and begin <= self.nodes['full48'][-1].end:
            raise ValueError('strictly new pair required')
        started = time.monotonic()
        block = raw_block(features, index, begin, end)
        self.nodes['full48'].append(raw(block))
        self.reduction_seconds['full48'] += time.monotonic() - started
        for arm, method in (('odd24', 'odd'), ('tome24', 'tome')):
            started = time.monotonic()
            selected = reduce(block, method)
            node = Node(selected.features, selected.mass,
                torch.cat((torch.full((24, 1), index, dtype=torch.long), selected.xy), 1),
                index, index, begin, end, (begin + end) / 2)
            validate(node)
            self.nodes[arm].append(node)
            self.reduction_seconds[arm] += time.monotonic() - started
        assert torch.equal(self.nodes['odd24'][-1].coords, self.nodes['tome24'][-1].coords)
        return self.status()

    def status(self):
        return dict(pairs=len(self.nodes['full48']),
            tokens={a: sum(len(n.features) for n in ns) for a, ns in self.nodes.items()},
            tensor_bytes={a: sum(n.metadata()['tensor_bytes'] for n in ns) for a, ns in self.nodes.items()},
            diagnostic_archive_NOT_bounded_long_memory=True)

    def fingerprint(self):
        from fixed_stage2 import node_digest
        return {a: node_digest(ns) for a, ns in self.nodes.items()}

    def clear(self):
        self.nodes = {arm: [] for arm in ARMS}


@torch.inference_mode()
def context(model, builder, nodes, question, *, context_cap=20000, max_new_tokens=8):
    if not nodes or len(nodes) > 320 or sum(len(n.features) for n in nodes) > 15360:
        raise ValueError('finite full-clip diagnostic cap exceeded; never truncate')
    offset = builder.prefix_ids.shape[1]
    ids_all = [builder.prefix_ids]
    positions = [torch.arange(offset).expand(3, -1)]
    labels = []
    if any(a.last >= b.first or a.end >= b.begin for a, b in zip(nodes, nodes[1:])):
        raise ValueError('causal nonoverlapping node sequence required')
    for node in nodes:
        validate(node)
        if node.first != node.last:
            raise ValueError('this comparison never fuses different time pairs')
        if node.features.shape[1] != model.get_input_embeddings().embedding_dim:
            raise ValueError('projection width mismatch')
        # All arms represent the SAME frame pair, even when some spatial tokens
        # were removed/merged. Do not let an extra "compressed memory" prompt vary.
        label = f'<{node.center:.1f} seconds>'
        labels.append(label)
        ids = builder.tokenize(label + builder.processor.vision_start_token +
            builder.processor.video_token * len(node.features) + builder.processor.vision_end_token)
        coords = []
        for is_visual, group in itertools.groupby((ids[0] == builder.processor.video_token_id).tolist()):
            size = len(list(group))
            if is_visual:
                if size != len(node.features):
                    raise ValueError('placeholder mismatch')
                coords.extend((offset, offset + int(y), offset + int(x)) for _, y, x in node.coords)
                offset += 8
            else:
                coords.extend((offset + i,) * 3 for i in range(size))
                offset += size
        ids_all.append(ids)
        positions.append(torch.tensor(coords).T)
    suffix = builder.question_ids(question)
    if suffix.shape[1] > 128:
        raise ValueError('full question exceeds reserve; no truncation')
    ids_all.append(suffix)
    positions.append(torch.arange(offset, offset + suffix.shape[1]).expand(3, -1))
    ids = torch.cat(ids_all, 1)
    if ids.shape[1] + max_new_tokens > context_cap:
        raise ValueError('full input cap exceeded; no truncation')
    four = torch.cat((torch.arange(ids.shape[1])[None, :], torch.cat(positions, 1)), 0)[:, None, :]
    device = model.get_input_embeddings().weight.device
    embeddings = model.get_input_embeddings()(ids.to(device))
    visual = torch.cat([n.features for n in nodes]).to(embeddings)
    mask = (ids.to(device) == builder.processor.video_token_id).unsqueeze(-1).expand_as(embeddings)
    if int(mask.sum()) != visual.numel():
        raise ValueError('feature insertion mismatch')
    return embeddings.masked_scatter(mask, visual), four.to(device), ids, dict(
        input_tokens=ids.shape[1], visual_tokens=len(visual), pairs=len(nodes), labels=labels,
        suffix_tokens=suffix.shape[1], original_spatial_coordinates=True,
        no_cross_time_merge=True, equal_pair_labels=True, no_truncation=True)
