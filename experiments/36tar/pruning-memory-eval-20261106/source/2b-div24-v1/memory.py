"""Finite five-arm archive; not a deployment memory bank."""
import itertools
import time
import torch
from common import ARMS
from features import raw_block, reduce
from forest import Node, raw, validate
from selection import div_indices, grid_indices


def variants(features, index, begin, end):
    block = raw_block(features, index, begin, end)
    nodes = {'full48': raw(block)}
    for arm in ARMS[1:]:
        if arm in ('odd24', 'tome24'):
            selected = reduce(block, 'odd' if arm == 'odd24' else 'tome')
            values, mass, xy = selected.features, selected.mass, selected.xy
        else:
            ids = grid_indices() if arm == 'grid24' else div_indices(block.features)
            values, mass, xy = (t[ids].clone() for t in (block.features, block.mass, block.xy))
        nodes[arm] = Node(values, mass, torch.cat((torch.full((24,1),index,dtype=torch.long),xy),1),
                          index,index,begin,end,(begin+end)/2)
        validate(nodes[arm])
    return nodes


class Archive:
    def __init__(self, pair_cap=320):
        self.pair_cap = pair_cap
        self.nodes = {a: [] for a in ARMS}
        self.reduction_seconds = {a: 0. for a in ARMS}

    @torch.inference_mode()
    def append(self, features, index, begin, end):
        if index != len(self.nodes['full48']) or index >= self.pair_cap:
            raise ValueError('finite contiguous archive required')
        if index and begin <= self.nodes['full48'][-1].end:
            raise ValueError('new real time pair required')
        started = time.monotonic()
        nodes = variants(features,index,begin,end)
        self.reduction_seconds['full48'] += time.monotonic()-started
        # Above is shared preparation timing, not a per-method performance result.
        for a in ARMS:
            self.nodes[a].append(nodes[a])
        return self.status()

    def status(self):
        return dict(pairs=len(self.nodes['full48']),
            tokens={a:sum(len(n.features) for n in ns) for a,ns in self.nodes.items()},
            tensor_bytes={a:sum(n.metadata()['tensor_bytes'] for n in ns) for a,ns in self.nodes.items()},
            diagnostic_archive_NOT_bounded_long_memory=True,
            reduction_timing='shared_five_arm_preparation_not_per_arm')

    def fingerprint(self):
        from fixed_stage2 import node_digest
        return {a:node_digest(ns) for a,ns in self.nodes.items()}

    def clear(self):
        self.nodes = {a:[] for a in ARMS}


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
