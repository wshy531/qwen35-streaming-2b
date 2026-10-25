"""Grid-aware LS/A06 adapters; frozen files and default policies are untouched.

Merge math, event threshold, four recent pairs and selector are preserved.
Token budget is eight full projected pairs, NOT constant across resolutions.
"""
from collections import deque
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from functools import lru_cache
import torch
from ..online_memory_2b.frozen import policies, THRESHOLD
from .controls import PRESETS,geometry


@lru_cache(maxsize=1)
def numeric():
    return policies()


@dataclass(frozen=True)
class Node:
    features: object
    mass: object
    coords: object
    first: int
    last: int
    begin: float
    end: float
    center: float
    grid: tuple
    groups: int = 1
    depth: int = 0

    def copy(self):
        return replace(self, **{k: getattr(self, k).clone() for k in ('features', 'mass', 'coords')})

    def metadata(self):
        return dict(first=self.first, last=self.last, begin=self.begin, end=self.end,
            center=self.center, groups=self.groups, depth=self.depth, tokens=len(self.features),
            original_pairs=self.last-self.first+1, mass=float(self.mass.sum()),
            surviving_source_pair_ids=sorted(set(self.coords[:, 0].tolist())),
            tensor_bytes=sum(t.numel()*t.element_size() for t in (self.features, self.mass, self.coords)),
            exact_pair=self.first == self.last and len(self.features) == math.prod(self.grid)
                       and bool(torch.all(self.mass == 1)),
            span_is_provenance_not_full_detail=True, grid=list(self.grid))


def validate(n):
    if not isinstance(n, Node) or n.grid not in tuple(tuple(geometry(r)['grid']) for r in PRESETS):
        raise ValueError('registered grid-aware node required')
    x, m, c = n.features, n.mass, n.coords
    if any(t.device.type != 'cpu' or t.requires_grad for t in (x, m, c)):
        raise ValueError('owned CPU inference tensors required')
    if (x.dtype != torch.float32 or x.ndim != 2 or not 1 <= len(x) <= math.prod(n.grid)
            or not 1 <= x.shape[1] <= 2048 or m.dtype != torch.float32 or m.shape != (len(x), 1)
            or c.dtype != torch.long or c.shape != (len(x), 3)
            or not torch.isfinite(x).all() or not torch.isfinite(m).all() or (m <= 0).any()
            or len(torch.unique(c, dim=0)) != len(c)):
        raise ValueError('invalid features/mass/coordinates')
    if (type(n.first) is not int or type(n.last) is not int or not 0 <= n.first <= n.last <= 2**31-1
            or (c[:, 0] < n.first).any() or (c[:, 0] > n.last).any() or (c[:, 1:] < 0).any()
            or (c[:, 1] >= n.grid[0]).any() or (c[:, 2] >= n.grid[1]).any()
            or not 1 <= n.groups <= n.last-n.first+1 or n.depth < 0
            or any(not math.isfinite(t) or t < 0 for t in (n.begin, n.end, n.center))
            or not n.begin <= n.center <= n.end):
        raise ValueError('invalid provenance')


def raw(features, index, begin, end, resolution):
    grid = tuple(geometry(resolution)['grid'])
    if len(features) != math.prod(grid) or not begin < end:
        raise ValueError('complete real chronological pair required')
    xy = torch.tensor(list(itertools.product(range(grid[0]), range(grid[1]))))
    n = Node(features.detach().to(device='cpu', dtype=torch.float32).clone(),
             torch.ones((len(features), 1)), torch.cat((torch.full((len(features), 1), index), xy), 1),
             index, index, float(begin), float(end), (begin+end)/2, grid)
    validate(n)
    return n


@torch.no_grad()
def merge(left, right, within=False):
    validate(left); validate(right)
    if (left.grid != right.grid or left.last+1 != right.first or left.end >= right.begin
            or left.features.shape[1] != right.features.shape[1]):
        raise ValueError('adjacent same-grid nodes required')
    x, mass, coords = [torch.cat((getattr(left, k), getattr(right, k))) for k in ('features', 'mass', 'coords')]
    values = numeric().forest.metric(x)
    scores = values[::2] @ values[1::2].T
    best, destination = scores.max(1)
    remove = len(x)//2
    order = torch.argsort(best, descending=True, stable=True)
    chosen, unmerged = order[:remove], order[remove:]
    weighted = (x[1::2]*mass[1::2]).scatter_add(0,
        destination[chosen, None].expand(-1, x.shape[1]), x[::2][chosen]*mass[::2][chosen])
    weights = mass[1::2].scatter_add(0, destination[chosen, None], mass[::2][chosen])
    output = torch.cat((x[::2][unmerged], weighted/weights))
    output_mass = torch.cat((mass[::2][unmerged], weights))
    output_coords = torch.cat((coords[::2][unmerged], coords[1::2]))
    h, w = left.grid
    permutation = torch.argsort(output_coords[:, 0]*(h*w)+output_coords[:, 1]*w+output_coords[:, 2], stable=True)
    nl, nr = left.last-left.first+1, right.last-right.first+1
    parent = Node(output[permutation], output_mass[permutation], output_coords[permutation],
        left.first, right.last, left.begin, right.end, (left.center*nl+right.center*nr)/(nl+nr),
        left.grid, max(left.groups, right.groups) if within else left.groups+right.groups,
        max(left.depth, right.depth)+1)
    validate(parent)
    return parent


class Bank:
    def __init__(self, arm, resolution):
        if arm not in ('LS', 'A06'):
            raise ValueError('LS/A06 only')
        self.arm, self.resolution = arm, resolution
        self.geometry = geometry(resolution)
        self.legacy = numeric()
        self.reset()

    def reset(self):
        self.recent, self.history, self.roots = deque(), [], []
        self.open = self.previous = None
        self.last = -1
        self.last_end = self.width = None
        self.event = -1
        self.within = self.cross = self.selection_count = 0
        self.valid = True

    def nodes(self):
        if self.arm == 'LS':
            return [n for n, _, _ in (*self.history, *self.recent)]
        return [*self.roots, *([] if self.open is None else [self.open]), *(n for n, _ in self.recent)]

    def digest(self):
        h = hashlib.sha256(json.dumps(self.controls(), sort_keys=True).encode())
        for n in self.nodes():
            h.update(json.dumps(n.metadata(), sort_keys=True).encode())
            for t in (n.features, n.mass, n.coords): h.update(t.numpy().tobytes())
        if self.previous is not None: h.update(self.previous.numpy().tobytes())
        return h.hexdigest()

    def controls(self):
        return dict(arm=self.arm, resolution=self.resolution, last=self.last, last_end=self.last_end,
            width=self.width, event=self.event, within=self.within, cross=self.cross,
            selection_count=self.selection_count, valid=self.valid,
            history_events=[[a, b] for _, a, b in self.history],
            recent_flags=[list(e[1:]) for e in self.recent])

    def status(self):
        nodes = self.nodes()
        return dict(tokens=sum(len(n.features) for n in nodes), nodes=[n.metadata() for n in nodes],
            resident_bytes=sum(n.metadata()['tensor_bytes'] for n in nodes)
                + (0 if self.previous is None else self.previous.numel()*self.previous.element_size()),
            state_digest=self.digest(), full_history_features_retained=False,
            language_cache_retained=False, **self.controls())

    @torch.no_grad()
    def append(self, n):
        validate(n)
        if (not self.valid or n.grid != tuple(self.geometry['grid']) or not n.metadata()['exact_pair']
                or n.first != self.last+1 or (self.last_end is not None and n.begin <= self.last_end)
                or (self.width is not None and n.features.shape[1] != self.width)):
            raise ValueError('sequential complete same-grid pair required')
        desc = n.features.mean(0)
        starts = self.previous is None or self.legacy.distance(self.previous, desc) > THRESHOLD
        p = self.geometry['pair_tokens']
        try:
            if self.arm == 'LS':
                event = 0 if self.last == -1 else self.event+int(starts)
                if len(self.recent) == 4:
                    old = self.recent.popleft()
                    self.history.append(old)
                    while sum(len(e[0].features) for e in self.history) > 4*p:
                        costs = self.legacy.forest.penalties([e[0] for e in self.history], old[0].end)
                        i = min(range(len(costs)), key=lambda j: (costs[j]['total'], j))
                        a, b = self.history[i:i+2]
                        within = a[1] == a[2] == b[1] == b[2]
                        self.history[i:i+2] = [(merge(a[0], b[0], within), a[1], b[2])]
                        self.within += int(within); self.cross += int(not within)
                self.recent.append((n.copy(), event, event)); self.event = event
            else:
                if len(self.recent) == 4:
                    old, boundary = self.recent.popleft()
                    if boundary and self.open is not None:
                        self.roots.append(self.open); self.open = None
                    if self.open is None: self.open = old
                    else:
                        self.open = merge(self.open, old, True); self.within += 1
                    while sum(len(v.features) for v in self.roots) > 3*p:
                        costs = self.legacy.forest.penalties(self.roots, old.end)
                        i = min(range(len(costs)), key=lambda j:
                            (1+max(self.roots[j].depth, self.roots[j+1].depth), costs[j]['total'], j))
                        self.roots[i:i+2] = [merge(self.roots[i], self.roots[i+1])]
                        self.cross += 1; self.selection_count += 1
                self.recent.append((n.copy(), starts))
            self.previous = desc.clone()
            self.last, self.last_end, self.width = n.last, n.end, n.features.shape[1]
            state = self.status()
            if (state['tokens'] > 8*p or len(self.nodes()) > 8
                    or len(json.dumps(state).encode()) > 32768
                    or state['resident_bytes'] > 8*p*(4*self.width+28)+4*self.width):
                raise RuntimeError('bounded storage exceeded')
        except BaseException:
            self.valid = False
            raise


@torch.inference_mode()
def context(model, builder, nodes, question, *, limits, max_new_tokens=8):
    """Frozen readout layout; only grid span and validator are generalized."""
    if not nodes or len(nodes) > limits.nodes or sum(len(n.features) for n in nodes) > limits.visual_tokens:
        raise ValueError('declared memory budget exceeded')
    if len({n.grid for n in nodes}) != 1 or any(a.last >= b.first or a.end >= b.begin for a,b in zip(nodes,nodes[1:])):
        raise ValueError('same-grid chronological memory required')
    offset = builder.prefix_ids.shape[1]
    all_ids, positions = [builder.prefix_ids], [torch.arange(offset).expand(3, -1)]
    for n in nodes:
        validate(n)
        if n.features.shape[1] != model.get_input_embeddings().embedding_dim:
            raise ValueError('model feature width mismatch')
        label = f'<{n.center:.1f} seconds>' if n.metadata()['exact_pair'] else f'<{n.begin:.1f}-{n.end:.1f} seconds, compressed memory>'
        ids = builder.tokenize(label+builder.processor.vision_start_token+builder.processor.video_token*len(n.features)+builder.processor.vision_end_token)
        coords = []
        for modality, group in itertools.groupby((ids[0] == builder.processor.video_token_id).tolist()):
            size = len(list(group))
            if modality:
                if size != len(n.features): raise ValueError('placeholder mismatch')
                coords.extend((offset+int(t)-n.first, offset+int(y), offset+int(x)) for t,y,x in n.coords)
                offset += max(*n.grid, n.last-n.first+1)
            else:
                coords.extend((offset+i,)*3 for i in range(size)); offset += size
        all_ids.append(ids); positions.append(torch.tensor(coords).T)
    suffix = builder.question_ids(question)
    if suffix.shape[1] > limits.suffix_tokens: raise ValueError('full suffix exceeds bound')
    all_ids.append(suffix); positions.append(torch.arange(offset,offset+suffix.shape[1]).expand(3,-1))
    ids = torch.cat(all_ids, 1)
    if ids.shape[1]+max_new_tokens > limits.context_tokens: raise ValueError('context exceeded; no truncation')
    four = torch.cat((torch.arange(ids.shape[1])[None,:], torch.cat(positions,1)),0)[:,None,:]
    device = model.get_input_embeddings().weight.device
    emb = model.get_input_embeddings()(ids.to(device))
    visual = torch.cat([n.features for n in nodes]).to(emb)
    mask = (ids.to(device) == builder.processor.video_token_id).unsqueeze(-1).expand_as(emb)
    if int(mask.sum()) != visual.numel(): raise ValueError('visual insertion mismatch')
    return emb.masked_scatter(mask,visual), four.to(device), ids, dict(
        input_tokens=ids.shape[1], visual_tokens=len(visual), nodes=[n.metadata() for n in nodes])


def answer(env, model, builder, bank, prompt):
    from ..sember_readout_diagnostic.inference import isolated, tensor_sha
    before = bank.digest()
    nodes = [n.copy() for n in bank.nodes()]
    emb, pos, ids, meta = context(model, builder, nodes, prompt,
        limits=env['Limits'](8192, bank.geometry['total_tokens'], 8), max_new_tokens=8)
    def run():
        session = env['HybridSequence'](model, max_tokens=8192, session_id='resolution-memory')
        try:
            for i in range(0, emb.shape[1], 512):
                end = min(i+512, emb.shape[1])
                logits = session.append(inputs_embeds=emb[:,i:end], position_ids=pos[:,:,i:end],
                    logits_mode='last' if end == emb.shape[1] else 'none')
            if not torch.isfinite(logits).all(): raise FloatingPointError('memory logits')
            eos = model.generation_config.eos_token_id
            stops = set(eos if isinstance(eos,list) else [eos])
            stops.add(builder.processor.tokenizer.eos_token_id)
            generated, reason = env['greedy_decode'](session, logits, 8, stops)
            return dict(text=builder.processor.tokenizer.decode(generated,skip_special_tokens=True),
                generated_ids=generated, stop_reason=reason, input_ids_sha256=tensor_sha(ids),
                position_ids_sha256=tensor_sha(pos), input_tokens=meta['input_tokens'],
                visual_tokens=meta['visual_tokens'], node_digest=before, node_metadata=meta['nodes'])
        finally:
            session.cache.reset()
    try:
        return isolated(env,model,run)
    finally:
        if bank.digest() != before: raise RuntimeError('question changed bank')
