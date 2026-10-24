"""Session-local, explicit Qwen3.5 pairwise visual compute reuse.

Adaptation of the inspected STC selective-key path, not its CUDA graph runner.
No pretrained weights, global monkey patches, or language-state changes here.
"""
from contextlib import ExitStack, contextmanager
import math
import types

import torch
import torch.nn.functional as F


class PairVisionCache:
    def __init__(self, vision, *, update_ratio=0.25, refresh_interval=2):
        if not 0 < update_ratio <= 1 or not isinstance(refresh_interval, int) or refresh_interval < 1:
            raise ValueError('invalid fixed cache configuration')
        self.vision = vision
        self.ratio = float(update_ratio)
        self.interval = refresh_interval
        self.active = False
        self._bypass = False
        self._restore = []
        self.reset()

    def reset(self):
        self.references = {}
        self.pair_index = 0
        self.signature = None
        self.selection = None
        self.last = None
        self.full_pairs = self.selective_pairs = 0
        self.reference_revision = 0

    def __enter__(self):
        if self.active or getattr(self.vision, '_pair_cache_owner', None) is not None:
            raise RuntimeError('vision already has a cache owner')
        if self.vision.config._attn_implementation != 'sdpa':
            raise ValueError('only the checked SDPA visual path is supported')
        self.active = True
        self.vision._pair_cache_owner = self
        self._original_vision = self.vision.forward
        self._original_blocks = [block.forward for block in self.vision.blocks]
        try:
            self._bind(self.vision, self._visual_forward)
            for index, block in enumerate(self.vision.blocks):
                self._bind(block, self._make_block_forward(index))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _bind(self, module, forward):
        self._restore.append((module, 'forward' in module.__dict__, module.__dict__.get('forward')))
        module.forward = types.MethodType(forward, module)

    def __exit__(self, *_):
        for module, existed, previous in reversed(self._restore):
            if existed:
                module.forward = previous
            else:
                delattr(module, 'forward')
        self._restore.clear()
        if getattr(self.vision, '_pair_cache_owner', None) is self:
            delattr(self.vision, '_pair_cache_owner')
        self.active = False
        self.reset()

    def _visual_forward(self, module, hidden_states, grid_thw, **kwargs):
        if self._bypass:
            return self._original_vision(hidden_states, grid_thw=grid_thw, **kwargs)
        try:
            return self._visual_step(module, hidden_states, grid_thw, **kwargs)
        except BaseException:
            self.reset()
            raise

    @contextmanager
    def bypass(self):
        """Native baseline call; cannot advance or write candidate references."""
        if not self.active or self._bypass:
            raise RuntimeError('one active, nonnested baseline bypass required')
        self._bypass = True
        try:
            yield
        finally:
            self._bypass = False

    def fingerprint(self):
        """Cheap mutation audit; no transfer/hash of all GPU reference values."""
        return (self.pair_index, self.signature, self.reference_revision, tuple(
            (i, name, value.data_ptr(), tuple(value.shape))
            for i, layer in sorted(self.references.items()) for name, value in sorted(layer.items())))

    def _visual_step(self, module, hidden_states, grid_thw, **kwargs):
        if module.training or torch.is_grad_enabled():
            raise ValueError('cache requires evaluation and no gradients')
        if kwargs.pop('return_dict', True) is not True:
            raise ValueError('structured visual output required')
        if kwargs:
            raise ValueError('extra visual options are not supported in this fixed adapter')
        if tuple(grid_thw.shape) != (1, 3):
            raise ValueError('one complete temporal patch per call required')
        grid = tuple(int(x) for x in grid_thw[0].tolist())
        if grid[0] != 1 or min(grid[1:]) < 2 or any(x % 2 for x in grid[1:]):
            raise ValueError('one paired temporal patch with even spatial dimensions required')
        signature = (grid, hidden_states.dtype, hidden_states.device)
        reason = 'periodic' if self.pair_index % self.interval == 0 else None
        if not self.references:
            reason = 'cold'
        elif signature != self.signature:
            reason = 'grid_or_tensor_change'
        self._refresh = reason is not None
        self.signature = signature
        self.selection = None
        self._rows = []
        try:
            output = self._original_vision(hidden_states, grid_thw=grid_thw, return_dict=True)
            if len(self._rows) != len(module.blocks):
                raise AssertionError('incomplete visual traversal')
            n = grid[1] * grid[2]
            if output.pooler_output.shape[0] != n // 4:
                raise AssertionError('visual merger output changed')
            self.last = dict(pair_index=self.pair_index, full=self._refresh, reason=reason,
                             patch_tokens=n, projected_tokens=n // 4,
                             updated_tokens=n if self._refresh else int(self.selection.numel()),
                             layers=self._rows, reference_bytes=self.reference_bytes())
            self.pair_index += 1
            self.full_pairs += int(self._refresh)
            self.selective_pairs += int(not self._refresh)
            return output
        except BaseException:
            self.reset()
            raise

    def reference_bytes(self):
        return sum(t.numel() * t.element_size() for layer in self.references.values() for t in layer.values())

    def _store(self, index, name, value):
        entries = self.references.setdefault(index, {})
        old = entries.get(name)
        if old is None or old.shape != value.shape or old.dtype != value.dtype or old.device != value.device:
            old = torch.empty(value.shape, dtype=value.dtype, device=value.device)
            entries[name] = old
        old.copy_(value.detach())
        self.reference_revision += 1

    def _make_block_forward(self, index):
        def forward(block, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
            if self._bypass:
                return self._original_blocks[index](hidden_states, cu_seqlens=cu_seqlens,
                    rotary_pos_emb=rotary_pos_emb, position_embeddings=position_embeddings, **kwargs)
            if position_embeddings is None or rotary_pos_emb is not None:
                raise ValueError('explicit visual position embeddings required')
            n, d = hidden_states.shape
            if cu_seqlens.tolist() != [0, n]:
                raise ValueError('no packed multi-video input in pair cache')
            if self._refresh:
                return self._full(index, block, hidden_states, cu_seqlens, position_embeddings, kwargs)
            return self._selective(index, block, hidden_states, position_embeddings)
        return forward

    def _full(self, index, block, hidden, cu_seqlens, positions, kwargs):
        n, d = hidden.shape
        def qkv_hook(_module, _args, output):
            packed = output.reshape(n, 3, d)
            self._store(index, 'key', packed[:, 1])
            self._store(index, 'value', packed[:, 2])
        def attn_hook(_module, _args, output):
            self._store(index, 'attention', output)
        def mlp_hook(_module, _args, output):
            self._store(index, 'mlp', output)
        with ExitStack() as stack:
            for module, callback in ((block.attn.qkv, qkv_hook), (block.attn, attn_hook), (block.mlp, mlp_hook)):
                handle = module.register_forward_hook(callback)
                stack.callback(handle.remove)
            output = self._original_blocks[index](hidden, cu_seqlens=cu_seqlens,
                position_embeddings=positions, **kwargs)
        self._rows.append(dict(key=n, query=n, value=n, attention_query=n, projection=n, mlp=n))
        return output

    @staticmethod
    def _rotate(x, cos, sin):
        # Same float32 rotation and half split as local Qwen visual RoPE.
        dtype = x.dtype
        xf = x.float()
        half = torch.cat((-xf[..., xf.shape[-1] // 2:], xf[..., :xf.shape[-1] // 2]), dim=-1)
        return (xf * cos.unsqueeze(1).float() + half * sin.unsqueeze(1).float()).to(dtype)

    def _selective(self, index, block, hidden, positions):
        n, d = hidden.shape
        ref = self.references[index]
        normalized = block.norm1(hidden)
        weight, bias = block.attn.qkv.weight, block.attn.qkv.bias
        key = F.linear(normalized, weight[d:2*d], None if bias is None else bias[d:2*d])
        if index == 0:
            similarity = F.cosine_similarity(key.float(), ref['key'].float(), dim=-1)
            k = max(1, min(n, math.floor(n * self.ratio)))
            self.selection = similarity.topk(k, largest=False).indices.sort().values
        if self.selection is None:
            raise AssertionError('first-layer shared selection missing')
        indices = self.selection
        selected = normalized.index_select(0, indices)
        query = F.linear(selected, weight[:d], None if bias is None else bias[:d])
        value = F.linear(selected, weight[2*d:], None if bias is None else bias[2*d:])
        h, hd = block.attn.num_heads, block.attn.head_dim
        cos, sin = positions
        query = self._rotate(query.reshape(-1, h, hd), cos.index_select(0, indices), sin.index_select(0, indices))
        key = self._rotate(key.reshape(n, h, hd), cos, sin)
        mixed_value = ref['value'].clone()
        mixed_value.index_copy_(0, indices, value)
        output = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0), key.transpose(0, 1).unsqueeze(0),
            mixed_value.reshape(n, h, hd).transpose(0, 1).unsqueeze(0),
            dropout_p=0.0, is_causal=False, scale=block.attn.scaling)
        output = output.squeeze(0).transpose(0, 1).reshape(indices.numel(), d).contiguous()
        selected_attention = block.attn.proj(output)
        attention = ref['attention'].clone()
        attention.index_copy_(0, indices, selected_attention)
        hidden = hidden + attention
        selected_mlp = block.mlp(block.norm2(hidden).index_select(0, indices))
        mlp = ref['mlp'].clone()
        mlp.index_copy_(0, indices, selected_mlp)
        count = int(indices.numel())
        self._rows.append(dict(key=n, query=count, value=count, attention_query=count, projection=count, mlp=count))
        return hidden + mlp
