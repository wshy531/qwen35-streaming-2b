"""Reuse frozen causal ingestion; replace only the isolated question readout."""
import hashlib
import importlib.util
import time
import torch
from common import ROOT, ARMS
from bank_readout import context, Limits
from online_bank import nodes_digest
from observer import emit_decode, cache_bytes
from probe import score_visual, select_half

_spec = importlib.util.spec_from_file_location('_frozen_E2_session',
    ROOT/'experiments/2b-readout-compression-v1/session.py')
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
LIMITS = Limits(2048, 384, 16, 128)


class Session(_legacy.Session):
    @torch.inference_mode()
    def ask_arm(self, arm, question, on_token):
        if arm not in ARMS or self.invalid or not self.pairs or not callable(on_token):
            raise ValueError('observed evidence and explicit arm/callback required')
        before, digest = self.status(), self.bank.fingerprint()
        selector_before = self.bank.selector_digest()
        saved = self.model.model.rope_deltas
        branch = None
        device = self.model.get_input_embeddings().weight.device
        sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
        sync(); started = time.monotonic()
        probe = None
        selected = None
        try:
            if arm == 'LS_probe24':
                original = self.bank.full()
                emb, pos, ids, meta = context(self.model, self.builder, original, question,
                    limits=LIMITS, max_new_tokens=8)
                scores, probe = score_visual(self.model, self.builder, question, emb, pos, ids)
                nodes, selected = select_half(original, scores)
                # No hidden state or partial cache is reused for the second pass.
                del emb, pos, ids, meta, scores, original
            else:
                nodes = self.bank.export(arm)
            emb, pos, ids, meta = context(self.model, self.builder, nodes, question,
                limits=LIMITS, max_new_tokens=8)
            sync(); prepared = time.monotonic()
            branch = self.env['HybridSequence'](self.model, max_tokens=2048, session_id='qprobe-qa')
            for i in range(0, emb.shape[1], 512):
                end = min(i+512, emb.shape[1])
                logits = branch.append(inputs_embeds=emb[:, i:end], position_ids=pos[:, :, i:end],
                    logits_mode='last' if end == emb.shape[1] else 'none')
            sync(); prefilled = time.monotonic()
            cfg = self.model.generation_config.eos_token_id
            stops = set(cfg if isinstance(cfg, list) else [] if cfg is None else [cfg])
            stops.add(self.builder.processor.tokenizer.eos_token_id)
            first = []
            def emit(item):
                if not first:
                    first.append(time.monotonic()-started)
                on_token(item)
            result = emit_decode(branch, logits, self.builder.processor, 8, stops, emit)
            sync(); finished = time.monotonic()
            payload = cache_bytes(branch)
        finally:
            if branch is not None:
                branch.cache.reset()
            self.model.model.rope_deltas = saved
        assert self.status() == before and self.bank.fingerprint() == digest
        assert self.bank.selector_digest() == selector_before
        assert not self.model.model.language_model._qprobe_active if hasattr(self.model.model.language_model, '_qprobe_active') else True
        return dict(arm=arm, **result, input_tokens=meta['input_tokens'], visual_tokens=meta['visual_tokens'],
            first_token_seconds=first[0], storage_node_digest=digest['LS'], nodes=meta['nodes'],
            context_build_and_probe_seconds=prepared-started, prefill_seconds=prefilled-prepared,
            decode_seconds=finished-prefilled, QA_total_seconds=finished-started,
            QA_includes_full_probe_and_rebuild=True, QA_cache_bytes=payload, probe=probe,
            selected_indices=selected, input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
            node_digest=nodes_digest(nodes), banks_unchanged=True, branch_released=True,
            no_main_language_cache=True)
