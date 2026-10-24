"""Bank-only causal encoding and isolated token-streaming QA; no main KV."""
import time
import hashlib
import torch
from common import ARMS
from banks import Banks
from bank_readout import context, Limits
from observer import emit_decode, cache_bytes
from online_bank import nodes_digest

LIMITS = Limits(2048, 384, 16, 128)


class Session:
    def __init__(self, env, model, builder):
        self.env, self.model, self.builder = env, model, builder
        self.bank = Banks()
        self.pairs = self.vision_encodings = self.queries = 0
        self.last_frame_id = self.last_pts = None
        self.tail_snapshot_only = self.invalid = False
        self.timing = dict(preprocess_seconds=0.,vision_seconds=0.,selection_banks_seconds=0.,pair_total_seconds=0.)

    @torch.inference_mode()
    def _encode(self, first, second, *, padding=False):
        if self.invalid or self.tail_snapshot_only:
            raise RuntimeError('closed, failed or query-only tail session')
        if self.last_frame_id is not None and (first.frame_id <= self.last_frame_id or first.timestamp <= self.last_pts):
            raise ValueError('future new pair required')
        if not padding and (first.frame_id >= second.frame_id or first.timestamp >= second.timestamp):
            raise ValueError('two increasing real frames required')
        total_started=time.monotonic()
        inputs, _ = self.builder.pair(first, second, 0, 0, synthetic_padding=padding)
        device = self.model.get_input_embeddings().weight.device
        saved = self.model.model.rope_deltas
        if device.type=='cuda':torch.cuda.synchronize(device)
        vision_started=time.monotonic()
        try:
            with self.env['fp32_reference']():
                inputs = inputs.to(device)
                vision = self.model.model.get_video_features(inputs.pixel_values_videos, inputs.video_grid_thw, return_dict=True)
                features = torch.cat(vision.pooler_output, 0)
                assert features.shape == (48, self.model.get_input_embeddings().embedding_dim)
                if device.type=='cuda':torch.cuda.synchronize(device)
                vision_finished=time.monotonic()
                trace=self.bank.append(features, self.pairs, first.timestamp, second.timestamp)
                banks_finished=time.monotonic()
        except BaseException:
            self.invalid = True
            raise
        finally:
            self.model.model.rope_deltas = saved
        self.timing['preprocess_seconds']+=vision_started-total_started
        self.timing['vision_seconds']+=vision_finished-vision_started
        self.timing['selection_banks_seconds']+=banks_finished-vision_finished
        self.timing['pair_total_seconds']+=banks_finished-total_started
        self.pairs += 1
        self.vision_encodings += 1
        self.last_frame_id, self.last_pts = second.frame_id, second.timestamp
        self.tail_snapshot_only = padding
        return dict(**self.status(),selection_trace=trace)

    def append_pair(self, first, second):
        return self._encode(first, second)

    def snapshot_tail(self, packet):
        return self._encode(packet, packet, padding=True)

    def status(self):
        return dict(pairs=self.pairs, vision_encodings=self.vision_encodings, last_pts=self.last_pts,
            last_frame_id=self.last_frame_id, invalid=self.invalid, tail_snapshot_only=self.tail_snapshot_only,
            banks=self.bank.status(), no_main_language_cache=True)

    @torch.inference_mode()
    def ask_arm(self, arm, question, on_token):
        if arm not in ARMS or self.invalid or not self.pairs or not callable(on_token):
            raise ValueError('valid observed evidence and explicit arm/callback required')
        before, digest = self.status(), self.bank.fingerprint()
        selector_before=self.bank.selector_digest()
        saved = self.model.model.rope_deltas
        branch = None
        device = self.model.get_input_embeddings().weight.device
        sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
        sync(); started = time.monotonic()
        try:
            nodes = self.bank.export(arm)
            emb, pos, ids, meta = context(self.model, self.builder, nodes, question,
                limits=LIMITS, max_new_tokens=8)
            sync(); prepared = time.monotonic()
            branch = self.env['HybridSequence'](self.model, max_tokens=2048, session_id='readout-compression-qa')
            for i in range(0, emb.shape[1], 512):
                end = min(i+512, emb.shape[1])
                logits = branch.append(inputs_embeds=emb[:, i:end], position_ids=pos[:, :, i:end],
                    logits_mode='last' if end == emb.shape[1] else 'none')
            sync(); prefilled = time.monotonic()
            cfg = self.model.generation_config.eos_token_id
            stops = set(cfg if isinstance(cfg, list) else [] if cfg is None else [cfg])
            stops.add(self.builder.processor.tokenizer.eos_token_id)
            first_token=[]
            def emit(item):
                if not first_token:first_token.append(time.monotonic()-started)
                on_token(item)
            result = emit_decode(branch, logits, self.builder.processor, 8, stops, emit)
            sync(); finished = time.monotonic()
            payload = cache_bytes(branch)
        finally:
            if branch is not None:
                branch.cache.reset()
            self.model.model.rope_deltas = saved
        assert self.status() == before and self.bank.fingerprint() == digest
        assert self.bank.selector_digest()==selector_before
        return dict(arm=arm, **result, input_tokens=meta['input_tokens'], visual_tokens=meta['visual_tokens'],
            first_token_seconds=first_token[0], storage_node_digest=digest[arm.split('_')[0]], nodes=meta['nodes'], context_build_seconds=prepared-started, prefill_seconds=prefilled-prepared,
            decode_seconds=finished-prefilled, QA_total_seconds=finished-started, QA_cache_bytes=payload,
            input_ids_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(), node_digest=nodes_digest(nodes),
            banks_unchanged=True, branch_released=True, no_main_language_cache=True)

    def ask(self, question, *, read_from, on_token):
        if read_from != 'readout_banks':
            raise ValueError('six registered readouts required')
        offset = (self.pairs+self.queries) % len(ARMS)
        order = ARMS[offset:] + ARMS[:offset]
        result = {a: self.ask_arm(a, question, lambda t, arm=a: on_token(dict(arm=arm, **t))) for a in order}
        self.queries += 1
        return dict(arms=result, order=list(order), all_QA_branches_released=True)

    def close(self):
        self.bank.clear()
        self.invalid = True
