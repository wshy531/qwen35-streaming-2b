"""Same uncompressed observation main; three isolated fresh QA paths."""
import time
import torch

from common import ARMS
from memory import Archive, context
from observer import Observer as BaseObserver, emit_decode


class CompressionObserver:
    def __init__(self, env, model, builder, *, main_tokens=2048):
        self.env, self.model, self.builder = env, model, builder
        self.core = BaseObserver(env, model, builder, policy=None, main_tokens=main_tokens)
        self.archive = Archive()
        self.core.bank = self.archive
        self.queries = 0
        self.tail_snapshot_only = False

    def append_pair(self, first, second):
        if self.tail_snapshot_only:
            raise RuntimeError('query-only padded tail cannot enter a continued observation')
        return self.core.append_pair(first, second)

    @torch.inference_mode()
    def snapshot_tail(self, packet):
        if self.tail_snapshot_only or packet.frame_id <= self.core.last_frame_id:
            raise ValueError('single final pending real frame required')
        inputs,_ = self.builder.pair(packet,packet,0,0,synthetic_padding=True)
        device = self.model.get_input_embeddings().weight.device
        with self.env['fp32_reference']():
            inputs=inputs.to(device)
            vision=self.model.model.get_video_features(inputs.pixel_values_videos,inputs.video_grid_thw,return_dict=True)
            features=torch.cat(vision.pooler_output,0)
            self.archive.append(features,self.core.pairs,packet.timestamp,packet.timestamp)
        self.tail_snapshot_only=True

    def status(self):
        return self.core.status()

    @torch.inference_mode()
    def ask_arm(self, arm, question, on_token):
        if arm not in ARMS or not callable(on_token):
            raise ValueError('fixed arm and token callback required')
        before = self.core.status()
        fingerprints = self.archive.fingerprint()
        saved = self.model.model.rope_deltas
        branch = None
        torchmod = self.env['torch']
        device = self.model.get_input_embeddings().weight.device
        if device.type == 'cuda':
            torchmod.cuda.synchronize(device)
        started = time.monotonic()
        try:
            with self.env['fp32_reference']():
                emb, pos, _, meta = context(self.model, self.builder, self.archive.nodes[arm], question)
                if device.type == 'cuda':
                    torchmod.cuda.synchronize(device)
                prepared = time.monotonic()
                branch = self.env['HybridSequence'](self.model, max_tokens=20000, session_id='compression-qa')
                for i in range(0, emb.shape[1], 512):
                    end = min(i + 512, emb.shape[1])
                    logits = branch.append(inputs_embeds=emb[:, i:end], position_ids=pos[:, :, i:end],
                        logits_mode='last' if end == emb.shape[1] else 'none')
                if device.type == 'cuda':
                    torchmod.cuda.synchronize(device)
                prefilled = time.monotonic()
                configured = self.model.generation_config.eos_token_id
                stops = set(configured if isinstance(configured, list) else [configured])
                stops.add(self.builder.processor.tokenizer.eos_token_id)
                result = emit_decode(branch, logits, self.builder.processor, 8, stops, on_token)
                if device.type == 'cuda':
                    torchmod.cuda.synchronize(device)
                finished = time.monotonic()
                cache_bytes = sum(t.numel() * t.element_size() for name in
                    ('key_cache', 'value_cache', 'conv_states', 'recurrent_states')
                    for t in getattr(branch.cache, name) if t is not None)
        finally:
            if branch is not None:
                branch.cache.reset()
            self.model.model.rope_deltas = saved
        assert self.core.status() == before and self.archive.fingerprint() == fingerprints
        return dict(**result, arm=arm, input_tokens=meta['input_tokens'],visual_tokens=meta['visual_tokens'],
            context_build_seconds=prepared-started,prefill_seconds=prefilled-prepared,
            decode_seconds=finished-prefilled,QA_total_seconds=finished-started,QA_cache_bytes=cache_bytes,
            archive_unchanged=True,main_unchanged=True,diagnostic_full_history_NOT_bounded_memory=True)

    def ask(self, question, *, read_from, on_token):
        if read_from != 'compression_five':
            raise ValueError('explicit three-arm regression required')
        # Rotate by current pair count to reduce a fixed first-arm ordering bias.
        offset = (self.core.pairs + self.queries) % len(ARMS)
        order = ARMS[offset:] + ARMS[:offset]
        answers = {}
        for arm in order:
            answers[arm] = self.ask_arm(arm, question, lambda t: on_token(dict(arm=arm, **t)))
        self.queries += 1
        return dict(arms=answers, order=list(order), all_QA_branches_released=True)

    def close(self):
        self.core.close()
        self.archive.clear()
