"""CPU contracts for storage, unchanged LS, readout and causal continuation."""
import importlib.util
import os
import sys
import time
import types
from pathlib import Path
import torch
from common import ROOT, OUT, ARMS, read, save, sha, runtime
from shared import SharedHistory384
from banks import Banks
from features import raw_block
from legacy_sources import load
from online_bank import nodes_digest


def reference():
    package = types.ModuleType('ls_reference_4212'); package.__path__ = []
    frozen = types.ModuleType('ls_reference_4212.frozen'); frozen.policies = load
    sys.modules[package.__name__] = package; sys.modules[frozen.__name__] = frozen
    spec = importlib.util.spec_from_file_location('ls_reference_4212.shared', Path(__file__).parent/'reference_shared_4212.py')
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SharedHistory384()


def owned_tensor_bytes(obj):
    """Walk persistent object storage, deduplicating aliases; modules are not payload."""
    seen, storage = set(), {}
    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            assert item.device.type == 'cpu' and not item.requires_grad
            s = item.untyped_storage(); storage[s.data_ptr()] = s.nbytes()
        elif isinstance(item, (list, tuple)) or type(item).__name__ == 'deque':
            for child in item: visit(child)
        elif isinstance(item, dict):
            for child in item.values(): visit(child)
        elif not isinstance(item, (types.ModuleType, type)) and hasattr(item, '__dict__'):
            for key, child in vars(item).items():
                if key != 'legacy': visit(child)
    visit(obj)
    return sum(storage.values())


def contracts():
    cases, difference = 0, 0
    generator = torch.Generator().manual_seed(20261021)
    from memory384 import Forest384
    from policy import DepthForest384
    for scenario in ('one_event', 'each_pair', 'mixed'):
        expected = reference(); compat = SharedHistory384(); depth = SharedHistory384(depth_first=True)
        m2, a06_compat = Forest384(), DepthForest384(original_selector=True)
        for i in range(96):
            flag = i == 0 or scenario == 'each_pair' or (scenario == 'mixed' and i % 5 == 0)
            x = torch.randn(48, 32, generator=generator)
            block = raw_block(x, i, float(2*i), float(2*i+1))
            for bank in (expected, compat, depth, m2, a06_compat):
                bank.append(block, starts_event=flag)
            assert expected.status() == compat.status() and expected.controls() == compat.controls()
            assert nodes_digest(expected.export()) == nodes_digest(compat.export())
            assert m2.status() == a06_compat.status() and nodes_digest(m2.export()) == nodes_digest(a06_compat.export())
            for bank in (compat, depth):
                ns = bank.export(); state = bank.status()
                assert state['tokens'] == min(i+1, 8)*48 and state['history_tokens'] <= 192
                assert abs(sum(float(n.mass.sum()) for n in ns) - 48*(i+1)) < .01
                assert ns[0].first == 0 and ns[-1].last == i
                assert all(a.last+1 == b.first for a, b in zip(ns, ns[1:]))
                assert owned_tensor_bytes(bank) == state['resident_tensor_bytes']
                before = nodes_digest(bank.export()); ns[0].features.zero_()
                assert nodes_digest(bank.export()) == before
                if i < 8:
                    assert all(n.metadata()['exact_pair'] for n in bank.export())
                if i == 8:
                    assert bank.within_merges + bank.long_merges == 1
            difference += nodes_digest(compat.export()) != nodes_digest(depth.export())
            cases += 1
        before = nodes_digest(depth.export())
        try:
            depth.append(block, starts_event=True)
        except ValueError:
            pass
        else:
            raise AssertionError('duplicate input accepted')
        assert nodes_digest(depth.export()) == before
    assert difference > 0, 'candidate must actually change overflow choices'
    # Deliberately unequal synthetic costs: depth has absolute priority, original
    # mode uses cost; verify tie handling without relying on an observed score.
    from shared import EventNode
    from dataclasses import replace
    node = load().forest.raw(raw_block(torch.ones(48, 32), 0, 0., 1.))
    probe = SharedHistory384(depth_first=True)
    probe.history = [EventNode(replace(node, depth=d), 0, 0) for d in (9, 9, 0, 0)]
    costs = [dict(total=v) for v in (0., 0., 999.)]
    assert probe.select(costs) == 2
    probe.depth_first = False; assert probe.select(costs) == 0
    real_width = Banks()
    for i in range(24):
        real_width.append(torch.randn(48,2048,generator=generator),i,float(2*i),float(2*i+1))
    state = real_width.status()
    actual = owned_tensor_bytes(real_width)
    declared = sum(s['resident_tensor_bytes'] for s in state['banks'].values())+state['shared_descriptor_bytes']
    assert actual == declared and state['shared_descriptor_bytes'] == 8192
    assert all(s['resident_tensor_bytes'] <= 3156480 for s in state['banks'].values())
    real_width.clear(); assert owned_tensor_bytes(real_width) == 0
    return dict(unchanged_LS_steps=cases, unchanged_M2_A06_compat_steps=cases,
        differing_candidate_steps=difference, selector_oracle=True,
        first8_raw_9th_merge=True, mass_provenance_export_isolation=True,
        persistent_storage_walk=True, real2048_width_storage_bytes=actual,
        descriptor2048_bytes=8192, duplicate_rejected_without_mutation=True)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1); torch.manual_seed(123)
    assert not torch.cuda.is_available()
    started = time.monotonic()
    proof = contracts()
    env = runtime()
    build = env['PairInputBuilder'](env['AutoProcessor'].from_pretrained(
        ROOT/'models/Qwen3.5-2B', local_files_only=True), canvas=(192,256), resize_mode='letterbox')
    model = env['Qwen3_5ForConditionalGeneration'](env['Qwen3_5Config'](
        **env['tiny_config'](build.processor), attn_implementation='sdpa')).float().eval()
    from readout import context, Limits
    from score_memory_eval import prompt
    rows = read(OUT/'inputs/questions.json'); groups = read(OUT/'inputs/groups.json')
    layouts = []
    with torch.inference_mode(), env['fp32_reference']():
        for group in groups:
            banks = Banks()
            qs = sorted([q for q in rows if q['group'] == group['id']], key=lambda q:(q['pairs'],q['id']))
            for q in qs:
                while banks.pairs < q['pairs']:
                    i = banks.pairs
                    first, second = group['frames'][2*i], group['frames'][min(2*i+1,len(group['frames'])-1)]
                    banks.append(torch.randn(48,32), i, first['pts_s'], second['pts_s'])
                assert banks.last_pts <= q['query_time_s']+1e-9
                before = banks.fingerprint()
                for arm in ARMS:
                    emb, pos, ids, meta = context(model, build, banks.export(arm), prompt(q),
                        limits=Limits(2048,384,8,128), max_new_tokens=8)
                    assert pos.shape == (4,1,ids.shape[1]) and meta['visual_tokens'] <= 384
                    layouts.append(dict(id=q['id'], arm=arm, input_tokens=ids.shape[1],
                        suffix_tokens=meta['suffix_tokens'], visual_tokens=meta['visual_tokens']))
                    del emb, pos, ids
                assert banks.fingerprint() == before
            declared = sum(s['resident_tensor_bytes'] for s in banks.status()['banks'].values()) + banks.status()['shared_descriptor_bytes']
            assert owned_tensor_bytes(banks) == declared
            banks.clear(); assert owned_tensor_bytes(banks) == 0
            assert time.monotonic()-started < 300
    assert len(layouts) == 1818
    from session import Session
    from replay import VideoSource
    group = next(g for g in groups if g['cohort'] == 'stream34')
    key = 'cluster_media' if (ROOT/'repo').exists() else 'local_media'
    source = VideoSource(ROOT/group[key], dict(frames=group['frames'],media_sha256=group['sha256']), env['FramePacket'])
    stream = iter(source.frames()); packets = []; pending = []
    for _ in range(20):
        frame = next(stream); pending.append(frame.frame_id)
        if len(pending) == 2:
            packets.extend(source.take_pair(pending)); pending = []
    stream.close()
    session = Session(env,model,build)
    original = model.model.get_video_features; calls = []
    def counted(*args, **kwargs):
        calls.append(1); return original(*args, **kwargs)
    model.model.get_video_features = counted
    qa = []
    try:
        with torch.inference_mode(), env['fp32_reference']():
            for i in range(0,16,2): session.append_pair(*packets[i:i+2])
            for phase in ('continued', 'odd_tail'):
                if phase == 'odd_tail':
                    session.append_pair(*packets[16:18])
                    assert session.pairs == 9
                    session.snapshot_tail(packets[18])
                before = session.bank.fingerprint()
                for arm in ARMS:
                    tokens = []; result = session.ask_arm(arm,'What is visible?',tokens.append)
                    assert result['generated_ids'] == [t['token_id'] for t in tokens]
                    assert session.bank.fingerprint() == before
                    qa.append(dict(arm=arm,phase=phase,generated_tokens=len(tokens)))
            try: session.append_pair(*packets[18:20])
            except RuntimeError: pass
            else: raise AssertionError('query-only tail continued')
    finally:
        model.model.get_video_features = original
        session.close()
    assert len(calls) == 10 and owned_tensor_bytes(session.bank) == 0
    receipt = dict(status='passed', contracts=proof, context_layouts=len(layouts), questions=303,
        actual_RGB_frames=20, shared_vision_encodings=10, tiny_QA=qa,
        no_persistent_main_or_full_history=True, new_real_pair_after_QA=True, odd_tail_continuation_rejected=True,
        pretrained_loads=0, GPU_submissions=0, seconds=time.monotonic()-started,
        max_input_tokens=max(l['input_tokens'] for l in layouts),
        observed_RSS_bytes=env['psutil'].Process().memory_info().rss,
        protocol_sha256=sha(OUT/'inputs/protocol.json'), layouts=layouts)
    save(OUT/'CPU_preflight.json',receipt)
    print({k:v for k,v in receipt.items() if k != 'layouts'})


if __name__ == '__main__':
    main()
