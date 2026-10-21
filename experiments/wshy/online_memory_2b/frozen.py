"""Read-only adapters to the delivered algorithms; no historic path rewriting."""
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
PACKAGE = REPO / 'experiments/36tar/memory-eval-2b'
NATURAL = PACKAGE / 'bundle/experiments/2b-natural-eval-v1'
SHORT = PACKAGE / 'bundle/experiments/2b-short-event-v1'
ACTIVE = REPO / 'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
THRESHOLD = 0.047536224126815796
ARMS = ('M0', 'M2', 'reservoir4', 'text_only')


def module(name, folder):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    value = importlib.import_module(name)
    if Path(value.__file__).resolve().parent != folder.resolve():
        raise ImportError('module collision: ' + name)
    return value


def policies():
    for name in ('features', 'forest', 'legacy_sources', 'memory_policy', 'memory384'):
        module(name, NATURAL)
    event_path = NATURAL / 'diag/event_memory_v1/memory.py'
    spec = importlib.util.spec_from_file_location('_online2b_frozen_distance', event_path)
    event = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = event
    spec.loader.exec_module(event)
    return SimpleNamespace(features=module('features', NATURAL), forest=module('forest', NATURAL),
                           fifo=module('memory_policy', NATURAL),
                           forest384=module('memory384', NATURAL),
                           reservoir=module('policies', SHORT), distance=event.distance)


def runtime():
    legacy = policies()
    sys.path.insert(0, str(ACTIVE))
    import torch
    import numpy as np
    import av
    import psutil
    from transformers import AutoProcessor
    from qwen35_local import Qwen3_5ForConditionalGeneration, Qwen3_5Config
    actual = Path(sys.modules[Qwen3_5ForConditionalGeneration.__module__].__file__).resolve()
    if actual.parent != ACTIVE / 'qwen35_local':
        raise ImportError('active model implementation was not selected')
    from streaming.online import PairInputBuilder, fp32_reference, greedy_decode
    from streaming.hybrid import HybridSequence
    from streaming.window import FramePacket
    readout = module('readout', NATURAL)
    pilot = module('capability_pilot', NATURAL)
    context, Limits = readout.context, readout.Limits
    canvas_bytes = module('audit_sampling', NATURAL).canvas_bytes
    prompt = module('score_memory_eval', NATURAL).prompt
    tiny_config = module('first_2b_baseline', NATURAL).tiny_config
    answer = pilot.answer
    return locals()
