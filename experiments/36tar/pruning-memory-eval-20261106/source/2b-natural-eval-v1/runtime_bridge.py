"""Select the actual 2B implementation explicitly, with no 0.8B profile alias."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPO=ROOT/'repo'
ACTIVE=REPO/'experiments/36tar/streaming-learning/code/qwen35-streaming-s4'
MODEL=ROOT/'models/Qwen3.5-2B'
def runtime(stage):
    sys.path.insert(0,str(REPO if stage=='native' else ACTIVE))
    from qwen35_local import Qwen3_5ForConditionalGeneration,Qwen3_5Config
    actual_source=Path(sys.modules[Qwen3_5ForConditionalGeneration.__module__].__file__).resolve()
    expected=(REPO if stage=='native' else ACTIVE)/'qwen35_local'
    if actual_source.parent!=expected.resolve():raise ValueError('wrong isolated implementation')
    sys.path.append(str(ACTIVE))
    import torch,numpy as np,av,psutil
    from transformers import AutoProcessor
    from streaming.online import PairInputBuilder,fp32_reference,greedy_decode
    from streaming.hybrid import HybridSequence
    from streaming.window import FramePacket
    from audit_sampling import canvas_bytes
    from score_memory_eval import prompt
    from readout import context,Limits
    from first_2b_baseline import complete,tiny_config
    from legacy_sources import load
    import capability_pilot,fixed_stage2
    capability_pilot.ROOT=ROOT
    fixed_stage2.ROOT=ROOT;fixed_stage2.REPO=REPO;fixed_stage2.ACTIVE=ACTIVE;fixed_stage2.DIAG=HERE/'diag'
    return locals()
