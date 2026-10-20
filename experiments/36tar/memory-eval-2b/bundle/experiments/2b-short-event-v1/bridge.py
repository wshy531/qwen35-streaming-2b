"""Read-only reuse of the frozen 2B stage; no duplicated model or source checkout."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'experiments/2b-natural-eval-v1'
sys.path.insert(0, str(BASE))

def runtime():
    from runtime_bridge import runtime as base_runtime
    return base_runtime('memory')

