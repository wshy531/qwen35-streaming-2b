"""Run inherited CPU contracts in isolated processes; never load pretrained weights."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiments/36tar/streaming-learning"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Local processor assets; processor-only download suffices")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    if not (model_dir / "tokenizer.json").is_file():
        parser.error("Prepare local processor assets first")
    env = dict(os.environ, QWEN35_MODEL_DIR=str(model_dir), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    for cwd, directory in [(ROOT, "tests"), (PROJECT / "code/qwen35-streaming", "tests"),
                           (PROJECT / "code/qwen35-streaming-s4", "tests"), (PROJECT, "tools/tests")]:
        print(f"CPU suite: {cwd / directory}", flush=True)
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", directory, "-v"], cwd=cwd, env=env)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
