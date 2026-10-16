"""Download pinned 2B assets, or verify a previously transferred offline directory."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "experiments/36tar/streaming-learning/code/qwen35-streaming-s4/scripts/model_assets.py"), run_name="__main__")
