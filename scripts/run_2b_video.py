"""Pinned 2B FP32 file-video entrypoint; same arguments for preflight and GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "experiments/36tar/streaming-learning"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["preflight", "gpu"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument("--question", default="Describe the visible subject and action in the video.")
    args = parser.parse_args()
    args.model_dir = args.model_dir.resolve()
    manifest = json.loads((args.model_dir / "DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))
    if (manifest.get("model"),manifest.get("revision")) != ("Qwen/Qwen3.5-2B","15852e8c16360a2fea060d615a32b45270f8a8fc"):
        parser.error("Download/verify the pinned 2B model first")
    if args.stage == "gpu" and args.preflight is None:
        parser.error("GPU requires a successful --preflight directory")
    command = [sys.executable, str(PROJECT / "scripts/run_video_demo.py"), "--stage", args.stage,
               "--model-dir", str(args.model_dir), "--video", str(args.video.resolve()), "--out", str(args.out.resolve()),
               "--fps", str(args.fps), "--question", args.question, "--duration", "60", "--max-frames", "8",
               "--answer-tokens", "16", "--max-tokens", "288", "--first-qa-frames", "2",
               "--wall-seconds", "600", "--min-free-gpu-gib", "12", "--min-free-ram-gib", "12"]
    if args.preflight:
        command.extend(["--preflight", str(args.preflight.resolve())])
    env = dict(os.environ, QWEN35_MODEL_DIR=str(args.model_dir), PYTHONIOENCODING="utf-8",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
