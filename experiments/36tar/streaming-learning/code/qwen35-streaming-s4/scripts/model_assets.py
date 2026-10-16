"""Pinned official assets, shared by download and offline verification. No GPU imports."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
REVISIONS = {MODEL: REVISION, "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17"}
PROCESSOR_FILES = ("chat_template.jinja", "config.json", "merges.txt", "preprocessor_config.json",
                   "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json")
WEIGHT_FILES = ("model.safetensors.index.json", "model.safetensors-00001-of-00001.safetensors")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_assets(model_dir):
    model_dir = Path(model_dir).resolve()
    manifest = json.loads((model_dir / "DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))
    if manifest.get("model") not in REVISIONS or manifest.get("revision") != REVISIONS[manifest["model"]]:
        raise ValueError("Unsupported model or unpinned revision")
    files = manifest.get("files", [])
    expected = set(PROCESSOR_FILES + WEIGHT_FILES)
    if len(files) != len(expected) or {item["name"] for item in files} != expected:
        raise ValueError("Manifest must contain each required asset exactly once")
    for item in files:
        path = model_dir / item["name"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"].lower():
            raise ValueError(f"Asset integrity failed: {item['name']}")
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    expected_hidden = 2048 if manifest["model"] == MODEL else 1024
    if config.get("model_type") != "qwen3_5" or config["text_config"]["hidden_size"] != expected_hidden:
        raise ValueError("Model configuration does not match manifest identity")
    return manifest


def download(destination, processor_only=False):
    from huggingface_hub import HfApi, hf_hub_download
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # The destination belongs to this pinned model; never mix existing model versions.
    for name in ("DOWNLOAD_MANIFEST.json", "PROCESSOR_MANIFEST.json"):
        if (destination / name).exists():
            old = json.loads((destination / name).read_text(encoding="utf-8"))
            if (old.get("model"), old.get("revision")) != (MODEL, REVISION):
                raise ValueError("Destination contains another model/revision; choose a new directory")
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("Hub returned an unexpected revision")
    metadata = {item.rfilename: item for item in info.siblings}
    records = []
    for name in PROCESSOR_FILES + (() if processor_only else WEIGHT_FILES):
        path = Path(hf_hub_download(MODEL, name, revision=REVISION, local_dir=destination))
        digest = file_hash(path)
        item = metadata[name]
        expected_lfs = item.lfs.sha256 if item.lfs else None
        if path.stat().st_size != item.size or (expected_lfs and digest != expected_lfs):
            raise ValueError(f"Official metadata mismatch: {name}")
        records.append({"name": name, "bytes": path.stat().st_size, "sha256": digest,
                        "expected_lfs_sha256": expected_lfs})
    manifest = {"model": MODEL, "revision": REVISION, "files": records}
    name = "PROCESSOR_MANIFEST.json" if processor_only else "DOWNLOAD_MANIFEST.json"
    temporary = destination / (name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination / name)
    if not processor_only:
        verify_assets(destination)
    print(json.dumps({"model": MODEL, "revision": REVISION, "directory": str(destination),
                      "files": len(records), "weights_downloaded": not processor_only}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--processor-only", action="store_true", help="Small assets for CPU tests; not usable for GPU inference")
    parser.add_argument("--verify-only", action="store_true", help="Offline full-model integrity check")
    args = parser.parse_args()
    if args.verify_only and args.processor_only:
        parser.error("--verify-only requires the full model")
    if args.verify_only:
        manifest = verify_assets(args.model_dir)
        if manifest["model"] != MODEL:
            parser.error("This entrypoint requires Qwen3.5-2B")
        print("Verified pinned Qwen3.5-2B assets")
    else:
        download(args.model_dir, args.processor_only)


if __name__ == "__main__":
    main()
