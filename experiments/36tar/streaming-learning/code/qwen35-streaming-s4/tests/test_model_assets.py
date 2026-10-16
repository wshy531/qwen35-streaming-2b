"""Asset identities and corruption are checked without real weights or network."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.model_assets import MODEL, REVISIONS, PROCESSOR_FILES, WEIGHT_FILES, file_hash, verify_assets


class ModelAssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, model=MODEL):
        files = []
        for name in PROCESSOR_FILES + WEIGHT_FILES:
            path = self.root / name
            path.write_bytes(b"fixture")
            if name == "config.json":
                path.write_text(json.dumps({"model_type":"qwen3_5", "text_config":{
                    "hidden_size":2048 if model == MODEL else 1024}}), encoding="utf-8")
            files.append({"name":name,"bytes":path.stat().st_size,"sha256":file_hash(path).upper()})
        manifest = {"model":model,"revision":REVISIONS[model],"files":files}
        self.write(manifest)
        return manifest

    def write(self, manifest):
        (self.root / "DOWNLOAD_MANIFEST.json").write_text(json.dumps(manifest),encoding="utf-8")

    def test_both_pinned_models_and_old_uppercase_hashes(self):
        for model in REVISIONS:
            manifest = self.fixture(model)
            self.assertEqual(manifest,verify_assets(self.root))

    def test_wrong_revision_rejected(self):
        manifest = self.fixture()
        manifest["revision"] = "unverified"
        self.write(manifest)
        with self.assertRaisesRegex(ValueError,"unpinned"):
            verify_assets(self.root)

    def test_missing_or_duplicate_file_rejected(self):
        manifest = self.fixture()
        manifest["files"][-1] = manifest["files"][0]
        self.write(manifest)
        with self.assertRaisesRegex(ValueError,"exactly once"):
            verify_assets(self.root)

    def test_outside_path_rejected(self):
        manifest = self.fixture()
        manifest["files"][-1]["name"] = "../outside"
        self.write(manifest)
        with self.assertRaisesRegex(ValueError,"exactly once"):
            verify_assets(self.root)

    def test_corrupt_asset_rejected(self):
        self.fixture()
        (self.root / WEIGHT_FILES[-1]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"integrity"):
            verify_assets(self.root)

    def test_config_identity_checked_even_with_matching_hash(self):
        manifest = self.fixture()
        path = self.root / "config.json"
        path.write_text(json.dumps({"model_type":"qwen3_5","text_config":{"hidden_size":1024}}),encoding="utf-8")
        item = next(v for v in manifest["files"] if v["name"] == "config.json")
        item.update(bytes=path.stat().st_size,sha256=file_hash(path))
        self.write(manifest)
        with self.assertRaisesRegex(ValueError,"identity"):
            verify_assets(self.root)


if __name__ == "__main__":
    unittest.main()
