import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from kg_llm_tune.resume_bundle import build_resume_archive


def make_run(root: Path):
    for name in ("config.yaml", "env.json", "metrics.json", "log.txt"):
        (root / name).write_text("fixture\n", encoding="utf-8")
    checkpoint = root / "checkpoints" / "step-00000007"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (checkpoint / "state.pt").write_bytes(b"optimizer-rng-state")
    (checkpoint / "manifest.json").write_text(json.dumps({
        "optimizer_steps": 7,
        "signature": {"train_sha256": "train", "val_sha256": "val"},
    }), encoding="utf-8")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")


class ResumeBundleTests(unittest.TestCase):
    def test_archive_contains_complete_resume_state_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            run = tmp_path / "run"
            run.mkdir()
            make_run(run)
            archive_path = tmp_path / "handoff.zip"

            report = build_resume_archive(run, archive_path, "abc123", "test/model")

            self.assertTrue(report["verified"])
            self.assertEqual(report["optimizer_steps"], 7)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertIsNone(archive.testzip())
                names = set(archive.namelist())
                self.assertIn("resume_bundle/checkpoints/step-00000007/state.pt", names)
                manifest = json.loads(archive.read("resume_bundle/artifact_manifest.json"))
                self.assertEqual(manifest["source_commit"], "abc123")
                self.assertEqual(manifest["model_id"], "test/model")


    def test_archive_refuses_incomplete_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            run = tmp_path / "run"
            run.mkdir()
            make_run(run)
            (run / "checkpoints" / "step-00000007" / "adapter" / "adapter_model.safetensors").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "complete resumable checkpoint"):
                build_resume_archive(run, tmp_path / "handoff.zip", "abc123", "test/model")
            self.assertFalse((tmp_path / "handoff.zip").exists())


if __name__ == "__main__":
    unittest.main()
