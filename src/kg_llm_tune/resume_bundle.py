"""Build and validate portable Kaggle training-resume archives."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _latest_complete_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for manifest_path in (run_dir / "checkpoints").glob("step-*/manifest.json"):
        checkpoint = manifest_path.parent
        adapter = checkpoint / "adapter"
        weights = list(adapter.glob("*.safetensors")) + list(adapter.glob("*.bin"))
        if not ((checkpoint / "state.pt").is_file() and
                (adapter / "adapter_config.json").is_file() and weights):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidates.append((int(manifest["optimizer_steps"]), checkpoint, manifest))
    if not candidates:
        raise FileNotFoundError(f"No complete resumable checkpoint under {run_dir / 'checkpoints'}")
    _, checkpoint, manifest = max(candidates, key=lambda item: item[0])
    for name in ("config.yaml", "env.json", "metrics.json", "log.txt"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Resume run is missing {name}")
    return checkpoint, manifest


def build_resume_archive(run_dir: str | Path, archive_path: str | Path,
                         source_commit: str, model_id: str) -> dict[str, Any]:
    """Write a single verified ZIP compatible with notebook resume discovery."""
    run_dir = Path(run_dir).resolve()
    archive_path = Path(archive_path).resolve()
    checkpoint, checkpoint_manifest = _latest_complete_checkpoint(run_dir)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    payload = sorted(path for path in run_dir.rglob("*") if path.is_file())
    manifest = {
        "format_version": 1,
        "optimizer_steps": int(checkpoint_manifest["optimizer_steps"]),
        "checkpoint": str(checkpoint.relative_to(run_dir)).replace(os.sep, "/"),
        "source_commit": source_commit,
        "model_id": model_id,
        "files": {
            str(path.relative_to(run_dir)).replace(os.sep, "/"): {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in payload
        },
    }
    readme = (
        "# Kaggle training resume bundle\n\n"
        f"Optimizer step: {manifest['optimizer_steps']}\nModel: {model_id}\n"
        f"Source commit: {source_commit}\n\n"
        "Attach this ZIP as a private Kaggle Dataset input with the original "
        "train/validation dataset. The notebook validates data hashes and resumes "
        "from the highest complete matching checkpoint.\n"
    )
    with tempfile.NamedTemporaryFile(dir=archive_path.parent, suffix=".tmp", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for path in payload:
                relative = Path("resume_bundle") / path.relative_to(run_dir)
                archive.write(path, str(relative).replace(os.sep, "/"))
            archive.writestr("resume_bundle/artifact_manifest.json",
                             json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            archive.writestr("resume_bundle/README.md", readme)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip():
                raise zipfile.BadZipFile("ZIP CRC validation failed")
            names = set(archive.namelist())
            required = {
                "resume_bundle/config.yaml", "resume_bundle/env.json",
                "resume_bundle/metrics.json", "resume_bundle/log.txt",
                f"resume_bundle/{manifest['checkpoint']}/state.pt",
                f"resume_bundle/{manifest['checkpoint']}/manifest.json",
                f"resume_bundle/{manifest['checkpoint']}/adapter/adapter_config.json",
                "resume_bundle/artifact_manifest.json", "resume_bundle/README.md",
            }
            if not required.issubset(names):
                raise zipfile.BadZipFile(f"Missing archive members: {sorted(required - names)}")
            for relative, expected in manifest["files"].items():
                actual = hashlib.sha256(archive.read(f"resume_bundle/{relative}")).hexdigest()
                if actual != expected["sha256"]:
                    raise zipfile.BadZipFile(f"SHA-256 mismatch for {relative}")
        os.replace(temporary, archive_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "archive": str(archive_path), "size_bytes": archive_path.stat().st_size,
        "optimizer_steps": manifest["optimizer_steps"], "checkpoint": manifest["checkpoint"],
        "file_count": len(manifest["files"]) + 2, "sha256": _sha256(archive_path),
        "verified": True,
    }
