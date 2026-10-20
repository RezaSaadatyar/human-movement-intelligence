"""Verify the original Stage 23 notebook through pinned Git history.

The original saved execution is not duplicated in the current file tree.
A full Git checkout is required to retrieve its immutable historical bytes.
"""
import hashlib
from pathlib import Path
import subprocess


STAGE23_EVIDENCE_COMMIT = "433f972676a3a0a39c05b34612f201bb27366d36"
STAGE23_EVIDENCE_PATH = (
    "studies/wearable_activity_recognition/experiments/"
    "frozen_notebook_evidence/23_unified_matched_model_benchmark.ipynb"
)


def read_historical_stage23_bytes(repository_root, approved_sha256):
    """Return original executed bytes only when the approved hash matches."""
    root = Path(repository_root).resolve()
    if not (root / ".git").exists():
        raise ValueError(
            "Stage 23 provenance needs a full Git checkout with history."
        )
    reference = f"{STAGE23_EVIDENCE_COMMIT}:{STAGE23_EVIDENCE_PATH}"
    process = subprocess.run(
        ["git", "show", reference],
        cwd=root, capture_output=True, check=False,
    )
    if process.returncode != 0:
        raise ValueError(
            "Original Stage 23 Git evidence is unavailable. "
            "Fetch full repository history before verifying this study."
        )
    recorded = hashlib.sha256(process.stdout).hexdigest()
    if recorded != approved_sha256:
        raise ValueError("Historical Stage 23 SHA256 differs from approval.")
    return process.stdout
