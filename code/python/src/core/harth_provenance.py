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


# Stage 24 is archived as a 13-code-cell Colab execution. The current readable
# study notebook intentionally omits the original Colab clone/setup cell.
# Two remaining cells have saved outputs but missing execution-count metadata.
STAGE24_EVIDENCE_COMMIT = "c4548596656acaffad4b024203e554c2c3aa3da0"
STAGE24_EVIDENCE_PATH = (
    "studies/wearable_activity_recognition/notebooks/"
    "24_multiseed_and_imbalance_robustness.ipynb"
)


def read_historical_stage24_bytes(repository_root, approved_sha256):
    """Retrieve and hash-check the original executed Stage 24 from Git history."""
    root = Path(repository_root).resolve()
    if not (root / ".git").exists():
        raise ValueError("Stage 24 provenance needs a full Git checkout.")
    process = subprocess.run(
        ["git", "show", f"{STAGE24_EVIDENCE_COMMIT}:{STAGE24_EVIDENCE_PATH}"],
        cwd=root, capture_output=True, check=False,
    )
    if process.returncode != 0:
        raise ValueError(
            "Original Stage 24 Git evidence unavailable; fetch full Git history."
        )
    if hashlib.sha256(process.stdout).hexdigest() != approved_sha256:
        raise ValueError("Historical Stage 24 SHA256 differs from approval.")
    return process.stdout


def verify_stage24_visual_format(active_path, historical_bytes):
    """Verify only permitted presentation differences, without editing the notebook.

    The archived first code cell is Colab setup, not a study experiment. All 12
    subsequent code cells must have exactly the same executable statements and
    saved outputs as the 12 current code cells. Exactly two current code cells
    have missing execution counts (positions 1 and 10); this remains an explicit
    metadata gap, never evidence of newly rerun execution.
    """
    import json

    archived = json.loads(historical_bytes.decode("utf-8"))
    current = json.loads(Path(active_path).read_text(encoding="utf-8"))
    original_code = [
        cell for cell in archived["cells"] if cell.get("cell_type") == "code"
    ]
    current_code = [
        cell for cell in current["cells"] if cell.get("cell_type") == "code"
    ]
    if len(original_code) != 13 or len(current_code) != 12:
        raise ValueError(
            "Stage 24 code-cell counts must be 13 archived / 12 current."
        )
    def joined(value):
        return "".join(value) if isinstance(value, list) else str(value)

    setup = joined(original_code[0].get("source", ""))
    if "%cd /content" not in setup or "!git clone " not in setup:
        raise ValueError("Archived Stage 24 initial Colab setup is unexpected.")

    def executable_lines(cell):
        return [
            line for line in joined(cell.get("source", "")).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    original_research_code = original_code[1:]
    missing_counts = []
    for index, (historic, displayed) in enumerate(
        zip(original_research_code, current_code), start=1
    ):
        if historic.get("execution_count") is None:
            raise ValueError("Historical Stage 24 has missing execution metadata.")
        if executable_lines(historic) != executable_lines(displayed):
            raise ValueError(
                f"Stage 24 executable code changed in study cell {index}."
            )
        if historic.get("outputs") != displayed.get("outputs"):
            raise ValueError(
                f"Stage 24 saved scientific outputs changed in study cell {index}."
            )
        if any(
            output.get("output_type") == "error"
            for output in historic.get("outputs", [])
        ):
            raise ValueError("Archived Stage 24 contains a saved execution error.")
        if displayed.get("execution_count") is None:
            if not displayed.get("outputs"):
                raise ValueError(
                    f"Stage 24 study cell {index} lacks both count and output."
                )
            missing_counts.append(index)
        elif displayed["execution_count"] != historic["execution_count"]:
            raise ValueError(
                f"Stage 24 execution metadata changed in study cell {index}."
            )

    if missing_counts != [1, 10]:
        raise ValueError(
            "Stage 24 missing execution metadata differs from documented history."
        )
    for notebook, codes in (
        (archived, original_code), (current, current_code)
    ):
        last_outputs = codes[-1].get("outputs", [])
        gate = "".join(
            joined(output.get("text", "")) for output in last_outputs
        )
        if "Stage 24 scientific quality gate: PASSED" not in gate:
            raise ValueError("Stage 24 final saved quality gate is missing.")

    return {
        "current_code_cells": len(current_code),
        "historical_code_cells": len(original_code),
        "saved_output_cells": sum(bool(c.get("outputs")) for c in current_code),
        "missing_execution_counts": missing_counts,
        "execution_metadata_complete": False,
        "historical_execution_verified": True,
    }
