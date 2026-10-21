# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Non-destructive HARTH pre-release provenance and local-artifact audit.

Uses Python standard library only. Does NOT load HARTH CSVs, PyTorch, joblib
models, execute notebooks, change manifests, or evaluate Test participants.
Run from any directory: python scripts/check_harth_release_readiness.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "wearable_activity_recognition"
NOTEBOOKS = STUDY / "notebooks"
MANIFESTS = STUDY / "evidence"
EXPECTED_MODELS = ("resnet_state_dict.pt", "random_forest.joblib", "inference_config.json")
EXPECTED_CLASSES = list(range(1, 9))
EXPECTED_TEST_SUBJECTS = ["S023", "S032"]
FROZEN_FILES = (
    "code/python/src/core/advanced_temporal_models.py",
    "code/python/src/core/temporal_cnn.py",
    "code/python/src/core/temporal_benchmark.py",
    "code/python/src/core/feature_engineering.py",
    "code/python/src/core/robustness_experiments.py",
    "studies/wearable_activity_recognition/src/harth_data.py",
)


sys.path.insert(0, str(ROOT / "code/python/src"))
from core.harth_provenance import (
    read_historical_stage23_bytes,
    read_historical_stage24_bytes,
    verify_stage24_visual_format,
    verify_stage23_visual_format as verified_stage23_replay,
)


def file_sha256(path):
    """Return the original byte-level SHA256, without modifying the file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_approved_text_bytes(path, expected_sha256):
    """
    Strictly verify bytes against an approved SHA256 with Windows EOL allowance.

    Args:
        path (str | Path): Source or saved notebook that Git may check out as CRLF.
        expected_sha256 (str): Approved hash from original LF-based evidence.

    Returns:
        str: EXACT or CRLF_TO_LF_ONLY.

    Notes:
        The only tolerated difference is converting CRLF newline pairs to LF.
        No other whitespace, comments, code, characters or labels are changed.
        Never replaces the stored file or the approved manifest hash.
    """
    path = Path(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() == expected_sha256:
        return "EXACT"
    if b"\r\n" in raw:
        canonical = raw.replace(b"\r\n", b"\n")
        if hashlib.sha256(canonical).hexdigest() == expected_sha256:
            return "CRLF_TO_LF_ONLY"
    raise ValueError(
        "Frozen source checksum mismatch after strict CRLF-to-LF check: "
        + str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    )


def notebook_execution_check(
    path, expected_code_cells, final_marker,
    allow_missing_counts_with_outputs=False,
):
    """Audit committed output evidence and report missing execution metadata.

    Record saved output availability and original execution metadata.
    Counter irregularities are reported as raw metadata, not as scientific
    performance failures. The final Stage 27 report includes Colab setup
    plus ten scientific cells and must carry a genuine final Quality Gate.
    """
    path = Path(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
    if len(cells) != expected_code_cells:
        raise ValueError("Wrong saved notebook code-cell count: " + path.name)
    missing = [
        index + 1 for index, cell in enumerate(cells)
        if cell.get("execution_count") is None
    ]
    if missing and not allow_missing_counts_with_outputs:
        raise ValueError(
            "Unverified notebook execution counts: " + path.name
        )
    if missing and any(not cells[index - 1].get("outputs") for index in missing):
        raise ValueError(
            "Missing both execution count and saved cell output: " + path.name
        )
    if any(o.get("output_type") == "error"
           for c in cells for o in c.get("outputs", [])):
        raise ValueError("Saved notebook error: " + path.name)
    last = "\n".join(
        "".join(o.get("text", [])) if isinstance(o.get("text"), list)
        else str(o.get("text", ""))
        for o in cells[-1].get("outputs", []) if o.get("output_type") == "stream"
    )
    if final_marker not in last:
        raise ValueError("Missing final saved Quality Gate: " + path.name)
    return {
        "cells": len(cells),
        "notebook_sha256": file_sha256(path),
        "saved_output_cells": sum(bool(c.get("outputs")) for c in cells),
        "missing_execution_counts": missing,
        "execution_metadata_complete": not missing,
    }



def verify_stage23_visual_format(current_path, historic_bytes):
    """Delegate strict science-aware replay checks to the common verifier."""
    return verified_stage23_replay(current_path, historic_bytes)


def verify_recorded_evidence():
    """
    Validate approved protocol and exact frozen code, without source/model access.

    Returns:
        dict: Versioned manifest lineage and saved Notebook 23–27 metadata.
    """
    pending_path = MANIFESTS / "stage_25_protocol_proposal_reconciled.json"
    approved_path = MANIFESTS / "stage_25_protocol_approved_qualified_test.json"
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    approved = json.loads(approved_path.read_text(encoding="utf-8"))
    if pending["status"] != "PENDING_USER_REVIEW" or approved["status"] != "APPROVED":
        raise ValueError("Stage 25 lineage is not the reviewed pending/approved pair.")
    if {k: v for k, v in approved.items() if k not in ("status", "approval_record")} != {
        k: v for k, v in pending.items() if k != "status"
    }:
        raise ValueError("Frozen approved and reconciled protocols differ.")
    if not approved.get("approval_record"):
        raise ValueError("Explicit approval provenance missing.")
    spec = approved["source"]
    if (spec["dataset"] != "HARTH v2.0" or
        spec["task_codes"] != EXPECTED_CLASSES or
        spec["reserved_test_people"] != EXPECTED_TEST_SUBJECTS or
        spec["development_participants"] != 29 or
        spec["expected_development_windows"] != 14201):
        raise ValueError("Unexpected frozen cohort, task or original Test policy.")
    recorded = approved["module_sha256"]
    if set(recorded) != set(FROZEN_FILES):
        raise ValueError("Six frozen source modules were not recorded.")
    eol_adaptations = []
    for relative, expected in recorded.items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():
            raise ValueError("Missing/unsafe frozen module: " + relative)
        status = verify_approved_text_bytes(source, expected)
        if status == "CRLF_TO_LF_ONLY":
            eol_adaptations.append(relative)

    notebooks = {}
    expected = {
        23: (12, "Stage 23 scientific quality gate: PASSED"),
        24: (13, "Stage 24 scientific quality gate: PASSED"),
        25: (11, "Stage 25 development selection audit: PASSED"),
        26: (13, "STAGE 26 | QUALIFIED HELD-OUT EVALUATION: PASSED"),
        27: (11, "STAGE 27 | SOURCE-GROUNDED RESEARCH SYNTHESIS QUALITY GATE: PASSED"),
    }
    for stage, (count, gate) in expected.items():
        matches = list(NOTEBOOKS.glob(f"{stage:02d}_*.ipynb"))
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous notebook Stage {stage}")
        if stage == 24:
            historical = read_historical_stage24_bytes(
                ROOT, approved["stage24_saved_notebook_sha256"]
            )
            verified = verify_stage24_visual_format(matches[0], historical)
            notebooks[str(stage)] = {
                "cells": verified["current_code_cells"],
                "historical_executed_code_cells": verified["historical_code_cells"],
                "notebook_sha256": file_sha256(matches[0]),
                "historical_notebook_sha256": approved["stage24_saved_notebook_sha256"],
                "saved_output_cells": verified["saved_output_cells"],
                "missing_execution_counts": verified["missing_execution_counts"],
                "execution_metadata_complete": verified["execution_metadata_complete"],
                "historical_execution_verified": True,
            }
        else:
            notebooks[str(stage)] = notebook_execution_check(
                matches[0], count, gate,
                allow_missing_counts_with_outputs=False,
            )
    # Recover the original Stage 23 execution from a pinned Git revision.
    # Do not weaken byte-level approval checks or create duplicate notebooks.
    stage23_active = next(NOTEBOOKS.glob("23_*.ipynb"))
    historical23 = read_historical_stage23_bytes(
        ROOT, approved["stage23_saved_notebook_sha256"]
    )
    verify_stage23_visual_format(stage23_active, historical23)

    required = [
        "README.md",
        "studies/wearable_activity_recognition/README.md",
        "studies/wearable_activity_recognition/docs/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md",
        "docs/ROADMAP.md",
        "docs/HARTH_REPRODUCIBILITY.md",
        "LICENSE",
        ".gitignore",
    ]
    if any(not (ROOT / path).is_file() for path in required):
        raise ValueError("A required documentation/licence/ignore file is missing.")
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    if any(item not in ignored for item in ("data/raw/", "outputs/", ".env", "*.joblib", "*.mat")):
        raise ValueError("Required sensitive/raw-data/artifact ignore patterns missing.")

    return {
        "provenance": "VERIFIED",
        "source_dataset": spec["dataset"],
        "source_revision": spec["revision"],
        "development_cohort_sha256": approved["source_development_sha256"],
        "approved_manifest_sha256": file_sha256(approved_path),
        "frozen_module_hashes": len(recorded),
        "verified_crlf_to_lf_paths": eol_adaptations,
        "saved_notebooks": notebooks,
        "original_test_reopened_by_this_audit": False,
    }


def inspect_optional_artifacts(artifact_dir, expected_development_sha256):
    """
    Inspect local trusted files by byte hashing and metadata only.

    Args:
        artifact_dir (Path | None): Opt-in local trusted model directory.
        expected_development_sha256 (str): Exact source cohort fingerprint.

    Returns:
        dict: VERIFIED, NOT_CHECKED or FAILED state and SHA256 values.

    Notes:
        Merely checking a checksum does NOT authenticate an untrusted joblib
        file. Never deserialize untrusted joblib or pickle artefacts.
    """
    if artifact_dir is None:
        return {"status": "NOT_CHECKED", "reason": "No artifact directory supplied"}
    directory = Path(artifact_dir).expanduser().resolve()
    if not directory.is_dir():
        return {"status": "FAILED", "reason": "Artifact directory not found"}
    missing = [name for name in EXPECTED_MODELS if not (directory / name).is_file()]
    if missing:
        return {"status": "FAILED", "reason": "Missing: " + ", ".join(missing)}
    config = json.loads((directory / "inference_config.json").read_text(encoding="utf-8"))
    correct = (
        config.get("development_cohort_sha256") == expected_development_sha256
        and config.get("sensor_columns") ==
        ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
        and config.get("source_codes") == EXPECTED_CLASSES
        and config.get("window_size") == 100
        and config.get("window_stride") == 50
        and config.get("preprocessing_fit_scope") == "Development only"
        and config.get("test_fitted_values") is False
    )
    if not correct:
        return {"status": "FAILED", "reason": "Frozen inference metadata mismatch"}
    return {
        "status": "LOCAL_FILES_VERIFIED",
        "limitations": "Original artifact SHA256 values were not archived in full in saved Notebook 26 output. This check cannot independently prove these bytes match the original Colab artefacts.",
        "files_sha256": {name: file_sha256(directory / name) for name in EXPECTED_MODELS},
    }


def main():
    """Print a conservative and usable pre-release evidence audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=None,
                        help="Opt-in trusted directory holding three Stage 26 model files")
    parser.add_argument("--require-artifacts", action="store_true",
                        help="Fail if the original local model files cannot be checked")
    args = parser.parse_args()

    evidence = verify_recorded_evidence()
    artifacts = inspect_optional_artifacts(
        args.artifact_dir, evidence["development_cohort_sha256"]
    )
    print("=" * 70)
    print("HARTH | PRE-RELEASE REPRODUCIBILITY AUDIT")
    print("=" * 70)
    print("Saved research evidence and frozen source checks: VERIFIED")
    adaptations = evidence["verified_crlf_to_lf_paths"]
    if adaptations:
        print("Windows CRLF-to-LF-only verified paths:", len(adaptations))
        for path in adaptations:
            print("  -", path)
    else:
        print("All approved byte fingerprints: EXACT")
    print("Development participants: 29 | original reserved Test: S023, S032")
    print("Executed notebooks validated: 23, 24, 25, 26, 27")
    for stage, notebook_record in evidence["saved_notebooks"].items():
        missing = notebook_record["missing_execution_counts"]
        if missing:
            print(
                f"Notebook {stage} provenance warning: {len(missing)} "
                "saved output-bearing cells have no execution_count; "
                "do not claim fully verified execution metadata."
            )
    print("Local model artefacts:", artifacts["status"])
    if artifacts["status"] != "LOCAL_FILES_VERIFIED":
        print("Artifact note:", artifacts["reason"])
    print("New training or Test re-evaluation: NO")
    print("GitHub Release created: NO")
    print("=" * 70)
    print("Scientific evidence audit: PASSED")
    print("Full end-to-end GPU/model-artifact reproduction: NOT ESTABLISHED")
    if args.require_artifacts and artifacts["status"] != "LOCAL_FILES_VERIFIED":
        print("Strict local artifact check: FAILED")
        return 1
    if artifacts["status"] == "FAILED":
        print("Provided local artifact check: FAILED")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print("HARTH RELEASE AUDIT FAILED:", error, file=sys.stderr)
        sys.exit(1)
