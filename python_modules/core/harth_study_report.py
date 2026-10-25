# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Read-only, evidence-controlled reporting for HARTH Stages 23–27.

Read executed notebook outputs and approved manifests without importing raw
HARTH recordings, loading saved model weights, scoring Test or fitting models.
Displayed notebook metric tables are rounded and remain labelled as such.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.harth_selection_evidence import read_html_table

MODELS = ("1D-ResNet Weighted CE", "Random Forest")
RESERVED = ("S023", "S032")
CLASSES = tuple(range(1, 9))


def _joined(parts):
    """Convert a stored Jupyter source/output string or list to plain text."""
    return "".join(parts) if isinstance(parts, list) else str(parts or "")


def _load_executed_notebook(path, gate_text):
    """
    Load a saved, fully executed research notebook without running its cells.

    Args:
        path (str | Path): Existing executed .ipynb source.
        gate_text (str): Exact observed success phrase from the last cell.

    Returns:
        tuple[dict, str]: Notebook JSON and SHA256 of its saved bytes.

    Notes:
        A stored PASS marker demonstrates a recorded run, not independent
        reproduction on a new system or external validation.
    """
    path = Path(path)
    raw = path.read_bytes()
    notebook = json.loads(raw)
    code_cells = [c for c in notebook.get("cells", [])
                  if c.get("cell_type") == "code"]
    # A separately authored 00 checkout is preparation, not saved research.
    if code_cells and "00. Colab Repository Setup" in _joined(
        code_cells[0].get("source", "")
    ):
        checkout = code_cells.pop(0)
        if any(o.get("output_type") == "error"
               for o in checkout.get("outputs", [])):
            raise ValueError(f"Colab setup contains saved errors: {path.name}")
    if not code_cells or any(c.get("execution_count") is None for c in code_cells):
        raise ValueError(f"Saved notebook has unexecuted code: {path.name}")
    if any(output.get("output_type") == "error"
           for cell in code_cells for output in cell.get("outputs", [])):
        raise ValueError(f"Saved notebook contains errors: {path.name}")
    last_text = "\n".join(
        _joined(o.get("text", ""))
        for o in code_cells[-1].get("outputs", [])
        if o.get("output_type") == "stream"
    )
    if gate_text not in last_text:
        raise ValueError(f"Expected final Quality Gate absent: {path.name}")
    return notebook, hashlib.sha256(raw).hexdigest()


def _find_saved_table(notebook, section_text, required_columns):
    """
    Extract one source-grounded HTML table by uniquely identified code section.

    Args:
        notebook (dict): Executed Jupyter notebook JSON.
        section_text (str): Substring identifying the code cell with outputs.
        required_columns (sequence[str]): Exact result columns to locate.

    Returns:
        pandas.DataFrame: Saved displayed values, with original row ordering.

    Notes:
        Reads stored HTML tables only; never infers scores from figures.
        Displayed numbers are rounded; avoid pseudo-precision.
    """
    matches = [
        c for c in notebook["cells"]
        if c.get("cell_type") == "code" and section_text in _joined(c.get("source"))
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one executed section {section_text}.")
    found = []
    for output in matches[0].get("outputs", []):
        html = _joined(output.get("data", {}).get("text/html", ""))
        if "<table" not in html:
            continue
        try:
            frame = read_html_table(html)
        except ValueError:
            continue
        if set(required_columns).issubset(frame.columns):
            found.append(frame)
    if len(found) != 1 or found[0].isna().any().any():
        raise ValueError(f"Missing or ambiguous actual saved table: {section_text}.")
    return found[0]


def load_final_harth_evidence(stage26_path, reconciled_manifest_path,
                              approved_manifest_path, expected_development_sha256):
    """
    Verify Stage 26 held-out outputs and approval without Test re-execution.

    Args:
        stage26_path (str | Path): Committed executed Notebook 26.
        reconciled_manifest_path (str | Path): Existing original pending manifest.
        approved_manifest_path (str | Path): Separately approved frozen manifest.
        expected_development_sha256 (str): Cross-checked Stage 23/24 fingerprint.

    Returns:
        dict: Checked DataFrames (summary, per_subject, per_class, coverage),
        stored Notebook 26 SHA256, and unchanged scientific manifest.

    Notes:
        All reported metric values are extracted from displayed four-decimal
        tables, not new predictions or hidden local raw CSV outputs.
    """
    notebook, notebook_hash = _load_executed_notebook(
        stage26_path, "STAGE 26 | QUALIFIED HELD-OUT EVALUATION: PASSED"
    )
    pending = json.loads(Path(reconciled_manifest_path).read_text(encoding="utf-8"))
    approved = json.loads(Path(approved_manifest_path).read_text(encoding="utf-8"))
    if pending["status"] != "PENDING_USER_REVIEW" or approved["status"] != "APPROVED":
        raise ValueError("Invalid pending/approved protocol lineage.")
    if {k:v for k,v in pending.items() if k != "status"} != {
        k:v for k,v in approved.items() if k not in ("status","approval_record")
    }:
        raise ValueError("Approved manifest differs from frozen scientific protocol.")
    if not approved.get("approval_record") or approved["source_development_sha256"] != expected_development_sha256:
        raise ValueError("Missing approval record or Development source mismatch.")
    if approved["source"]["reserved_test_people"] != list(RESERVED):
        raise ValueError("Original Test subject set changed.")
    if approved["primary_model"]["architecture"] != "1D-ResNet":
        raise ValueError("Approved primary model changed.")

    source_text = "\n".join(
        _joined(o.get("text")) for c in notebook["cells"]
        for o in c.get("outputs", []) if o.get("output_type") == "stream"
    )
    if "Development SHA256 verified: " + expected_development_sha256 not in source_text:
        raise ValueError("Stage 26 Development cohort provenance mismatch.")
    if "Test-driven training, calibration, threshold tuning: NONE" not in source_text:
        raise ValueError("Test execution compliance record missing.")

    coverage = _find_saved_table(
        notebook, "11. Approved One-Time Qualified Test Inference",
        ("subject_id", "eligible_windows", "sampled_windows")
    )
    summary = _find_saved_table(
        notebook, "12. Frozen Qualified Test Results",
        ("model", "windows", "subjects", "fixed_eight_class_macro_f1",
         "balanced_accuracy_observed_classes", "accuracy", "true_classes_present")
    )
    subjects = _find_saved_table(
        notebook, "12. Frozen Qualified Test Results",
        ("model", "subject_id", "windows", "fixed_eight_class_macro_f1",
         "balanced_accuracy_observed_classes", "accuracy")
    )
    per_class = _find_saved_table(
        notebook, "12. Frozen Qualified Test Results",
        ("model", "source_code", "activity", "precision", "recall", "f1", "support")
    )

    if coverage.shape[0] != 2 or set(coverage.subject_id) != set(RESERVED):
        raise ValueError("Original reserved-participant coverage incomplete.")
    if (coverage.sampled_windows != 500).any() or coverage.sampled_windows.sum() != 1000:
        raise ValueError("Frozen Test cap or sample total changed.")
    if list(summary.model) != list(MODELS) or len(summary) != 2:
        raise ValueError("Frozen two-model result table missing.")
    if (summary.windows != 1000).any() or (summary.subjects != 2).any():
        raise ValueError("Frozen Test coverage mismatch.")
    if len(subjects) != 4 or set(subjects.subject_id) != set(RESERVED):
        raise ValueError("Expected four per-model/per-person records.")
    if subjects[["model", "subject_id"]].duplicated().any():
        raise ValueError("Repeated person/model metrics.")
    if (subjects.windows != 500).any():
        raise ValueError("Per-participant Test cap incorrect.")
    if len(per_class) != 16 or set(per_class.source_code) != set(CLASSES):
        raise ValueError("Expected sixteen model-class rows.")
    if per_class[["model", "source_code"]].duplicated().any():
        raise ValueError("Duplicate per-class metrics.")
    if set(per_class.model) != set(MODELS):
        raise ValueError("Unexpected model identity in per-class evidence.")
    if not all(int(group.support.sum()) == 1000
               for _, group in per_class.groupby("model")):
        raise ValueError("Per-class actual Test support fails to sum to 1,000.")
    if not all(np.isfinite(frame[col].to_numpy(dtype=float)).all() and
               frame[col].between(0,1).all()
               for frame, columns in [
                   (summary, ["fixed_eight_class_macro_f1", "accuracy",
                              "balanced_accuracy_observed_classes"]),
                   (subjects, ["fixed_eight_class_macro_f1", "accuracy"]),
                   (per_class, ["precision","recall","f1"]),
               ] for col in columns):
        raise ValueError("Metric outside feasible range.")
    for model, rows in per_class.groupby("model"):
        from_table = float(summary.set_index("model").loc[model, "fixed_eight_class_macro_f1"])
        if abs(float(rows.f1.mean()) - from_table) > 0.0002:
            raise ValueError("Reported eight-class Macro F1 not consistent with class F1.")
    return {
        "summary": summary, "per_subject": subjects, "per_class": per_class,
        "coverage": coverage, "notebook26_sha256": notebook_hash,
        "approved_manifest": approved,
    }
