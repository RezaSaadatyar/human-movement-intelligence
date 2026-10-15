# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Lightweight provenance audit of saved Jupyter notebook evidence.

This module does NOT run notebook cells, fit models, download source CSVs,
modify saved outputs or infer that an unexecuted cell was successfully run.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd


# =========================== 1. Notebook Inventory ============================
def collect_notebook_inventory(notebook_dir):
    """
    Read on-disk numbered notebooks and summarise their actual saved state.

    Args:
        notebook_dir (str | pathlib.Path): Directory with numbered .ipynb
            files produced by the HARTH research workflow.

    Returns:
        pd.DataFrame: One row per notebook with its filename, number,
            source SHA256, number of code cells, executed code cells,
            output-bearing cells, saved execution errors and explicit
            PASS markers contained in saved text.

    Notes:
        Execution counts and saved outputs are evidence of saved runs,
        not proof of scientific correctness or reproducibility on new data.
        Missing execution counts are retained as an explicit audit finding.
    """
    folder = Path(notebook_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Notebook directory is missing: {folder}")
    rows = []
    for path in sorted(folder.glob("[0-9][0-9]_*.ipynb")):
        raw = path.read_bytes()
        notebook = json.loads(raw)
        if notebook.get("nbformat") != 4 or not isinstance(
            notebook.get("cells"), list
        ):
            raise ValueError(f"Invalid Jupyter notebook structure: {path}")
        code_cells = [
            cell for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        errors = [
            str(output.get("ename", "UnspecifiedError"))
            for cell in code_cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        saved_text = extract_saved_output_text(notebook)
        rows.append({
            "stage": int(path.name.split("_", 1)[0]),
            "notebook": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "code_cells": len(code_cells),
            "executed_code_cells": sum(
                cell.get("execution_count") is not None
                for cell in code_cells
            ),
            "cells_with_outputs": sum(
                bool(cell.get("outputs", [])) for cell in code_cells
            ),
            "unexecuted_code_cells": sum(
                cell.get("execution_count") is None
                for cell in code_cells
            ),
            "saved_error_count": len(errors),
            "saved_errors": "; ".join(errors),
            "saved_pass_marker": "PASSED" in saved_text,
        })
    if not rows:
        raise FileNotFoundError(f"No numbered notebooks in {folder}")
    frame = pd.DataFrame(rows).sort_values("stage").reset_index(drop=True)
    if frame["stage"].duplicated().any():
        raise ValueError("Duplicate notebook stage numbers.")
    return frame


# ========================== 2. Saved Text Extraction =========================
def extract_saved_output_text(notebook):
    """
    Extract saved stream and plain-text output, without executing code.

    Args:
        notebook (dict): Parsed Jupyter notebook JSON object.

    Returns:
        str: Newline-joined saved 'stream' and 'text/plain' representations.

    Notes:
        Embedded plot images and HTML remain in the original notebook.
        This function does not perform OCR or convert rendered plots to text.
    """
    def joined(value):
        if isinstance(value, list):
            return "".join(str(part) for part in value)
        return str(value) if value is not None else ""

    collected = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                collected.append(joined(output.get("text")))
            elif output.get("output_type") in ("execute_result", "display_data"):
                data = output.get("data") or {}
                if "text/plain" in data:
                    collected.append(joined(data["text/plain"]))
    return "\n".join(collected)


# ======================= 3. Registry/Source Consistency =======================
def validate_evidence_register(inventory, experiment_register):
    """
    Check that the human-reviewed experiment register points to saved notebooks.

    Args:
        inventory (pd.DataFrame): collect_notebook_inventory() result.
        experiment_register (pd.DataFrame): Evidence registry with exactly one
            row per stage and a reference to the source notebook filename.

    Returns:
        pd.DataFrame: Registry joined to actual execution evidence.

    Notes:
        Manual numeric results in the register require human review against
        the saved source cell outputs; filenames alone do not verify metrics.
    """
    needed = {"stage", "notebook", "research_question", "protocol",
              "key_evidence", "limitations"}
    if not isinstance(experiment_register, pd.DataFrame) or not needed.issubset(
        experiment_register.columns
    ):
        raise ValueError("Evidence register is missing required columns.")
    if (experiment_register["stage"].duplicated().any()
            or experiment_register["stage"].isna().any()):
        raise ValueError("Evidence register requires unique stage numbers.")
    merged = experiment_register.merge(
        inventory, on=["stage", "notebook"], how="left",
        validate="one_to_one", indicator=True
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("Registered stage not found in numbered notebook files.")
    return merged.drop(columns="_merge").sort_values("stage").reset_index(drop=True)
