# ==================== Presented by: Reza Saadatyar (2026) ====================

"""Bounded inspection of MATLAB objects for trustworthy numeric recovery.

A numeric leaf inside an opaque MATLAB object is only a *candidate*, not a
verified IMU channel or finite-element strain. This module never invents
column names, silently interpolates FE outputs, or exports unknown arrays
as model-ready tensors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat




# ================ 01. Original Source and Registry Integrity =================

def check_record_registry(registry, audit):
    """Validate identities and record coverage against saved Stage 02 evidence."""
    expected = audit["source_record_summary"]
    required = {
        "subject_id", "source_file", "local_record_index",
        "local_record_key", "task", "updated_trial_id",
        "imu_representation", "fe_representation",
    }
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"Stage 02 record registry missing columns: {sorted(missing)}")
    if len(registry) != int(expected["source_records"]):
        raise ValueError("Stage 02 record count differs from its signed-off JSON.")
    if registry["subject_id"].nunique() != int(expected["source_participants"]):
        raise ValueError("Participant count differs from Stage 02 evidence.")
    if registry["local_record_key"].isna().any():
        raise ValueError("Missing file-local source record keys.")
    if not registry["local_record_key"].is_unique:
        raise ValueError("Repeated file-local source record keys.")
    if registry["updated_trial_id"].isna().any():
        raise ValueError("Stage 02 contains missing original trial identifiers.")
    joined = registry["subject_id"].str.cat(
        registry["updated_trial_id"].astype(str), sep="::"
    )
    if joined.duplicated().any():
        raise ValueError("Duplicate participant/trial identifiers.")
    if not (registry["source_file"].str.replace(
        ".mat", "", regex=False
    ) == registry["subject_id"]).all():
        raise ValueError("File and participant identities are inconsistent.")
    return {
        "records": int(len(registry)),
        "participants": int(registry["subject_id"].nunique()),
        "unique_source_record_keys": True,
        "unique_subject_trial_keys": True,
    }


def select_diagnostic_records(registry, max_samples=64):
    """Choose representative source records without using model outcomes."""
    order = registry.sort_values(
        ["subject_id", "local_record_index"], kind="stable"
    )
    # Include each participant and each activity at least once where possible.
    selected = [
        order.drop_duplicates("subject_id"),
        order.drop_duplicates("task"),
        order.drop_duplicates("fe_representation"),
        order.drop_duplicates(["task", "fe_representation"]),
    ]
    chosen = pd.concat(selected, ignore_index=True)
    chosen = chosen.drop_duplicates("local_record_key")
    chosen = chosen.head(max_samples).reset_index(drop=True)
    if chosen.empty:
        raise ValueError("No source records were selected for diagnostics.")
    return chosen




# =============== 02. Bounded MATLAB Nested-Value Introspection ===============

def _shape(value):
    """Describe an array shape without converting its payload to numbers."""
    if isinstance(value, np.ndarray):
        return str(tuple(int(x) for x in value.shape))
    return ""


def walk_matlab_object(value, *, max_depth=9, max_nodes=300, max_children=8):
    """Return bounded representation paths and genuine numeric leaf candidates.

    Numeric leaves below MATLAB object metadata are still unverified.
    No arbitrary byte payload or metadata index is relabelled as IMU/FE data.
    """
    if max_depth < 1 or max_nodes < 5 or max_children < 1:
        raise ValueError("Inspection bounds must be positive and meaningful.")

    pending = [("$", value, 0, False)]
    report = []
    inspected = 0
    while pending and inspected < max_nodes:
        path, item, depth, inside_opaque = pending.pop()
        inspected += 1
        kind = type(item).__name__
        opaque_here = inside_opaque or (
            "Opaque" in kind or "MatlabObject" in kind
        )
        data_type = ""
        size = 0
        shape = _shape(item)
        status = "scalar_or_unknown"
        children = []
        numeric_leaf = False
        finite_numeric = False

        if isinstance(item, np.ndarray):
            data_type = str(item.dtype)
            size = int(item.size)
            if item.dtype.names:
                status = "opaque_structured" if opaque_here else "structured"
                for field in item.dtype.names[:max_children]:
                    children.append((f"{path}.{field}", item[field], opaque_here))
            elif item.dtype.kind == "O":
                status = "object_array"
                for index in range(min(size, max_children)):
                    children.append((
                        f"{path}[{index}]", item.flat[index], opaque_here
                    ))
            elif item.dtype.kind in "fiu":
                status = "numeric_array"
                numeric_leaf = bool(size > 1)
                finite_numeric = bool(
                    numeric_leaf and np.isfinite(item).all()
                )
                if numeric_leaf and not finite_numeric:
                    status = "nonfinite_numeric"
            elif item.dtype.kind in "US":
                status = "text_array"
            elif item.dtype.kind == "b":
                status = "boolean_array"
            else:
                status = "other_array"
        elif isinstance(item, np.void) and item.dtype.names:
            data_type = str(item.dtype)
            status = "structured_scalar"
            for field in item.dtype.names[:max_children]:
                children.append((f"{path}.{field}", item[field], opaque_here))
        elif hasattr(item, "_fieldnames"):
            status = "matlab_struct"
            children = [
                (f"{path}.{field}", getattr(item, field, None), opaque_here)
                for field in list(item._fieldnames)[:max_children]
            ]
        elif isinstance(item, (tuple, list)):
            status = "python_sequence"
            children = [
                (f"{path}[{i}]", child, opaque_here)
                for i, child in enumerate(item[:max_children])
            ]
        elif isinstance(item, (str, bytes, np.str_)):
            status = "text_scalar"
        elif isinstance(item, (int, float, np.number)):
            status = "numeric_scalar"

        report.append({
            "path": path,
            "depth": depth,
            "python_class": kind,
            "dtype": data_type,
            "shape": shape,
            "size": size,
            "representation": status,
            "beneath_matlab_object": bool(opaque_here),
            "numeric_leaf": numeric_leaf,
            "finite_numeric": finite_numeric,
            "has_101_axis": bool(
                numeric_leaf and isinstance(item, np.ndarray)
                and 101 in item.shape
            ),
            "has_11_axis": bool(
                numeric_leaf and isinstance(item, np.ndarray)
                and 11 in item.shape
            ),
            "attribution": (
                "UNVERIFIED - requires native MATLAB or independent schema"
                if numeric_leaf else "NOT_APPLICABLE"
            ),
        })
        if depth < max_depth:
            for child in reversed(children):
                pending.append((child[0], child[1], depth + 1, child[2]))

    return report, {
        "nodes_inspected": inspected,
        "inspection_truncated": bool(pending),
        "max_depth": max_depth,
        "max_nodes": max_nodes,
        "max_children": max_children,
    }




# =========== 03. Source Record Probes and Candidate Classification ===========

def _field(record, name):
    if not hasattr(record, "_fieldnames"):
        raise ValueError("Expected original MATLAB Data struct.")
    return getattr(record, name, None)


def inspect_sampled_records(mat_dir, samples, *, max_depth=9, max_nodes=300):
    """Trace selected original IMU/FE objects without exporting raw signals."""
    mat_dir = Path(mat_dir)
    reports = []
    probe_registry = []
    for filename, group in samples.groupby("source_file", sort=True):
        source = mat_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Source participant MAT file absent: {source}")
        payload = loadmat(
            source, variable_names=["Data"], squeeze_me=False,
            struct_as_record=False, verify_compressed_data_integrity=True,
        )
        records = np.asarray(payload["Data"]).reshape(-1)
        for sampled in group.itertuples(index=False):
            idx = int(sampled.local_record_index)
            if idx < 0 or idx >= len(records):
                raise ValueError(f"Out-of-range record index in {filename}: {idx}")
            record = records[idx]
            for modality, field in [("IMU", "IMU"), ("FE", "e1e3")]:
                original = _field(record, field)
                rows, summary = walk_matlab_object(
                    original, max_depth=max_depth, max_nodes=max_nodes
                )
                for row in rows:
                    reports.append({
                        "local_record_key": sampled.local_record_key,
                        "subject_id": sampled.subject_id,
                        "task": sampled.task,
                        "updated_trial_id": sampled.updated_trial_id,
                        "modality": modality,
                        **row,
                    })
                probe_registry.append({
                    "local_record_key": sampled.local_record_key,
                    "subject_id": sampled.subject_id,
                    "task": sampled.task,
                    "modality": modality,
                    **summary,
                    "numeric_leaves_visible": sum(
                        int(row["numeric_leaf"]) for row in rows
                    ),
                })
    return pd.DataFrame(reports), pd.DataFrame(probe_registry)


def candidate_inventory(paths):
    """Describe actual numeric leaves, never silently adopting their meaning."""
    candidates = paths.loc[paths["numeric_leaf"]].copy()
    candidates["time_axis_candidate"] = (
        candidates["has_101_axis"] | candidates["has_11_axis"]
    )
    candidates["usable_for_prediction"] = False
    # A numeric leaf from an opaque object has no verified channel/target name.
    candidates["validation_reason"] = (
        "Numeric leaf only; labels, axes, units and timing unverified"
    )
    return candidates




# ============== 04. Source Data Coverage and Quality Decisions ===============

def activity_fe_coverage(registry):
    """Count FE source representations within each recorded task."""
    if registry["task"].isna().any():
        raise ValueError("Missing activity values in the source registry.")
    coverage = (
        registry.groupby(["task", "fe_representation"], dropna=False)
        .size().rename("record_count").reset_index()
    )
    if int(coverage["record_count"].sum()) != len(registry):
        raise ValueError("FE coverage fails to account for all source records.")
    return coverage


def assert_source_fe_status_matches(registry, audit):
    """Require consistency with immutable Stage 02 source counts."""
    observed = registry["fe_representation"].value_counts().to_dict()
    expected = audit["source_record_summary"]["fe_encoding_counts"]
    if observed != expected:
        raise ValueError(
            "FE field representations differ from Stage 02 source evidence."
        )
    return True

