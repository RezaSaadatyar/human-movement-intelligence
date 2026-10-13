"""Inspect nested MATLAB Data fields, reporting metadata rather than signals.

Object and struct arrays are sampled at their first element. This is a
bounded schema probe, not a validation of every trial or target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, whosmat
from scipy.io.matlab import mat_struct


def structure_summary(
    obj: Any, *, max_depth: int = 6, max_fields: int = 40, max_nodes: int = 400
) -> dict[str, Any]:
    """Summarise MATLAB structs and arrays without recording sample values."""
    if min(max_depth, max_fields, max_nodes) < 1:
        raise ValueError("All inspection limits must be positive.")

    nodes_seen = 0

    def walk(value: Any, depth: int) -> dict[str, Any]:
        nonlocal nodes_seen
        nodes_seen += 1
        if nodes_seen > max_nodes:
            return {"kind": "omitted", "reason": "node_limit"}
        if depth > max_depth:
            return {"kind": "omitted", "reason": "depth_limit"}

        if isinstance(value, mat_struct):
            names = list(value._fieldnames or [])
            node: dict[str, Any] = {
                "kind": "mat_struct", "field_count": len(names), "fields": {}
            }
            for name in names[:max_fields]:
                node["fields"][name] = walk(getattr(value, name), depth + 1)
            if len(names) > max_fields:
                node["omitted_fields"] = len(names) - max_fields
            return node

        if isinstance(value, np.ndarray):
            node = {
                "kind": "array", "shape": list(value.shape),
                "dtype": str(value.dtype), "size": int(value.size)
            }
            if not value.size:
                return node
            if value.dtype.names:
                names = list(value.dtype.names)
                node["kind"] = "structured_array"
                node["field_count"] = len(names)
                node["first_element_fields"] = {
                    name: walk(value[name].flat[0], depth + 1)
                    for name in names[:max_fields]
                }
                if len(names) > max_fields:
                    node["omitted_fields"] = len(names) - max_fields
            elif value.dtype.kind == "O":
                node["kind"] = "object_array"
                node["sample_only"] = "first element; remaining elements unexamined"
                node["first_element"] = walk(value.flat[0], depth + 1)
            return node

        if isinstance(value, dict):
            names = list(value)
            return {
                "kind": "mapping", "key_count": len(names),
                "fields": {str(k): walk(value[k], depth + 1)
                           for k in names[:max_fields]}
            }
        if isinstance(value, (list, tuple)):
            return {
                "kind": type(value).__name__, "length": len(value),
                "first_element": walk(value[0], depth + 1) if value else None
            }
        if np.isscalar(value):
            return {"kind": "scalar", "type": type(value).__name__}
        return {"kind": "other", "type": type(value).__name__}

    report = walk(obj, 0)
    report["reported_nodes"] = min(nodes_seen, max_nodes)
    report["limits"] = {
        "max_depth": max_depth,
        "max_fields": max_fields,
        "max_nodes": max_nodes
    }
    return report


def inspect_mat_file(
    path: Path, *, max_depth: int = 6, max_fields: int = 40,
    max_nodes: int = 400
) -> dict[str, Any]:
    """Inspect the named Data variable from a MATLAB v5-format file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    names = [name for name, _, _ in whosmat(str(path))]
    if "Data" not in names:
        raise ValueError(f"Expected 'Data' variable absent in {path.name}: {names}")
    value = loadmat(
        str(path), variable_names=["Data"], squeeze_me=False,
        struct_as_record=False, verify_compressed_data_integrity=True
    )["Data"]
    return {
        "filename": path.name, "top_level_variables": names,
        "inspected_variable": "Data",
        "data": structure_summary(
            value, max_depth=max_depth,
            max_fields=max_fields, max_nodes=max_nodes
        ),
        "notes": [
            "Object arrays sampled at first element only.",
            "No signal or numeric sample values are exported.",
            "Schema inspection does not establish trial completeness."
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "data/raw/essex_femoral_neck_loading"))
    parser.add_argument("--subjects", nargs="+", default=["subj04", "subj16"])
    parser.add_argument("--report", type=Path, default=Path(
        "outputs/essex_nested_mat_schema.json"))
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-fields", type=int, default=40)
    parser.add_argument("--max-nodes", type=int, default=400)
    args = parser.parse_args()

    if min(args.max_depth, args.max_fields, args.max_nodes) < 1:
        parser.error("All limits must be positive.")
    if not args.root.is_dir():
        parser.error(f"Data directory not found: {args.root}")

    files = list(args.root.rglob("*.mat"))
    index = {p.stem.casefold(): p for p in files}
    reports = []
    for subject in args.subjects:
        stem = Path(subject).stem.casefold()
        if stem not in index:
            parser.error(f"{stem}.mat not found under {args.root}")
        path = index[stem]
        try:
            result = inspect_mat_file(
                path, max_depth=args.max_depth,
                max_fields=args.max_fields, max_nodes=args.max_nodes
            )
        except (OSError, ValueError, TypeError, NotImplementedError) as exc:
            parser.exit(1, f"Failed to inspect {path}: {exc}\n")
        result["relative_path"] = path.relative_to(args.root).as_posix()
        reports.append(result)
        top = result["data"]
        node = top.get("first_element", {}) if top.get("kind") == "object_array" else top
        print(f"{path.name}: Data shape={top.get('shape')} | "
              f"metadata nodes={top['reported_nodes']}")
        print(f"  First-level Data fields: {list(node.get('fields', {}))}")

    report = {
        "inspected_files": len(reports), "mat_files_found": len(files),
        "results": reports,
        "purpose": "Metadata-only nested MATLAB schema inspection"
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report saved: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
