"""Inspect MATLAB file headers and top-level variables without training models.

MAT v5 is summarised via scipy.io.whosmat; MAT v7.3 via h5py.
The summary is intentionally shallow: nested fields are validated later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_file(path: Path) -> dict:
    """Describe MATLAB variable names, types and shapes without rewriting data."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install data dependencies: pip install -r requirements-data.txt") from exc
    try:
        from scipy.io import whosmat
    except ImportError as exc:
        raise RuntimeError("Install data dependencies: pip install -r requirements-data.txt") from exc

    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as stream:
            variables = [
                {"name": name, "shape": list(obj.shape) if hasattr(obj, "shape") else None,
                 "type": "HDF5 dataset" if hasattr(obj, "shape") else "HDF5 group"}
                for name, obj in stream.items()
            ]
        fmt = "MAT v7.3 / HDF5"
    else:
        variables = [
            {"name": name, "shape": list(shape), "type": datatype}
            for name, shape, datatype in whosmat(str(path))
        ]
        fmt = "MAT v5 or compatible"
    return {"filename": path.name, "bytes": path.stat().st_size, "format": fmt, "variables": variables}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/essex_femoral_neck_loading"))
    parser.add_argument("--report", type=Path, default=Path("outputs/mat_header_inventory.json"))
    parser.add_argument("--max-files", type=int, default=5)
    args = parser.parse_args()
    if args.max_files <= 0:
        parser.error("--max-files must be positive.")
    if not args.root.is_dir():
        parser.exit(1, f"Missing data folder: {args.root}\n")
    paths = sorted(args.root.rglob("*.mat"))
    if not paths:
        parser.exit(1, f"No MATLAB files (.mat) found in {args.root}\n")
    records = []
    for path in paths[:args.max_files]:
        entry = inspect_file(path)
        entry["relative_path"] = str(path.relative_to(args.root))
        records.append(entry)
        print(f"{entry['relative_path']}: {entry['format']} | {[v['name'] for v in entry['variables']]}")
    report = {
        "root": str(args.root), "total_mat_files_found": len(paths),
        "files_inspected": len(records), "files": records,
        "note": "MAT header inspection only; nested trial structures and units are not yet validated.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Header inventory saved: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
