"""Inspect a ZIP archive without extracting files or loading dataset arrays."""

from __future__ import annotations

import argparse
import json
import re
import stat
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo


def member_issues(info: ZipInfo) -> list[str]:
    """Flag unsafe member paths, symbolic links, and extreme compression."""
    name = info.filename
    path = PurePosixPath(name.replace("\\", "/"))
    issues = []
    if not name or name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name):
        issues.append("absolute_or_drive_path")
    if ".." in path.parts:
        issues.append("path_traversal")
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_IFMT(mode) == stat.S_IFLNK:
        issues.append("symbolic_link")
    if info.file_size and info.compress_size and info.file_size / info.compress_size > 1000:
        issues.append("unusually_high_compression_ratio")
    return issues


def inventory_zip(archive: Path) -> dict:
    """Inventory ZIP members, sizes, and warnings without extraction."""
    with ZipFile(archive) as handle:
        members = []
        seen = set()
        for entry in handle.infolist():
            issues = member_issues(entry)
            normalized = entry.filename.replace("\\", "/").casefold()
            if normalized in seen:
                issues.append("duplicate_case_insensitive_path")
            seen.add(normalized)
            members.append({
                "name": entry.filename,
                "compressed_bytes": entry.compress_size,
                "uncompressed_bytes": entry.file_size,
                "is_directory": entry.is_dir(),
                "issues": issues,
            })
        kinds = Counter(PurePosixPath(m["name"].replace("\\", "/")).suffix.lower() or "[none]"
                        for m in members if not m["is_directory"])
        return {
            "archive_name": archive.name,
            "member_count": len(members),
            "mat_file_count": kinds.get(".mat", 0),
            "total_uncompressed_bytes": sum(m["uncompressed_bytes"] for m in members),
            "suffix_counts": dict(sorted(kinds.items())),
            "issues_count": dict(Counter(x for m in members for x in m["issues"])),
            "members": members,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("outputs/essex_matlab_zip_inventory.json"))
    parser.add_argument("--show", type=int, default=20)
    args = parser.parse_args()
    try:
        if not args.archive.is_file():
            raise FileNotFoundError(args.archive)
        summary = inventory_zip(args.archive)
        print(f"Members: {summary['member_count']}; MAT files: {summary['mat_file_count']}")
        print(f"Uncompressed: {summary['total_uncompressed_bytes'] / 1048576:.2f} MiB")
        print(f"Types: {summary['suffix_counts']}")
        print(f"Safety warnings: {summary['issues_count'] or 'none'}")
        for entry in summary["members"][:max(0, args.show)]:
            print(f"  {entry['name']} ({entry['uncompressed_bytes'] / 1048576:.2f} MiB)")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.report}")
        return 2 if summary["issues_count"] else 0
    except (OSError, BadZipFile, ValueError) as exc:
        parser.exit(1, f"Archive inspection failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
