# ==================== Presented by: Reza Saadatyar (2026) ====================

"""Safely flatten retired consolidation output folders without data loss.

Only old project-specific notebook consolidation directories are considered.
This utility never rewrites an existing output or touches source datasets.
"""

from hashlib import sha256
from pathlib import Path


# ================== 01. Legacy Consolidation Output Migration =================

_LEGACY_DIRS = {"01_source_and_trial", "02_matlab_decoding"}
_ORIGINAL_NAME_COLLISIONS = {
    "01_femoral_strain_representation.csv",
    "01_figshare_file_manifest.csv",
    "01_matlab_archive_inventory.csv",
    "01_matlab_participant_inventory.csv",
    "01_matlab_record_schema_audit.csv",
}


def _file_sha256(path):
    """Compute a byte-level digest for comparing two existing output files."""
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_legacy_output_folders(output_dir):
    """Move historic nested outputs to a flat project output directory.

    Exact duplicates are kept once, at their top-level location. Different
    files with the same name are preserved using a source-directory suffix.
    Only the two known historical folders are traversed, and only empty
    directories are subsequently removed. Existing destination bytes are
    never overwritten.

    Returns:
        dict: Counts of moved, identical, and collision-renamed files.
    """
    output_dir = Path(output_dir)
    if output_dir.name != "output" or not output_dir.is_dir():
        raise ValueError("Expected the existing project output/ directory.")
    if output_dir.is_symlink():
        raise ValueError("Output root must not be a symbolic link.")

    legacy_root = output_dir / "_consolidation_drafts"
    counts = {"moved": 0, "identical": 0, "renamed": 0}
    if not legacy_root.exists() and not legacy_root.is_symlink():
        return counts
    if legacy_root.is_symlink() or not legacy_root.is_dir():
        raise ValueError("Unexpected legacy output directory type.")

    all_members = sorted(
        legacy_root.rglob("*"),
        key=lambda path: str(path.relative_to(legacy_root)),
    )
    for item in all_members:
        if item.is_symlink():
            raise ValueError(f"Symlink in legacy outputs: {item}")
        parts = item.relative_to(legacy_root).parts
        if not parts or parts[0] not in _LEGACY_DIRS:
            raise ValueError(f"Unknown legacy output location: {item}")
        if item.is_dir():
            if len(parts) != 1:
                raise ValueError(f"Unexpected nested output directory: {item}")
            continue
        if not item.is_file() or len(parts) != 2:
            raise ValueError(f"Unexpected legacy output item: {item}")
        if not item.name.startswith(("01_", "02_")):
            raise ValueError(f"Unexpected output filename: {item.name}")

    for item in all_members:
        if not item.is_file():
            continue
        folder = item.relative_to(legacy_root).parts[0]
        filename = item.name
        if folder == "01_source_and_trial" and filename in _ORIGINAL_NAME_COLLISIONS:
            filename = "01_dataset_" + filename[3:]

        destination = output_dir / filename
        if destination.exists():
            if destination.is_symlink() or not destination.is_file():
                raise ValueError(f"Unsafe destination: {destination}")
            if _file_sha256(item) == _file_sha256(destination):
                item.unlink()
                counts["identical"] += 1
                continue

            original = Path(filename)
            suffix = f"__from_{folder}"
            version = 1
            while True:
                postfix = suffix if version == 1 else f"{suffix}_{version}"
                candidate = output_dir / (
                    f"{original.stem}{postfix}{original.suffix}"
                )
                if not candidate.exists() and not candidate.is_symlink():
                    destination = candidate
                    counts["renamed"] += 1
                    break
                if candidate.is_symlink() or not candidate.is_file():
                    raise ValueError(f"Unsafe destination: {candidate}")
                if _file_sha256(item) == _file_sha256(candidate):
                    item.unlink()
                    counts["identical"] += 1
                    destination = None
                    break
                version += 1
            if destination is None:
                continue
        item.rename(destination)
        counts["moved"] += 1

    for folder in sorted(legacy_root.iterdir()):
        if folder.is_dir():
            folder.rmdir()  # Raises if unexpected files remain.
    legacy_root.rmdir()
    return counts
