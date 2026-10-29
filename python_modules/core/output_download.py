# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Package notebook-specific Colab output files for manual download.

This module never calls GitHub, modifies source outputs, or assumes that
browser-based download methods work in VS Code's Colab connection.
"""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import re


REPO_ROOT = Path(__file__).resolve().parents[2]


# ========================= Output Download ===================================
def download_outputs_from_colab(folder, stage):
    """Build a ZIP containing output files belonging to one notebook stage.

    Args:
        folder (str | Path): Project-level output/ directory within the
            repository, supplied as a relative or absolute path.
        stage (str | int): Notebook number, e.g. "01" or 20.

    Returns:
        Path: ZIP archive path for manual download via Colab Contents in
            VS Code (or via the file browser in web Colab).

    Note:
        This selects names starting with "<stage>_". Any files from previous
        runs sharing that prefix are included; the function does not verify
        that they were created during the current run.
    """
    if (
        isinstance(stage, bool)
        or re.fullmatch(r"[0-9]{1,2}", str(stage)) is None
        or not 1 <= int(stage) <= 99
    ):
        raise ValueError("stage must be a notebook number from 01 to 99.")

    stage_id = f"{int(stage):02d}"
    root = REPO_ROOT.resolve()
    requested = Path(folder)
    output_dir = (
        requested if requested.is_absolute() else root / requested
    ).resolve()

    if (
        output_dir.name != "output"
        or output_dir.parent.parent != root
        or not output_dir.is_dir()
    ):
        raise ValueError(
            "folder must be an existing <project>/output directory "
            "inside this repository."
        )

    selected = sorted(
        path for path in output_dir.rglob(f"{stage_id}_*")
        if path.is_file()
        and not path.is_symlink()
        and not any(
            part.startswith(".")
            for part in path.relative_to(output_dir).parts
        )
    )
    if not selected:
        raise FileNotFoundError(
            f"No files beginning with {stage_id}_ in {output_dir}"
        )

    # Keep the downloadable ZIP outside the repository, never in Git.
    download_dir = Path("/content") if Path("/content").is_dir() else root.parent
    archive_path = (
        download_dir / f"{output_dir.parent.name}_Stage_{stage_id}_Outputs.zip"
    )
    with ZipFile(
        archive_path, mode="w", compression=ZIP_DEFLATED, allowZip64=True
    ) as archive:
        for path in selected:
            archive.write(path, arcname=path.relative_to(output_dir).as_posix())

    print("\n-" * 72)
    print(f"Notebook stage: {stage_id}")
    print(f"Files packaged: {len(selected)}")
    for path in selected:
        print("  +", path.relative_to(output_dir).as_posix())
    print(f"Download ZIP from Colab Contents: {archive_path}")
    print("GitHub Commit/Push performed: NO")
    # return archive_path
