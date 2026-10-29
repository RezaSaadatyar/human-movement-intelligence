"""Review and publish new research outputs from a Git checkout.

Small new files can be committed to GitHub after explicit confirmation.
Files above the configured limit are bundled for manual download instead.
Neither existing Git-tracked files nor source outputs are overwritten.
"""

from datetime import datetime, timezone
from pathlib import Path
import math
import re
import subprocess
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
HARD_LIMIT_BYTES = 50_000_000

# These types are not safe to publish automatically, even when Git permits them.
BLOCKED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".hdf5",
    ".joblib", ".pkl", ".pickle", ".mat", ".npy", ".npz",
    ".safetensors", ".tflite", ".pem", ".key",
    ".zip", ".7z", ".tar", ".gz", ".sqlite", ".db",
}
SENSITIVE_NAMES = {
    "credentials.json", "secrets.json", "token.json",
    "private_key.json", "service_account.json",
}


def _git(*args):
    """Run a Git command within this repository, failing visibly on errors."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Git command failed: git {' '.join(args)}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


def sync_outputs(folder, max_mb=50):
    """Publish new eligible output files; package oversized files for download.

    Args:
        folder (str | Path): A project-level output/ path, relative to the
            repository root or an absolute path inside the repository.
        max_mb (int | float): Per-file limit in decimal MB. The hard upper
            bound is always 50,000,000 bytes, even if a larger value is given.

    Returns:
        dict: Selected/published filenames and any generated ZIP path.

    This function never modifies previously tracked files, runs experiments,
    deletes source outputs, or automatically publishes model binary files.
    """
    if (
        isinstance(max_mb, bool)
        or not isinstance(max_mb, (int, float))
        or not math.isfinite(max_mb)
        or max_mb <= 0
    ):
        raise ValueError("max_mb must be a positive finite number.")

    limit = min(int(max_mb * 1_000_000), HARD_LIMIT_BYTES)
    if limit < 1:
        raise ValueError("max_mb is too small.")

    supplied = Path(folder)
    output_dir = (
        supplied if supplied.is_absolute() else REPO_ROOT / supplied
    ).resolve()
    root = REPO_ROOT.resolve()

    # Limit this utility to <repository>/<project>/output.
    if (
        output_dir.name != "output"
        or output_dir.parent.parent != root
        or not output_dir.is_dir()
    ):
        raise ValueError(
            "folder must identify an existing project-level output/ "
            "directory inside human-movement-intelligence."
        )

    relative_dir = output_dir.relative_to(root).as_posix()

    # Git itself decides whether a file is new and non-ignored.
    untracked = set(filter(
        None,
        _git(
            "ls-files", "--others", "--exclude-standard", "-z",
            "--", relative_dir
        ).split("\0")
    ))

    publishable = []
    oversized = []
    skipped_count = 0

    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue

        relative_inside = path.relative_to(output_dir)
        relative = path.relative_to(root).as_posix()

        if any(part.startswith(".") for part in relative_inside.parts):
            skipped_count += 1
            continue

        size = path.stat().st_size
        if size > limit:
            oversized.append(path)
        elif (
            path.suffix.lower() in BLOCKED_SUFFIXES
            or path.name.lower() in SENSITIVE_NAMES
            or re.search(
                r"(?:password|secret|api[_-]?key|credential|private[_-]?key|token)",
                path.name,
                flags=re.IGNORECASE,
            )
        ):
            skipped_count += 1
        elif relative in untracked:
            publishable.append(relative)
        else:
            # Previously tracked, Git-ignored or already accounted for.
            skipped_count += 1

    archive = None
    if oversized:
        # Deliberately outside the repository and its Git working tree.
        download_dir = root.parent / "colab_output_downloads"
        download_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        archive = download_dir / f"{output_dir.parent.name}_{stamp}.zip"

        with zipfile.ZipFile(
            archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        ) as bundle:
            for path in oversized:
                bundle.write(path, arcname=path.relative_to(output_dir).as_posix())

    print(f"\nOutput folder: {relative_dir}")
    print(f"GitHub file size limit: {limit / 1_000_000:g} MB")
    print(f"New files eligible for GitHub: {len(publishable)}")
    print(f"Oversized files for download: {len(oversized)}")
    print(f"Tracked, ignored or blocked files: {skipped_count}")

    if archive:
        print(f"Download ZIP: {archive}")
        print("Download it manually from Colab's file browser in VS Code.")

    result = {
        "eligible_files": tuple(publishable),
        "published_files": (),
        "download_zip": str(archive) if archive else None,
    }
    if not publishable:
        print("No new eligible files to publish.")
        return result

    print("\nReview new files before publishing:")
    for name in publishable:
        print("  +", name)

    if _git("branch", "--show-current") != "main":
        raise RuntimeError("Only the main branch is supported.")
    if _git("diff", "--cached", "--name-only"):
        raise RuntimeError("Unrelated staged changes exist; no files added.")

    remote = _git("remote", "get-url", "origin")
    if not re.fullmatch(
        r"(?:https://github\.com/|git@github\.com:|"
        r"ssh://git@github\.com/)"
        r"RezaSaadatyar/human-movement-intelligence(?:\.git)?/?",
        remote,
    ):
        raise RuntimeError("Unexpected GitHub origin; refusing to push.")

    remote_line = _git("ls-remote", "origin", "refs/heads/main")
    if not remote_line:
        raise RuntimeError("Unable to locate origin/main.")
    remote_head = remote_line.split()[0]
    if _git("rev-parse", "HEAD") != remote_head:
        raise RuntimeError(
            "Colab main and GitHub main differ. Synchronise the "
            "checkout safely before trying again."
        )

    if input("\nType PUSH to commit and upload these files: ").strip() != "PUSH":
        print("GitHub publication cancelled. Source files are unchanged.")
        return result

    # Guard again against changes to the selected files during review.
    for name in publishable:
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size > limit
            or path.suffix.lower() in BLOCKED_SUFFIXES
        ):
            raise RuntimeError(f"File changed or exceeds the limit: {name}")

    for name in publishable:
        _git("add", "--", name)

    staged = set(filter(
        None, _git("diff", "--cached", "--name-only", "-z").split("\0")
    ))
    if staged != set(publishable):
        raise RuntimeError(
            "Unexpected staged files. Do not commit or push; "
            "inspect Git status first."
        )

    print(_git(
        "commit", "-m", f"Add {output_dir.parent.name} research outputs"
    ))
    print(_git("push", "origin", "main"))
    print("New eligible outputs published to GitHub.")
    result["published_files"] = tuple(publishable)
    return result
