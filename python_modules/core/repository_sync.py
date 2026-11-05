# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Safe Git updates and project-scoped Python reload for research notebooks.

A Git fast-forward updates ALL tracked source and report files, including
CSV, JSON, Excel, TXT and MATLAB files when they are tracked. Ignored raw
data and local processed caches are preserved. In-memory DataFrames/arrays
must be explicitly read again after their source files change.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Mapping


# ================== 01. Safe Research Repository Synchronisation =============

# ------------------------ 1. Execute a Checked Git Command ------------------
def _git(*args: str, cwd: Path) -> str:
    """Run Git without shell expansion or destructive reset operations."""
    process = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"git {args[0]} failed: {detail}")
    return process.stdout.strip()


# ---------------------- 2. Clone or Fast-Forward Tracked Files ---------------
def sync_repository(
    url: str, branch: str = "main", *, parent: str | Path = "/content",
    sync_mode: str = "update", allow_stale_modules: bool = False,
) -> Path:
    """Synchronise a clean checkout without overwriting local work.

    sync_mode: update (default), check (fetch only), or off (reuse checkout).
    Fast-forward always updates all tracked files, not only Python files.
    Ignored raw data and intermediate caches are never deleted or replaced.
    The default direct API remains conservative for non-notebook callers.
    """
    if sync_mode not in {"update", "check", "off"}:
        raise ValueError("sync_mode must be 'update', 'check' or 'off'.")

    parent = Path(parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    name = url.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]
    if not name or name in {".", ".."}:
        raise ValueError("Repository URL must end in a valid repository name.")
    path = parent / name

    if not path.exists():
        if sync_mode == "off":
            raise FileNotFoundError(
                "The checkout is absent and sync_mode='off' prohibits cloning."
            )
        _git(
            "clone", "--branch", branch, "--single-branch", url, str(path),
            cwd=parent,
        )
        action = "CLONED"
    else:
        if not path.is_dir() or not (path / ".git").exists():
            raise RuntimeError("Existing destination is not a Git checkout.")
        root = Path(_git("rev-parse", "--show-toplevel", cwd=path)).resolve()
        if root != path:
            raise RuntimeError("Repository checkout root does not match the path.")
        origin = _git("remote", "get-url", "origin", cwd=path)
        normalise = lambda value: value.rstrip("/").removesuffix(".git")
        if normalise(origin) != normalise(url):
            raise RuntimeError("Configured repository differs from Git origin.")
        active = _git("symbolic-ref", "--quiet", "--short", "HEAD", cwd=path)
        if active != branch:
            raise RuntimeError(f"Expected branch {branch!r}, found {active!r}.")

        if sync_mode == "off":
            action = "SKIPPED"
        else:
            if sync_mode == "update":
                dirty = _git(
                    "status", "--porcelain", "--untracked-files=normal", cwd=path
                )
                if dirty:
                    raise RuntimeError(
                        "Local tracked/untracked changes exist. Save them before "
                        "synchronising; no local files were changed."
                    )

            _git("fetch", "--no-tags", "origin", branch, cwd=path)
            local = _git("rev-parse", "HEAD", cwd=path)
            remote = _git("rev-parse", "FETCH_HEAD", cwd=path)
            action = "ALREADY CURRENT"

            if local != remote:
                ancestor = subprocess.run(
                    ["git", "merge-base", "--is-ancestor", local, remote],
                    cwd=path, capture_output=True, check=False,
                )
                if ancestor.returncode != 0:
                    raise RuntimeError(
                        "Local branch cannot fast-forward. Resolve divergence "
                        "manually; automatic merge and reset are disabled."
                    )
                if sync_mode == "check":
                    action = "UPDATE AVAILABLE"
                else:
                    changed = _git("diff", "--name-only", local, remote, cwd=path)
                    changed_files = changed.splitlines()
                    imported = [
                        name for name, module in tuple(sys.modules.items())
                        if getattr(module, "__file__", None)
                        and Path(module.__file__).resolve().is_relative_to(path)
                        and name not in {
                            __name__, "python_modules", "python_modules.core"
                        }
                    ]
                    _git("merge", "--ff-only", "FETCH_HEAD", cwd=path)
                    importlib.invalidate_caches()
                    action = "FAST-FORWARDED"
                    changed_python = any(
                        filename.endswith(".py") for filename in changed_files
                    )
                    helper_changed = (
                        "python_modules/core/repository_sync.py" in changed_files
                    )
                    if (helper_changed or changed_python and imported) and not (
                        allow_stale_modules
                    ):
                        raise RuntimeError(
                            "Repository files updated successfully, but loaded "
                            "Python modules may be stale. Reinitialise the "
                            "notebook environment before using changed code."
                        )

    messages = {
        "CLONED": "Repository cloned.",
        "ALREADY CURRENT": "Repository up to date.",
        "FAST-FORWARDED": "Repository updated.",
        "UPDATE AVAILABLE": "Repository update available; no files changed.",
        "SKIPPED": "Repository sync skipped; local checkout reused.",
    }
    print(messages[action])
    return path


# ================== 02. Scoped Python Module Auto-Refresh ====================

# --------------------- 1. Find Loaded Research Python Modules ----------------
def _loaded_research_modules(root: Path, study: str) -> set[str]:
    """Only target existing modules from the shared or requested study code."""
    locations = (
        root / "python_modules" / "core",
        root / study / "code" / "src",
    )
    found = set()
    for name, module in tuple(sys.modules.items()):
        if name == __name__:
            continue
        source = getattr(module, "__file__", None)
        if not source:
            continue
        try:
            path = Path(source).resolve()
            if path.suffix == ".py" and any(
                path.is_relative_to(directory) for directory in locations
            ):
                found.add(name)
        except (OSError, TypeError, ValueError):
            continue
    return found


# ----------------- 2. Activate IPython Autoreload Inside Helper --------------
def _enable_project_autoreload(root: Path, study: str) -> bool:
    """Register already imported and subsequently imported project modules.

    IPython's autoreload replaces many 'from module import function'
    references on later cell executions. Third-party packages are excluded
    by using explicit mode 1 rather than global mode 2.
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False

    shell = get_ipython()
    if shell is None or not hasattr(shell, "run_line_magic"):
        return False

    if "autoreload" not in shell.extension_manager.loaded:
        shell.run_line_magic("load_ext", "autoreload")
    shell.run_line_magic("autoreload", "1")

    previous = getattr(shell, "_hmi_project_autoreload_hook", None)
    if previous is not None:
        shell.events.unregister("post_run_cell", previous)

    registered: set[str] = set()

    def register_project_modules(_result=None) -> None:
        for module_name in sorted(_loaded_research_modules(root, study)):
            if module_name not in registered:
                shell.run_line_magic("aimport", module_name)
                registered.add(module_name)

    register_project_modules()
    shell.events.register("post_run_cell", register_project_modules)
    shell._hmi_project_autoreload_hook = register_project_modules

    # Refresh already imported project functions immediately after a Git
    # update; subsequent cells refresh registered modules automatically.
    shell.run_line_magic("autoreload", "now")
    return True


# ================= 03. Optional Raw Data Integrity Verification ==============

# ---------------------- 1. Verify Explicit Trusted File Hashes ---------------
def _verify_local_data(
    root: Path, expected_hashes: Mapping[str, str] | None,
) -> None:
    """Verify selected local files; never download or overwrite raw data."""
    if not expected_hashes:
        raise ValueError(
            "data_mode='verify' requires expected_data_sha256: "
            "{'relative/data/file': 'trusted_sha256'}."
        )
    for relative, expected in expected_hashes.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Data verification path must stay in the repository.")
        if not path.is_file():
            raise FileNotFoundError(f"Data file to verify is missing: {path}")
        expected = str(expected).lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Invalid trusted SHA256 for {relative}.")
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(
                f"SHA256 mismatch for {relative}. Local data were not modified."
            )
    print(f"Verified local data files: {len(expected_hashes)}.")


# =================== 04. Complete Notebook Repository Setup ==================

# --------------------- 1. Connect, Synchronise and Refresh -------------------
def prepare_notebook_repository(
    url: str, study: str, branch: str = "main", *,
    parent: str | Path = "/content",
    sync_mode: str = "update",
    module_refresh: str = "auto",
    data_mode: str = "reuse",
    expected_data_sha256: Mapping[str, str] | None = None,
) -> Path:
    """Prepare a research notebook with safe defaults and minimal cell code.

    Defaults: update tracked GitHub code AND committed CSV/JSON/TXT/XLSX/MAT
    outputs, autoreload project Python modules, reuse ignored raw datasets.

    'check' inspects upstream updates without applying them; 'off' avoids
    fetching. 'verify' checks explicit trusted local data SHA256 values.
    Data re-download must use the study-specific pinned source adapter.
    """
    if module_refresh not in {"auto", "off"}:
        raise ValueError("module_refresh must be 'auto' or 'off'.")
    if data_mode not in {"reuse", "verify"}:
        raise ValueError("data_mode must be 'reuse' or 'verify'.")
    if not isinstance(study, str) or not study or Path(study).name != study:
        raise ValueError("study must be a direct repository folder name.")

    # Use the project-scoped hook if already running in a notebook.
    parent_dir = Path(parent).expanduser().resolve()
    repository_name = url.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]
    expected_root = parent_dir / repository_name
    autoreload_active = False
    if module_refresh == "auto" and expected_root.is_dir():
        autoreload_active = _enable_project_autoreload(expected_root, study)

    root = sync_repository(
        url, branch, parent=parent, sync_mode=sync_mode,
        allow_stale_modules=autoreload_active,
    )
    if not (root / study / "code" / "src").is_dir():
        raise FileNotFoundError(f"Study Python directory is missing: {study!r}")

    # Register module search paths without repeatedly reordering sys.path.
    for directory in (root / "python_modules", root / study / "code" / "src"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

    if module_refresh == "auto":
        autoreload_active = _enable_project_autoreload(root, study)
        if not autoreload_active:
            print("IPython autoreload unavailable; restart for incompatible changes.")
    if data_mode == "verify":
        _verify_local_data(root, expected_data_sha256)

    # Re-reading result CSV/JSON/Excel/etc is the responsibility of the next
    # notebook evidence-loading cell; in-memory objects are not overwritten.
    return root
