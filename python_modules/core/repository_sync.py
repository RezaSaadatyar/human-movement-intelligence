# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Generic, non-destructive Git repository synchronisation for notebooks."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


def _git(*args: str, cwd: Path) -> str:
    """Run Git without a shell and stop on any non-zero exit status."""
    process = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"git {args[0]} failed: {detail}")
    return process.stdout.strip()


def sync_repository(
    url: str, branch: str = "main", *, parent: str | Path = "/content"
) -> Path:
    """Clone or fast-forward a clean checkout; never discard local changes.

    Reuses ignored local datasets. Stops on dirty working trees, wrong
    origins/branches, diverged commits or stale imported Python modules.
    """
    parent = Path(parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    name = url.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]
    if not name or name in {".", ".."}:
        raise ValueError("Repository URL must end in a valid repository name.")
    path = parent / name
    if not path.exists():
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
            ancestry = subprocess.run(
                ["git", "merge-base", "--is-ancestor", local, remote],
                cwd=path, capture_output=True, check=False,
            )
            if ancestry.returncode != 0:
                raise RuntimeError(
                    "Local branch cannot fast-forward. Resolve divergence "
                    "manually; automatic merge and reset are disabled."
                )
            changed = _git("diff", "--name-only", local, remote, cwd=path)
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
                file.endswith(".py") for file in changed.splitlines()
            )
            helper_changed = (
                "python_modules/core/repository_sync.py"
                in changed.splitlines()
            )
            if helper_changed or (changed_python and imported):
                raise RuntimeError(
                    "Repository files updated successfully, but Python "
                    "modules in this runtime may be stale. Restart the "
                    "kernel and rerun the notebook from the first cell."
                )
    messages = {
        "CLONED": "Repository cloned.",
        "ALREADY CURRENT": "Repository up to date.",
        "FAST-FORWARDED": "Repository updated.",
    }
    print(messages[action])
    return path
