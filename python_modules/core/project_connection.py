"""Connect local Jupyter, VS Code and Google Colab to a research checkout.

This module configures Python import paths. When explicitly given a public
GitHub URL and no matching working checkout, it can make a local clone.
It does not authenticate, push, pull, mount Google Drive or execute notebooks.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


def _find_repository(start: Path) -> Path | None:
    """Find an ancestor containing the approved shared Python package."""
    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (
            (candidate / "README.md").is_file()
            and (candidate / "python_modules" / "core" / "__init__.py").is_file()
        ):
            return candidate
    return None


def _github_repository(url: str) -> tuple[str, str]:
    """Validate a public GitHub HTTPS clone URL and derive its folder name."""
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use a plain https://github.com/OWNER/REPO URL.")

    match = re.fullmatch(
        r"/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?",
        parsed.path,
    )
    if match is None:
        raise ValueError("Expected https://github.com/OWNER/REPO.")

    owner, name = match.groups()
    if name in {".", ".."}:
        raise ValueError("Invalid GitHub repository name.")
    return f"https://github.com/{owner}/{name}.git", name


def connect_project(
    study: str | None = None,
    *,
    project_root: str | Path | None = None,
    github_url: str | None = None,
    clone_parent: str | Path | None = None,
) -> Path:
    """Locate or clone a research repository and configure Python imports.

    Args:
        study: Optional subproject directory directly under the repository.
        project_root: Optional explicit local checkout path (or a path in it).
        github_url: Optional public GitHub URL. Used for cloning only when
            a checkout cannot be located from the working directory.
        clone_parent: Parent folder for an optional clone (defaults to cwd).

    Returns:
        Path: Resolved repository root, suitable for constructing data paths.

    Notes:
        - Local VS Code/Jupyter: discovers a checkout from the working
          directory, or from this imported module when no URL is supplied.
        - Colab: clone the repository once before importing this module,
          unless it is already importable through another checkout.
        - GitHub cloning is explicit, uses git without a shell, and never
          modifies an existing checkout.
        - Does not install dependencies or change model/device/plot settings.
    """
    if project_root is not None:
        root = _find_repository(Path(project_root))
        if root is None:
            raise FileNotFoundError(
                f"No supported research repository at {project_root!s}."
            )
    else:
        root = _find_repository(Path.cwd())
        if root is None and github_url is not None:
            clone_url, repository_name = _github_repository(github_url)
            parent = Path(clone_parent or Path.cwd()).expanduser().resolve()
            if not parent.is_dir():
                raise FileNotFoundError(f"Clone parent does not exist: {parent}")
            target = parent / repository_name
            if target.exists():
                root = _find_repository(target)
                if root != target:
                    raise FileExistsError(
                        f"Clone target exists but is not a valid checkout: {target}"
                    )
            else:
                if shutil.which("git") is None:
                    raise RuntimeError("Git is required to clone the repository.")
                subprocess.run(
                    ["git", "clone", "--depth", "1", clone_url, str(target)],
                    check=True,
                )
                root = _find_repository(target)
                if root != target:
                    raise RuntimeError(
                        "Cloned checkout does not contain the expected structure."
                    )
        if root is None:
            root = _find_repository(Path(__file__))

    if root is None:
        raise FileNotFoundError(
            "Research repository not found. Open the local checkout or clone "
            "the repository from GitHub before importing this module."
        )

    module_dir = root / "python_modules"
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    if study is not None:
        if not isinstance(study, str) or study in {"", ".", ".."}:
            raise ValueError("study must be a non-empty subproject folder name.")
        if Path(study).name != study or "/" in study or "\\" in study:
            raise ValueError("study must be one folder name, not a file path.")
        study_src = root / study / "code" / "src"
        if not study_src.is_dir():
            raise FileNotFoundError(
                f"Study source directory was not found: {study_src}"
            )
        if str(study_src) not in sys.path:
            sys.path.insert(0, str(study_src))

    environment = (
        "Google Colab" if "google.colab" in sys.modules
        or "COLAB_RELEASE_TAG" in os.environ else "Local Jupyter / VS Code"
    )
    print(f"Environment: {environment}")
    print(f"Research repository: {root}")
    if study is not None:
        print(f"Study: {study}")
    print("Python module paths: configured")
    return root
