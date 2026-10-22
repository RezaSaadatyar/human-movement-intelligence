"""Set up repository imports for notebooks running in Google Colab."""

import sys
from pathlib import Path


def setup_colab(study=None):
    """Find the cloned repository and register its Python source directories.

    Call after cloning the repository and changing into its root directory.

    Args:
        study (str | None): Optional direct subproject directory name.

    Returns:
        pathlib.Path: Absolute repository root path.
    """
    root = Path.cwd().resolve()
    while not (
        (root / "requirements.txt").is_file()
        and (root / "python_modules" / "core" / "__init__.py").is_file()
    ):
        if root.parent == root:
            raise FileNotFoundError(
                "Repository not found. Clone it and change into its directory."
            )
        root = root.parent

    directories = [root / "python_modules"]
    if study is not None:
        source = root / study / "code" / "src"
        if not source.is_dir():
            raise FileNotFoundError(
                f"Study source directory not found: {source}"
            )
        directories.append(source)

    for directory in directories:
        path = str(directory)
        if path not in sys.path:
            sys.path.insert(0, path)

    return root
