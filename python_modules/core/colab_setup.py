# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Set up repository imports for notebooks running in Google Colab."""

import sys
from pathlib import Path


# ======================== 01. Colab Environment Setup ========================
def setup_colab(
    study=None, *, root=None, sync_mode="update", module_refresh="auto",
    data_mode="reuse", expected_data_sha256=None,
):
    """Set up imports, safe Git updates and automatic cell-boundary refresh.

    Call after cloning the repository and changing into its root directory.

    Args:
        study (str | None): Optional direct subproject directory name.
        root: Optional checkout path; otherwise search from the current folder.
        sync_mode: 'update', 'check' (fetch only), or 'off' (no hook/network).
        module_refresh: 'auto' for compatible project modules, or 'off'.
        data_mode: 'reuse', 'verify', or explicit pinned-source 'refresh'.
        expected_data_sha256: Trusted file hashes for 'verify'.

    In IPython, updates happen before the next cell, never during a running
    cell. Existing models and unmanaged tables are not recomputed. The active
    commit and revision history are on shell._hmi_notebook_sync. Incompatible
    class, setup-controller and frozen-evidence changes require a fresh session.

    Returns:
        pathlib.Path: Absolute repository root path.
    """
    if sync_mode not in {"update", "check", "off"}:
        raise ValueError("sync_mode must be 'update', 'check' or 'off'.")
    if module_refresh not in {"auto", "off"}:
        raise ValueError("module_refresh must be 'auto' or 'off'.")
    if data_mode not in {"reuse", "verify", "refresh"}:
        raise ValueError("data_mode must be 'reuse', 'verify' or 'refresh'.")
    if study is not None and (
        not isinstance(study, str) or not study or study in {".", ".."}
        or Path(study).name != study
    ):
        raise ValueError("study must be a direct repository folder name.")
    root = Path(root or Path.cwd()).resolve()
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

    from .repository_sync import _configure_notebook_sync, _verify_local_data
    _configure_notebook_sync(root, study, sync_mode=sync_mode,
                             module_refresh=module_refresh)
    if data_mode == "verify":
        _verify_local_data(root, expected_data_sha256)
    elif data_mode == "refresh":
        if study != "biomechanical_load_estimation":
            raise ValueError("A pinned source adapter is available only for Essex.")
        from essex_figshare import ensure_pinned_matlab_archive
        ensure_pinned_matlab_archive(
            root / "data" / "raw" / "essex_femoral_neck_loading"
        )
    return root
