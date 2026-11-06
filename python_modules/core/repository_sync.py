# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Safe Git updates and project-scoped Python reload for research notebooks.

A Git fast-forward updates ALL tracked source and report files, including
CSV, JSON, Excel, TXT and MATLAB files when they are tracked. Ignored raw
data and local processed caches are preserved. In-memory DataFrames/arrays
refresh only through explicitly registered pure loaders; other in-memory
objects must be explicitly read again after their source files change.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Mapping


# ================== 01. Safe Research Repository Synchronisation =============

# ------------------------ 1. Execute a Checked Git Command -------------------
def _git(*args: str, cwd: Path) -> str:
    """Run Git without shell expansion or destructive reset operations."""
    process = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=False, timeout=30
    )
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"git {args[0]} failed: {detail}")
    return process.stdout.strip()


# ---------------------- 2. Clone or Fast-Forward Tracked Files ---------------
def sync_repository(
    url: str, branch: str = "main", *, parent: str | Path = "/content",
    sync_mode: str = "update", allow_stale_modules: bool = False,
    quiet: bool = False, before_update=None, checkout_path=None,
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
    path = Path(checkout_path).resolve() if checkout_path else parent / name

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
                    if sync_mode == "update":
                        dirty = _git(
                            "status", "--porcelain", "--untracked-files=normal", cwd=path
                        )
                        if dirty:
                            raise RuntimeError(
                                "Local tracked/untracked changes exist. Save them before "
                                "synchronising; no local files were changed."
                            )

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
                    if before_update is not None:
                        before_update(changed_files, remote)
                    _git("merge", "--ff-only", "--no-overwrite-ignore",
                         remote, cwd=path)
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
    if not quiet or action in {"CLONED", "FAST-FORWARDED"}:
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


# =================== 04. Notebook Cell Boundary Synchronisation =============


def _get_shell():
    """Return the active IPython shell, if available."""
    try:
        from IPython import get_ipython
        return get_ipython()
    except ImportError:
        return None


def _class_definitions(source):
    tree = ast.parse(source)
    return [ast.dump(node, include_attributes=False)
            for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


class _NotebookSync(ast.NodeTransformer):
    """Sync before execution and reject the cell if refresh cannot complete.

    IPython swallows event callback exceptions. InputRejected from an AST
    transformer instead prevents the scientific cell body from executing.
    Nothing runs in the background or during a training/evaluation cell.
    """

    def __init__(self, root, study, shell, sync_mode, module_refresh):
        self.root, self.study, self.shell = root, study, shell
        self.sync_mode, self.module_refresh = sync_mode, module_refresh
        self.url = _git("remote", "get-url", "origin", cwd=root)
        self.branch = _git("symbolic-ref", "--short", "HEAD", cwd=root)
        self.commit = _git("rev-parse", "HEAD", cwd=root)
        self.history = [self.commit]
        self.error = None
        self.managed, self.imports, self.old_objects = {}, {}, {}

    def _modules(self):
        names = _loaded_research_modules(self.root, self.study or "")
        # Do not reload the coordinator while one of its methods is running.
        return {name: sys.modules[name] for name in names
                if not name.endswith(("repository_sync", "colab_setup"))}

    def _preflight(self, changed, remote):
        # Changing the synchroniser itself needs a newly configured session,
        # not a mixture of old controller methods and new helper functions.
        helpers = {"python_modules/core/repository_sync.py",
                   "python_modules/core/colab_setup.py"}
        if helpers.intersection(changed):
            raise RuntimeError(
                "The setup implementation changed upstream. Open a fresh "
                "checkout/session to use the new coordinator."
            )
        for name, module in self._modules().items():
            path = Path(module.__file__).resolve()
            relative = path.relative_to(self.root).as_posix()
            if relative not in changed:
                continue
            if self.module_refresh == "off":
                raise RuntimeError(f"Loaded module changed with refresh off: {name}")
            old = path.read_text(encoding="utf-8")
            new = _git("show", f"{remote}:{relative}", cwd=self.root)
            compile(new, str(path), "exec")
            if _class_definitions(old) != _class_definitions(new):
                raise RuntimeError(
                    f"Class definition changed in {name}; live objects cannot "
                    "be safely rebuilt automatically. Use a fresh session."
                )
            old_tree, new_tree = ast.parse(old), ast.parse(new)
            old_names = {n.name for n in old_tree.body
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            new_names = {n.name for n in new_tree.body
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if old_names - new_names:
                raise RuntimeError(f"Functions removed from loaded module: {name}")

    def refresh(self, *, quiet=True):
        if self.error:
            raise RuntimeError(self.error)
        previous = self.commit
        try:
            sync_repository(
                self.url, self.branch, parent=self.root.parent,
                sync_mode=self.sync_mode, allow_stale_modules=True,
                quiet=quiet, before_update=self._preflight,
                checkout_path=self.root,
            )
            current = _git("rev-parse", "HEAD", cwd=self.root)
            if current != previous:
                changed = _git("diff", "--name-only", previous, current,
                               cwd=self.root).splitlines()
                modules = self._modules()
                if any(Path(m.__file__).resolve().relative_to(
                    self.root).as_posix() in changed for m in modules.values()):
                    self._reload_modules(modules)
                self._refresh_managed(changed)
                self.commit = current
                self.history.append(current)
        except Exception as exc:
            if _git("rev-parse", "HEAD", cwd=self.root) != previous:
                self.error = f"Refresh incomplete: {exc}. Use a fresh session."
            raise

    def _reload_modules(self, modules):
        from IPython.extensions.autoreload import superreload

        ordered, visiting, visited = [], set(), set()

        def visit(name):
            if name in visited:
                return
            if name in visiting:
                raise RuntimeError("Cyclic loaded-module dependency: " + name)
            visiting.add(name)
            source = Path(modules[name].__file__).read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source)):
                dependencies = []
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    dependencies = [node.module]
                elif isinstance(node, ast.Import):
                    dependencies = [item.name for item in node.names]
                for dependency in dependencies:
                    if dependency in modules and dependency != name:
                        visit(dependency)
            for value in tuple(vars(modules[name]).values()):
                dependency = getattr(value, "__module__", None)
                if isinstance(value, type(sys)):
                    dependency = value.__name__
                if dependency in modules and dependency != name:
                    visit(dependency)
            visiting.remove(name)
            visited.add(name)
            ordered.append(name)

        for name in sorted(modules):
            visit(name)
        bindings = {}
        for binding, (name, attribute) in self.imports.items():
            if name in modules and hasattr(modules[name], attribute):
                old = getattr(modules[name], attribute)
                if self.shell.user_ns.get(binding) is old:
                    bindings[binding] = (name, attribute)
        for name in ordered:
            module = modules[name]
            cached = getattr(module, "__cached__", None)
            if cached:
                # Equal-size edits can share the same filesystem timestamp.
                Path(cached).unlink(missing_ok=True)
            superreload(module, old_objects=self.old_objects)
        for binding, (name, attribute) in bindings.items():
            self.shell.user_ns[binding] = getattr(modules[name], attribute)

    def _refresh_managed(self, changed):
        replacements = {}
        for name, (path, loader, value) in self.managed.items():
            if path.relative_to(self.root).as_posix() in changed:
                if self.shell.user_ns.get(name) is not value:
                    raise RuntimeError(f"Managed binding replaced locally: {name}")
                replacements[name] = loader(path)
        for name, value in replacements.items():
            path, loader, _ = self.managed[name]
            self.managed[name] = (path, loader, value)
            self.shell.user_ns[name] = value

    def visit_Module(self, node):
        from IPython.core.error import InputRejected
        # A single recovery command remains executable when a guard rejects
        # cells. It only detaches the hook, without resetting or changing Git.
        if (len(node.body) == 1 and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Call)):
            call = node.body[0].value
            if (isinstance(call.func, ast.Name)
                    and call.func.id == "_hmi_stop_sync"
                    and not call.args and not call.keywords):
                return node
        try:
            self.refresh()
        except Exception as exc:
            raise InputRejected(f"Repository synchronisation: {exc}") from exc
        self._record_imports(node)
        return node

    def _record_imports(self, node):
        for statement in node.body:
            if isinstance(statement, ast.ImportFrom) and statement.level == 0:
                for item in statement.names:
                    if item.name != "*":
                        self.imports[item.asname or item.name] = (
                            statement.module, item.name)

    def record_completed_cell(self, result):
        # Includes imports later in the very cell that first called setup.
        try:
            source = self.shell.transform_cell(result.info.raw_cell)
            self._record_imports(ast.parse(source))
        except (AttributeError, SyntaxError):
            pass


def _configure_notebook_sync(root, study, *, sync_mode, module_refresh):
    shell = _get_shell()
    previous = getattr(shell, "_hmi_notebook_sync", None) if shell else None
    if previous is not None:
        if previous.error and sync_mode != "off":
            raise RuntimeError(previous.error)
        shell.ast_transformers.remove(previous)
        shell.events.unregister("post_run_cell", previous.record_completed_cell)
        del shell._hmi_notebook_sync
    legacy = getattr(shell, "_hmi_project_autoreload_hook", None) if shell else None
    if legacy is not None:
        shell.events.unregister("post_run_cell", legacy)
        del shell._hmi_project_autoreload_hook
        for name in sorted(_loaded_research_modules(root, study or "")):
            shell.run_line_magic("aimport", "-" + name)
    if sync_mode == "off":
        return
    if shell is None:
        url = _git("remote", "get-url", "origin", cwd=root)
        branch = _git("symbolic-ref", "--short", "HEAD", cwd=root)
        sync_repository(url, branch, parent=root.parent, sync_mode=sync_mode,
                        checkout_path=root)
        return
    session = _NotebookSync(root, study, shell, sync_mode, module_refresh)
    session.refresh(quiet=False)
    shell.ast_transformers.append(session)
    shell.events.register("post_run_cell", session.record_completed_cell)
    shell._hmi_notebook_sync = session

    def stop():
        active = getattr(shell, "_hmi_notebook_sync", None)
        if active is not None:
            shell.ast_transformers.remove(active)
            shell.events.unregister("post_run_cell", active.record_completed_cell)
            del shell._hmi_notebook_sync
        print("Automatic sync detached; checkout and memory unchanged.")

    shell.user_ns["_hmi_stop_sync"] = stop


def load_managed_data(name, path, loader):
    """Register a pure loader and bind its result to an IPython variable.

    Example: load_managed_data("report", path, pandas.read_csv).
    Only the registered binding is replaced on committed file changes. Aliases,
    derived tables and models are not recomputed. Loaders must not write files
    or train models. Existing scientific loading cells remain unchanged.
    """
    shell = _get_shell()
    session = getattr(shell, "_hmi_notebook_sync", None)
    if session is None:
        raise RuntimeError("Call setup_colab in IPython before managing data.")
    if (not isinstance(name, str) or not name.isidentifier()
            or name.startswith("_")):
        raise ValueError("Managed data needs a public Python variable name.")
    path = Path(path).expanduser().resolve()
    if not path.is_relative_to(session.root):
        raise ValueError("Managed data must be inside the repository.")
    value = loader(path)
    shell.user_ns[name] = value
    session.managed[name] = (path, loader, value)
    return value


# =================== 05. Backwards-Compatible Repository Entry ==============


def prepare_notebook_repository(
    url: str, study: str, branch: str = "main", *,
    parent: str | Path = "/content", sync_mode: str = "update",
    module_refresh: str = "auto", data_mode: str = "reuse",
    expected_data_sha256: Mapping[str, str] | None = None,
) -> Path:
    """Legacy entry point; new notebooks call setup_colab directly."""
    name = url.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1]
    root = sync_repository(
        url, branch, parent=parent,
        sync_mode="off" if (Path(parent) / name).exists() else sync_mode,
    )
    from .colab_setup import setup_colab
    return setup_colab(study, root=root, sync_mode=sync_mode,
                       module_refresh=module_refresh, data_mode=data_mode,
                       expected_data_sha256=expected_data_sha256)
