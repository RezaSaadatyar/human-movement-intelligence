"""Verify the original Stage 23 notebook through pinned Git history.

The original saved execution is not duplicated in the current file tree.
A full Git checkout is required to retrieve its immutable historical bytes.
"""
import hashlib
from pathlib import Path
import subprocess


# Only the three exact Stage 23/24 import-location edits are normalised.
# Scientific computation and all other executable lines remain strictly checked.
_APPROVED_IMPORT_RELOCATIONS = {
    'while not (root/"python_modules/core/data_loading.py").is_file():':
        'while not (root/"code/python/src/core/data_loading.py").is_file():',
    'sys.path.insert(0,str(root/"python_modules"))':
        'sys.path.insert(0,str(root/"code/python/src"))',
    'sys.path.insert(0,str(root/"wearable_activity_recognition/code/src"))':
        'sys.path.insert(0,str(root/"studies/wearable_activity_recognition/src"))',
}


STAGE23_EVIDENCE_COMMIT = "433f972676a3a0a39c05b34612f201bb27366d36"
STAGE23_EVIDENCE_PATH = (
    "studies/wearable_activity_recognition/experiments/"
    "frozen_notebook_evidence/23_unified_matched_model_benchmark.ipynb"
)


def read_historical_stage23_bytes(repository_root, approved_sha256):
    """Return original executed bytes only when the approved hash matches."""
    root = Path(repository_root).resolve()
    if not (root / ".git").exists():
        raise ValueError(
            "Stage 23 provenance needs a full Git checkout with history."
        )
    reference = f"{STAGE23_EVIDENCE_COMMIT}:{STAGE23_EVIDENCE_PATH}"
    process = subprocess.run(
        ["git", "show", reference],
        cwd=root, capture_output=True, check=False,
    )
    if process.returncode != 0:
        raise ValueError(
            "Original Stage 23 Git evidence is unavailable. "
            "Fetch full repository history before verifying this study."
        )
    recorded = hashlib.sha256(process.stdout).hexdigest()
    if recorded != approved_sha256:
        raise ValueError("Historical Stage 23 SHA256 differs from approval.")
    return process.stdout


# Stage 24 is archived as a 13-code-cell Colab execution. The current readable
# study notebook intentionally omits the original Colab clone/setup cell.
# Two remaining cells have saved outputs but missing execution-count metadata.
STAGE24_EVIDENCE_COMMIT = "c4548596656acaffad4b024203e554c2c3aa3da0"
STAGE24_EVIDENCE_PATH = (
    "studies/wearable_activity_recognition/notebooks/"
    "24_multiseed_and_imbalance_robustness.ipynb"
)


def read_historical_stage24_bytes(repository_root, approved_sha256):
    """Retrieve and hash-check the original executed Stage 24 from Git history."""
    root = Path(repository_root).resolve()
    if not (root / ".git").exists():
        raise ValueError("Stage 24 provenance needs a full Git checkout.")
    process = subprocess.run(
        ["git", "show", f"{STAGE24_EVIDENCE_COMMIT}:{STAGE24_EVIDENCE_PATH}"],
        cwd=root, capture_output=True, check=False,
    )
    if process.returncode != 0:
        raise ValueError(
            "Original Stage 24 Git evidence unavailable; fetch full Git history."
        )
    if hashlib.sha256(process.stdout).hexdigest() != approved_sha256:
        raise ValueError("Historical Stage 24 SHA256 differs from approval.")
    return process.stdout


# The study notebooks below were formatting-reviewed after model selection.
# Compare their 12 research cells with a pinned pre-review Git revision.
# The exact reviewed import/setup cells are separately pinned as Git blobs.
_REVIEWED_PRESENTATION_COMMIT = "d654ab5b774c522fcb1fdf4962ce8d2df3811931"
_GUARDED_COLAB_REVIEW_COMMIT = "50fb7b6494df0956e824921fd4741f50fdaa77d3"

_REVIEWED_PRESENTATION_FILES = {
    23: {
        "path": "wearable_activity_recognition/code/notebooks/23_unified_matched_model_benchmark.ipynb",
        "baseline_blob": "61740bee6c8585b5bbde1323da0136c974b90707",
        "setup_blob": "e9c377d9d390860a2e1d53b66783c29ef568e48c",
        "import_blob": "5968c8e60ac7de945ac60664ff404a50bac7e85c",
        "reviewed_reexecution_blob": "f178f13ece1c987f7deb7b02746ad77e606033aa",
        "current_reexecution_blob": "9d58028051a382b2f5f8ad70ed8885cc12fa796b",
        "guarded_colab_setup_blob": "4f96f44efe18761bea0599e343d0e27f2ac31bab",
        "baseline_code_cells": 13,
    },
    24: {
        "path": "wearable_activity_recognition/code/notebooks/24_multiseed_and_imbalance_robustness.ipynb",
        "baseline_blob": "7421fd1297673884dabe56702ce665be8b98ab9e",
        "setup_blob": "cf426bffe7ed531e554bfb25a0ef4c48602e23bd",
        "import_blob": "04fd1c7262717a8afedfc6f16a6d5f8e5451e68a",
        "reviewed_reexecution_blob": "44c5f79ffa0ee034eb8e0b13f728f2a0e2cb0876",
        "current_reexecution_blob": "3c37a397d101003039cc64b96981bcc326c137b6",
        "guarded_colab_setup_blob": "4f96f44efe18761bea0599e343d0e27f2ac31bab",
        "baseline_code_cells": 12,
    },
}


def _git_blob_digest(data):
    """Calculate the Git blob identity of exact UTF-8 source or file bytes."""
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + bytes([0]) + data
    ).hexdigest()


def _verified_reviewed_presentation(active_path, stage):
    """Verify formatting against pinned pre-review saved research evidence.

    The approved Stage 25 SHA256 values refer to immutable historical notebook
    executions. This check never replaces those hashes or permits arbitrary
    executable changes to the 12 scientific cells.
    """
    import json

    cfg = _REVIEWED_PRESENTATION_FILES[stage]
    active_path = Path(active_path).resolve()
    root = active_path.parents[3]
    if not (root / ".git").is_dir():
        raise ValueError("Historical provenance needs the full Git repository.")
    # Stage names were renumbered in October 2026 for presentation only.
    # Preserve the already-approved SHA256 evidence by accepting ONLY the
    # exact, reviewed renumbered notebook blobs and comparing the corresponding
    # original active snapshots from immutable Git history.
    renamed = {
        23: {
            "active": "22_unified_matched_model_benchmark.ipynb",
            "active_sha": "9fefe8ab3d9163fa629ab99b6ad50f42a59d0d07",
            "historical": (
                "wearable_activity_recognition/code/notebooks/"
                "23_unified_matched_model_benchmark.ipynb"
            ),
            "historical_sha": "5a6f45bde5c6972273d77737fc825b4ee271b132",
        },
        24: {
            "active": "23_multiseed_and_imbalance_robustness.ipynb",
            "active_sha": "29b773ab5a41df8a307d84370762d1ab72c26f9d",
            "historical": (
                "wearable_activity_recognition/code/notebooks/"
                "24_multiseed_and_imbalance_robustness.ipynb"
            ),
            "historical_sha": "226711d9efbee1102291f4f3593b93579f86e345",
        },
    }
    remapped = renamed.get(stage)
    if remapped and active_path.name == remapped["active"]:
        if _git_blob_digest(active_path.read_bytes()) != remapped["active_sha"]:
            raise ValueError(
                "Renumbered notebook has changed since its reviewed snapshot."
            )
        original_ref = (
            "0a516cc1d61506c6ddb7bf3de471aa277f129da8:"
            + remapped["historical"]
        )
        old_active = subprocess.run(
            ["git", "show", original_ref],
            cwd=root, capture_output=True, check=False,
        )
        if old_active.returncode != 0 or (
            _git_blob_digest(old_active.stdout) != remapped["historical_sha"]
        ):
            raise ValueError(
                "Original reviewed notebook required for stage renumbering "
                "verification is not available in Git history."
            )
        return json.loads(old_active.stdout.decode("utf-8"))

    reference = _REVIEWED_PRESENTATION_COMMIT + ":" + cfg["path"]
    result = subprocess.run(
        ["git", "show", reference], cwd=root,
        capture_output=True, check=False,
    )
    if result.returncode != 0 or (
        _git_blob_digest(result.stdout) != cfg["baseline_blob"]
    ):
        raise ValueError("Pinned pre-review notebook source is unavailable.")
    baseline = json.loads(result.stdout.decode("utf-8"))
    current = json.loads(active_path.read_text(encoding="utf-8"))
    older = [c for c in baseline["cells"] if c.get("cell_type") == "code"]
    newer = [c for c in current["cells"] if c.get("cell_type") == "code"]
    if len(older) != cfg["baseline_code_cells"] or len(newer) != 13:
        raise ValueError("Unexpected scientific or setup cell count.")
    setup = newer[0]
    if "00. Colab Repository Setup" not in "".join(setup["source"]):
        raise ValueError("Colab setup is not the first code cell.")
    # A new Colab checkout replaces only the previously reviewed setup cell.
    # Verify every scientific cell, its saved results, and all other notebook
    # metadata against the pinned completed rerun before accepting the change.
    if _git_blob_digest("".join(setup["source"]).encode("utf-8")) == cfg.get(
        "guarded_colab_setup_blob"
    ):
        reviewed_ref = _GUARDED_COLAB_REVIEW_COMMIT + ":" + cfg["path"]
        reviewed_result = subprocess.run(
            ["git", "show", reviewed_ref], cwd=root,
            capture_output=True, check=False,
        )
        if reviewed_result.returncode != 0 or _git_blob_digest(
            reviewed_result.stdout
        ) != cfg["reviewed_reexecution_blob"]:
            raise ValueError("Pinned reviewed scientific notebook is unavailable.")
        reviewed = json.loads(reviewed_result.stdout.decode("utf-8"))
        expected = json.loads(json.dumps(reviewed))
        if len(expected["cells"]) <= 2:
            raise ValueError("Pinned notebook lacks the Colab setup cell.")
        expected["cells"][2]["source"] = setup["source"]
        if current != expected:
            raise ValueError(
                "Only the reviewed Colab checkout may change; scientific "
                "cells, saved outputs and notebook metadata must match exactly."
            )
        return baseline

    # The 2026-10-26 standardisation was followed by saved Colab reruns.
    # Pin the complete reviewed notebook bytes, including execution metadata
    # and output payloads. Never replace approved historic notebook SHA256s.
    # Source-code identity of every scientific cell is checked below; only
    # the final decorative separator print may differ from the older review.
    # Accept only the exact reviewed executions and their separately audited
    # post-formatting snapshots. All scientific cells, their saved outputs and
    # the recorded Quality Gate remain checked below against the pinned baseline.
    # This does not approve arbitrary changes or alter historic selection hashes.
    active_blob = _git_blob_digest(active_path.read_bytes())
    if active_blob in (
        cfg["reviewed_reexecution_blob"],
        cfg["current_reexecution_blob"],
    ):
        if any(
            output.get("output_type") == "error"
            for output in setup.get("outputs", [])
        ):
            raise ValueError("Reviewed Colab setup contains a saved error.")
        original_study = older[1:] if stage == 23 else older
        current_study = newer[1:]
        if len(original_study) != 12 or len(current_study) != 12:
            raise ValueError("Reviewed research cell count has changed.")

        def active_statements(cell):
            return [
                line for line in "".join(cell["source"]).splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]

        for index, (previous, current_cell) in enumerate(
            zip(original_study, current_study), start=1
        ):
            if previous.get("id") != current_cell.get("id"):
                raise ValueError("Reviewed scientific cell identity differs.")
            if any(
                o.get("output_type") == "error"
                for o in current_cell.get("outputs", [])
            ):
                raise ValueError("Reviewed scientific cell contains a saved error.")
            if index == 1:
                # Exact complete-file pin protects the revised imports/setup.
                continue
            old_statements = active_statements(previous)
            new_statements = active_statements(current_cell)
            if (
                index == 12
                and old_statements[-1:] == ['print("="*70)']
                and old_statements[:-1] == new_statements
            ):
                continue
            if old_statements != new_statements:
                raise ValueError(
                    f"Stage {stage} scientific code differs at cell {index}."
                )
        gate = "".join(
            "".join(o.get("text", []))
            if isinstance(o.get("text", []), list) else str(o.get("text", ""))
            for o in current_study[-1].get("outputs", [])
        )
        if f"Stage {stage} scientific quality gate: PASSED" not in gate:
            raise ValueError("Reviewed scientific Quality Gate is absent.")
        return baseline

    if setup.get("outputs") or (
        any(o.get("output_type") == "error" for o in setup.get("outputs", []))
    ):
        raise ValueError("Checkout is not approved scientific evidence.")
    for cell, expected, label in (
        (setup, cfg["setup_blob"], "Colab checkout"),
        (newer[1], cfg["import_blob"], "reviewed scientific imports"),
    ):
        data = "".join(cell["source"]).encode("utf-8")
        if _git_blob_digest(data) != expected:
            raise ValueError(label + " differs from the reviewed source.")
    original_study = older[1:] if stage == 23 else older
    current_study = newer[1:]
    if len(original_study) != 12 or len(current_study) != 12:
        raise ValueError("Research code-cell count is not preserved.")
    for index, (old, new) in enumerate(zip(original_study, current_study)):
        old_metadata = {k: v for k, v in old.items() if k != "source"}
        new_metadata = {k: v for k, v in new.items() if k != "source"}
        if old_metadata != new_metadata:
            raise ValueError(
                f"Stage {stage} saved results or cell metadata changed at {index + 1}."
            )
        if index == 0:
            continue  # The reviewed import cell is pinned in full above.
        def statements(cell):
            return [
                line for line in "".join(cell["source"]).splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        if statements(old) != statements(new):
            raise ValueError(
                f"Stage {stage} experimental code changed at cell {index + 1}."
            )
    return baseline



def _verify_presentation_setup(stage, archived_source, current_source):
    """Audit explicitly approved runtime-only setup without relaxing science code.

    Immutable research constants and model-module imports must match the
    pinned executed source. Only repository bootstrap, device diagnostics and
    human-readable setup logging may differ. No model fitting happens here.
    """
    import re

    original = archived_source.splitlines()
    current = current_source.splitlines()

    def compact(line):
        return re.sub(r"\s+", "", line)

    def imports(lines, first, last):
        start = next(i for i, line in enumerate(lines) if line.startswith(first))
        end = next(i for i in range(start, len(lines)) if lines[i].startswith(last))
        return [compact(line) for line in lines[start:end + 1]
                if line.strip() and not line.lstrip().startswith("#")]

    last_import = (
        "from core.feature_engineering import extract_signal_features"
        if stage == 23 else "from core.robustness_experiments import ("
    )
    if stage == 23:
        # The complete set of source/model imports is immutable.
        if imports(original, "from harth_data import (", last_import) != imports(
            current, "from harth_data import (", last_import
        ):
            raise ValueError("Stage 23 frozen source/model imports changed.")
    else:
        def model_imports(lines):
            start = next(i for i, line in enumerate(lines)
                         if line.startswith("from harth_data import "))
            end = next(i for i in range(start, len(lines))
                       if lines[i].startswith("from core.robustness_experiments import ("))
            end = next(i for i in range(end, len(lines)) if lines[i].strip() == ")")
            return [compact(line) for line in lines[start:end + 1]
                    if line.strip() and not line.lstrip().startswith("#")]
        if model_imports(original) != model_imports(current):
            raise ValueError("Stage 24 frozen source/model imports changed.")

    keys = (
        ["RESERVED_TEST=", "DEVELOPMENT_SUBJECTS=", "CLASSES=",
         "FOLDS=", "MAX_WINDOWS_PER_SUBJECT=", "MAX_STEPS=",
         "BATCH_SIZE=", "SEED=", "REQUIRE_CUDA", "CACHE_DIR=",
         "assert len(DEVELOPMENT_SUBJECTS)", "assert len(MODEL_NAMES)"]
        if stage == 23 else
        ["SEEDS=", "CLASSES=", "RESERVED_TEST=", "DEVELOPMENT=",
         "CACHE_DIR=", "MAX_STEPS=", "assert len(DEVELOPMENT)"]
    )
    for key in keys:
        expected = [compact(line) for line in original
                    if compact(line).startswith(compact(key))]
        actual = [compact(line) for line in current
                  if compact(line).startswith(compact(key))]
        if not expected or actual != expected:
            raise ValueError(f"Stage {stage} frozen protocol value changed: {key}")

    joined = "\n".join(current)
    required = (
        'root = setup_colab("wearable_activity_recognition")',
        "device = computer_device()",
        "DEVICE = device.type",
        'torch.set_float32_matmul_precision("highest")',
        "torch.backends.cuda.matmul.allow_tf32 = False",
        "torch.backends.cudnn.allow_tf32 = False",
        ('if REQUIRE_CUDA and DEVICE != "cuda":' if stage == 23
         else 'if DEVICE != "cuda":'),
    )
    if any(compact(value) not in compact(joined) for value in required):
        raise ValueError(f"Stage {stage} required GPU or precision control missing.")
    if ("from python_modules.core.colab_setup import setup_colab" not in joined or
            "from core.computer_device import computer_device" not in joined):
        raise ValueError(f"Stage {stage} approved shared setup utility missing.")
    if "sys.path.insert" in joined or "Path.cwd().resolve()" in joined:
        raise ValueError(f"Stage {stage} redundant manual repository setup returned.")


def _approved_cosmetic_end_divider(before, after, index, last_index):
    """Permit only removal of one final decorative console separator."""
    return (index == last_index and before and
            before[-1] == 'print("="*70)' and before[:-1] == after)

def verify_stage24_visual_format(active_path, historical_bytes):
    """Compare an executed HARTH Stage 24 replay with hash-approved Git evidence.

    The original 13-code-cell notebook includes Colab setup; its remaining
    12 scientific code cells map exactly to the current presentation. The
    replay may restore two historically missing execution counts and change
    Colab download/cache chatter, ephemeral HTML ids and measured wall time.
    Neither predictive outputs nor executable statements may silently change.
    The immutable approved original remains the selection source.
    """
    import html
    import json
    import re

    def joined(value):
        return "".join(value) if isinstance(value, list) else str(value or "")

    def _table_values(markup):
        """Read a Pandas HTML table, expanding HTML rowspan/colspan cells.

        Pandas MultiIndex tables omit a spanned index cell from subsequent
        <tr> elements (e.g., seed 42 spans folds 1, 2 and 3). Retain that
        index value in every logical data row before comparing scientific
        metrics. The original HTML and saved notebook are never rewritten.
        """
        table = re.search(r"<table\b[^>]*>(.*?)</table>", markup, re.DOTALL)
        if table is None:
            return None

        def section_rows(section):
            match = re.search(
                r"<" + section + r"\b[^>]*>(.*?)</" + section + r">",
                table.group(1), re.DOTALL,
            )
            if match is None:
                return []
            rows = []
            pending = {}
            for raw_row in re.findall(
                r"<tr\b[^>]*>(.*?)</tr>", match.group(1), re.DOTALL
            ):
                values = []

                def take_spanned_columns():
                    while len(values) in pending:
                        column = len(values)
                        value, remaining = pending[column]
                        values.append(value)
                        if remaining == 1:
                            del pending[column]
                        else:
                            pending[column] = (value, remaining - 1)

                for attrs, entry in re.findall(
                    r"<(?:th|td)\b([^>]*)>(.*?)</(?:th|td)>",
                    raw_row, re.DOTALL,
                ):
                    take_spanned_columns()
                    value = html.unescape(
                        re.sub(r"<[^>]+>", "", entry)
                    ).strip()
                    row_match = re.search(
                        r"\browspan\s*=\s*['\"]?(\d+)", attrs
                    )
                    col_match = re.search(
                        r"\bcolspan\s*=\s*['\"]?(\d+)", attrs
                    )
                    rowspan = int(row_match.group(1)) if row_match else 1
                    colspan = int(col_match.group(1)) if col_match else 1
                    if not 1 <= rowspan <= 1000 or not 1 <= colspan <= 1000:
                        raise ValueError("Invalid HTML table span in Stage 24.")
                    for _ in range(colspan):
                        column = len(values)
                        values.append(value)
                        if rowspan > 1:
                            if column in pending:
                                raise ValueError(
                                    "Overlapping HTML table row spans in Stage 24."
                                )
                            pending[column] = (value, rowspan - 1)
                take_spanned_columns()
                rows.append(values)
            if pending:
                raise ValueError("Truncated HTML rowspan in Stage 24 table.")
            return rows

        headings = section_rows("thead")
        body = section_rows("tbody")
        if not headings or not body:
            raise ValueError("Stage 24 displayed HTML table is incomplete.")
        return headings, body

    def _display_difference(historic, replayed):
        """Explain the first scientific mismatch without relaxing the gate.

        DataFrame HTML may contain ephemeral Colab IDs; compare the underlying
        header and row values. Only explicitly declared runtime columns are
        exempt from exact comparison.
        """
        old_data = historic.get("data", {})
        new_data = replayed.get("data", {})
        if set(old_data) != set(new_data):
            return (
                "MIME output types differ: "
                f"historical={sorted(old_data)}, current={sorted(new_data)}"
            )

        old_html = joined(old_data.get("text/html"))
        new_html = joined(new_data.get("text/html"))
        compared_structured_table = False
        if old_html or new_html:
            old_table = _table_values(old_html)
            new_table = _table_values(new_html)
            if (old_table is None) != (new_table is None):
                return "a DataFrame HTML table is missing from one version"
            if old_table is not None:
                compared_structured_table = True
                headings, old_rows = old_table
                newer_headings, new_rows = new_table
                if headings != newer_headings:
                    return (
                        "table column headings or index labels changed: "
                        f"historical={headings!r}, current={newer_headings!r}"
                    )
                if len(old_rows) != len(new_rows):
                    return (
                        "table row count changed: "
                        f"historical={len(old_rows)}, current={len(new_rows)}"
                    )
                columns = headings[-1]
                timing_columns = {
                    "train_seconds", "total_train_s", "total_inference_s",
                    "total_training_s",
                }
                keep = [
                    index for index, column in enumerate(columns)
                    if column not in timing_columns
                ]
                if not keep:
                    return "there are no scientific table columns to compare"
                for row_index, (historical_row, current_row) in enumerate(
                    zip(old_rows, new_rows), start=1
                ):
                    if (len(historical_row) != len(columns) or
                            len(current_row) != len(columns)):
                        return (
                            f"table row {row_index} width differs from "
                            f"{len(columns)} column headings: "
                            f"historical={len(historical_row)}, "
                            f"current={len(current_row)}"
                        )
                    for column_index in keep:
                        before = historical_row[column_index]
                        after = current_row[column_index]
                        if before != after:
                            column = columns[column_index] or (
                                f"index/column position {column_index + 1}"
                            )
                            return (
                                f"row {row_index}, column {column!r}: "
                                f"historical={before!r}, current={after!r}"
                            )
            elif old_html != new_html:
                return "non-tabular HTML content changed"

        for key in old_data:
            if key == "text/html":
                continue
            if key == "text/plain" and compared_structured_table:
                # Ignore Pandas console formatting, but only after every
                # scientific table cell has been checked above.
                continue
            if joined(old_data[key]) != joined(new_data[key]):
                return f"{key} content differs (check saved plot/output bytes)"
        return None

    def _cached_download_equivalent(old_stream, new_stream):
        """Normalise only cache messages; never drop printed cohort statistics."""
        source = r"(?:[A-Za-z]:)?[^\n]*?/[^\n]*?\.csv"
        old_stream = re.sub(
            r"Downloading HARTH v2\.0 source file: [A-Za-z0-9_]+\.csv\n"
            r"Downloaded [0-9,]+ bytes to [^\n]+\.csv\n",
            "SOURCE_FILE_READY\n", old_stream,
        )
        new_stream = re.sub(
            r"Using cached HARTH data: [^\n]+\.csv\n",
            "SOURCE_FILE_READY\n", new_stream,
        )
        return old_stream == new_stream

    def _cached_jobs_match_approved_execution(historical, replay):
        """Verify that RESTORED logs replace only hash-approved completed jobs.

        A warm cache omits per-step training logs. Reconstruct the exact
        expected replay stream from the original completed 36-job run, replacing
        *only* the six-line blocks whose model, loss, seed and fold are marked
        RESTORED in the new stream. All other console output (including logged
        Macro F1 for freshly completed jobs) must match byte-for-byte.
        Saved metric tables are independently compared in subsequent cells.
        """
        restored_pattern = re.compile(
            r"RESTORED: (1D-ResNet|TCN) \| "
            r"(weighted_ce|focal|class_balanced) \| "
            r"seed=(42|123|2026) \| fold=([123])"
        )
        running_pattern = re.compile(
            r"RUNNING: (1D-ResNet|TCN) \| "
            r"(weighted_ce|focal|class_balanced) \| "
            r"seed=(42|123|2026) \| fold=([123])"
        )
        complete_pattern = re.compile(
            r"COMPLETED: (1D-ResNet|TCN) \| "
            r"(weighted_ce|focal|class_balanced) \| "
            r"seed (42|123|2026) \| fold ([123])"
            r" \| Macro F1=(0(?:\.\d+)?|1(?:\.0+)?)"
        )
        step_pattern = re.compile(
            r"  (1D-ResNet|TCN) \| "
            r"(weighted_ce|focal|class_balanced) \| "
            r"seed (42|123|2026) \| Fold ([123]): "
            r"(\d+)/400, Train loss=\d+(?:\.\d+)?"
        )
        replay_lines = replay.splitlines()
        requested = [
            match.groups() for line in replay_lines
            if (match := restored_pattern.fullmatch(line))
        ]
        if not requested or len(set(requested)) != len(requested):
            return False
        if any(
            line.startswith("RESTORED: ")
            and restored_pattern.fullmatch(line) is None
            for line in replay_lines
        ):
            return False
        restored = set(requested)
        original_lines = historical.splitlines()
        expected_lines = []
        approved_keys = set()
        index = 0
        complete_count = 0
        while index < len(original_lines):
            line = original_lines[index]
            running = running_pattern.fullmatch(line)
            if running is None:
                if line.startswith(("RUNNING:", "RESTORED:")):
                    return False
                if complete_pattern.fullmatch(line):
                    complete_count += 1
                expected_lines.append(line)
                index += 1
                continue
            key = running.groups()
            if key not in restored:
                expected_lines.append(line)
                index += 1
                continue

            # Original archived evidence must contain all four training steps
            # and the final, scored completion for this exact condition.
            if index + 5 >= len(original_lines):
                return False
            for offset, step_value in enumerate(("100", "200", "300", "400"), 1):
                step = step_pattern.fullmatch(original_lines[index + offset])
                if step is None or step.groups()[:4] != key or (
                    step.group(5) != step_value
                ):
                    return False
            completed = complete_pattern.fullmatch(original_lines[index + 5])
            if completed is None or completed.groups()[:4] != key:
                return False
            approved_keys.add(key)
            complete_count += 1
            expected_lines.append(
                f"RESTORED: {key[0]} | {key[1]} | "
                f"seed={key[2]} | fold={key[3]}"
            )
            index += 6
        if complete_count != 36 or approved_keys != restored:
            return False
        return expected_lines == replay_lines

    archived = json.loads(historical_bytes.decode("utf-8"))
    current = _verified_reviewed_presentation(active_path, 24)
    old_code = [c for c in archived["cells"] if c.get("cell_type") == "code"]
    new_code = [c for c in current["cells"] if c.get("cell_type") == "code"]
    if len(old_code) != 13 or len(new_code) != 13:
        raise ValueError("Stage 24 expects thirteen original/current code cells.")
    if "00. Colab Repository Setup" not in joined(new_code[0].get("source")):
        raise ValueError("Stage 24 approved Colab bootstrap is missing.")
    setup = joined(old_code[0].get("source"))
    if "%cd /content" not in setup or "!git clone " not in setup:
        raise ValueError("Original Stage 24 setup cell does not match protocol.")

    def executable_lines(cell):
        return [
            line for line in joined(cell.get("source")).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    missing_counts = []
    execution_shift = new_code[1].get("execution_count") - old_code[1].get("execution_count")
    if execution_shift < 0:
        raise ValueError("Stage 24 execution counters move backwards.")
    for index, (old, new) in enumerate(zip(old_code[1:], new_code[1:]), start=1):
        if old.get("execution_count") is None:
            raise ValueError("Original Stage 24 lacks execution metadata.")
        old_lines = executable_lines(old)
        new_lines = executable_lines(new)
        if index == 1:
            _verify_presentation_setup(24, joined(old.get("source")), joined(new.get("source")))
        elif old_lines != new_lines and not _approved_cosmetic_end_divider(
            old_lines, new_lines, index, 12
        ):
            raise ValueError(f"Stage 24 scientific code changed in cell {index}.")
        if new.get("execution_count") is None:
            if not new.get("outputs"):
                raise ValueError(f"Stage 24 code cell {index} has no saved evidence.")
            missing_counts.append(index)
        elif new["execution_count"] != old["execution_count"] + execution_shift:
            raise ValueError(f"Stage 24 execution sequence changed in cell {index}.")
        originals, replayed = old.get("outputs", []), new.get("outputs", [])
        if len(originals) != len(replayed):
            raise ValueError(f"Stage 24 output count changed in cell {index}.")
        for output_index, (before, after) in enumerate(
            zip(originals, replayed), start=1
        ):
            if before.get("output_type") != after.get("output_type"):
                raise ValueError("Stage 24 output type changed.")
            typ = before.get("output_type")
            if index == 1:
                # Device diagnostics changed, never trained model outputs.
                continue
            if typ == "error":
                raise ValueError("Stage 24 contains a recorded execution error.")
            if typ == "display_data":
                difference = _display_difference(before, after)
                if difference is not None:
                    raise ValueError(
                        "Stage 24 scientific displayed values changed in "
                        f"cell {index}, output {output_index}: {difference}"
                    )
            elif typ == "stream":
                first, second = joined(before.get("text")), joined(after.get("text"))
                if index == 12:
                    # A single trailing decorative divider is presentation-only.
                    for_divider = lambda t: re.sub(r"={70}\n?$", "", t)
                    first, second = for_divider(first), for_divider(second)
                if first != second and not (
                    (index == 2 and _cached_download_equivalent(first, second))
                    or (index == 5 and _cached_jobs_match_approved_execution(
                        first, second
                    ))
                ):
                    raise ValueError(
                        f"Stage 24 saved stream differs in cell {index}."
                    )
            elif before != after:
                raise ValueError(
                    f"Stage 24 unsupported output changed in cell {index}."
                )

    if missing_counts not in ([], [1, 10]):
        raise ValueError("Stage 24 has unexpected missing execution counts.")
    for codes in (old_code, new_code):
        gate = "".join(
            joined(o.get("text")) for o in codes[-1].get("outputs", [])
        )
        if "Stage 24 scientific quality gate: PASSED" not in gate:
            raise ValueError("Stage 24 scientific quality gate is missing.")

    return {
        "current_code_cells": len(new_code),
        "historical_code_cells": len(old_code),
        "saved_output_cells": sum(bool(c.get("outputs")) for c in new_code),
        "missing_execution_counts": missing_counts,
        "execution_metadata_complete": not missing_counts,
        "historical_execution_verified": True,
    }


def verify_stage23_visual_format(active_path, historical_bytes):
    """Verify Stage 23 replay against immutable code and predictive evidence.

    Cached source loading, random Colab HTML ids, observed training durations
    and the time-cost chart may differ across executions; model scores and all
    other experimental evidence must agree with approved Git history.
    """
    import html
    import json
    import re

    def joined(value):
        return "".join(value) if isinstance(value, list) else str(value or "")

    def table_values(markup):
        section = re.search(r"<table\b[^>]*>(.*?)</table>", markup, re.DOTALL)
        if section is None:
            return None

        def rows(part):
            s = re.search(
                r"<" + part + r"\b[^>]*>(.*?)</" + part + r">",
                section.group(1), re.DOTALL,
            )
            if s is None:
                return []
            return [
                [
                    html.unescape(re.sub(r"<[^>]+>", "", value)).strip()
                    for value in re.findall(
                        r"<(?:th|td)\b[^>]*>(.*?)</(?:th|td)>",
                        row, re.DOTALL,
                    )
                ]
                for row in re.findall(
                    r"<tr\b[^>]*>(.*?)</tr>", s.group(1), re.DOTALL
                )
            ]
        headings, body = rows("thead"), rows("tbody")
        if not headings or not body:
            raise ValueError("Stage 23 displayed table is incomplete.")
        return headings, body

    def display_matches(old, new, code_index, output_index, code_source):
        previous, current = old.get("data", {}), new.get("data", {})
        if set(previous) != set(current):
            return False
        old_html = joined(previous.get("text/html"))
        new_html = joined(current.get("text/html"))
        if old_html or new_html:
            t1,t2 = table_values(old_html), table_values(new_html)
            if (t1 is None) != (t2 is None):
                return False
            if t1 is not None:
                headings, rows1 = t1
                new_headings, rows2 = t2
                if headings != new_headings or len(rows1) != len(rows2):
                    return False
                headers = headings[-1]
                varying = {
                    "train_seconds", "validation_inference_seconds",
                    "total_train_seconds", "total_inference_seconds",
                    "total_train_s", "mean_train_s_per_fold",
                    "total_validation_inference_s",
                }
                keep = [i for i, key in enumerate(headers) if key not in varying]
                if not keep or any(
                    len(x) != len(headers) or len(y) != len(headers)
                    for x,y in zip(rows1,rows2)
                ):
                    return False
                if any(
                    [x[k] for k in keep] != [y[k] for k in keep]
                    for x,y in zip(rows1,rows2)
                ):
                    return False
            elif old_html != new_html:
                return False
        for key in previous:
            if key == "text/html" or (
                key == "text/plain" and old_html and new_html
            ):
                continue
            if key == "image/png" and code_index == 10 and output_index == 2:
                # Stage 23 section 10 plots measured runtime only.
                if "total_train" not in code_source:
                    return False
                continue
            if joined(previous[key]) != joined(current[key]):
                return False
        return True

    def cache_normalized(text):
        text = re.sub(
            r"Downloading HARTH v2\.0 source file: [A-Za-z0-9_]+\.csv\n"
            r"Downloaded [0-9,]+ bytes to [^\n]+\.csv\n",
            "SOURCE_FILE_READY\n", text,
        )
        return re.sub(
            r"Using cached HARTH data: [^\n]+\.csv\n",
            "SOURCE_FILE_READY\n", text,
        )

    original = json.loads(historical_bytes.decode("utf-8"))
    replay = _verified_reviewed_presentation(active_path, 23)
    # Baseline Stage 23 contains an added, non-scientific setup pair.
    replay["cells"] = [replay["cells"][0]] + replay["cells"][3:]
    if len(original["cells"]) != 25 or len(replay["cells"]) != 27:
        raise ValueError("Stage 23 expects 25 historical/27 current cells.")
    old_codes = [c for c in original["cells"] if c.get("cell_type")=="code"]
    new_codes = [c for c in replay["cells"] if c.get("cell_type")=="code"]
    if len(old_codes) != 12 or len(new_codes) != 13:
        raise ValueError("Stage 23 expects twelve scientific cells plus Colab bootstrap.")
    if "00. Colab Repository Setup" not in joined(new_codes[0].get("source")):
        raise ValueError("Stage 23 Colab bootstrap is missing.")
    execution_shift = new_codes[1].get("execution_count") - old_codes[0].get("execution_count")
    if execution_shift < 0:
        raise ValueError("Stage 23 execution counters move backwards.")
    for index,(old,new) in enumerate(zip(old_codes,new_codes[1:]),start=1):
        if old.get("execution_count") + execution_shift != new.get("execution_count"):
            raise ValueError(f"Stage 23 execution order changed in cell {index}.")
        left = [
            line for line in joined(old.get("source")).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        right = [
            line for line in joined(new.get("source")).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if index == 1:
            _verify_presentation_setup(23, joined(old.get("source")), joined(new.get("source")))
        elif left != right and not _approved_cosmetic_end_divider(
            left, right, index, 12
        ):
            raise ValueError(f"Stage 23 executable code changed at cell {index}.")
        older, newer = old.get("outputs", []),new.get("outputs", [])
        if len(older) != len(newer):
            raise ValueError(f"Stage 23 saved output count changed in cell {index}.")
        for output_index,(before,after) in enumerate(zip(older,newer),start=1):
            if before.get("output_type") != after.get("output_type"):
                raise ValueError("Stage 23 changed output type.")
            typ=before.get("output_type")
            if index == 1:
                # Shared device diagnostics replace the historical setup summary.
                continue
            if typ=="error":
                raise ValueError("Stage 23 contains saved execution errors.")
            if typ=="display_data":
                if not display_matches(
                    before,after,index,output_index,joined(old.get("source"))
                ):
                    raise ValueError(
                        f"Stage 23 predictive displayed values changed in cell {index}."
                    )
            elif typ=="stream":
                a,b=joined(before.get("text")),joined(after.get("text"))
                if index == 12:
                    a = re.sub(r"={70}\n?$", "", a)
                    b = re.sub(r"={70}\n?$", "", b)
                if index==2:
                    a,b=cache_normalized(a),cache_normalized(b)
                elif index==5:
                    # Only remove elapsed GPU/CPU time, not validation scores.
                    a=re.sub(r"(\| Train=)[0-9]+(?:\.[0-9]+)?( s)",r"\1RUNTIME\2",a)
                    b=re.sub(r"(\| Train=)[0-9]+(?:\.[0-9]+)?( s)",r"\1RUNTIME\2",b)
                # Execution reruns can add/remove blank presentation lines.
                # Compare all non-empty lines verbatim to retain scientific
                # scores, cohort counts, warnings and quality-gate evidence.
                if [line for line in a.splitlines() if line.strip()] != [
                    line for line in b.splitlines() if line.strip()
                ]:
                    raise ValueError(f"Stage 23 logged scientific values changed in cell {index}.")
            elif before!=after:
                raise ValueError(f"Stage 23 unsupported output changed in cell {index}.")
    gate = "".join(
        joined(o.get("text")) for o in new_codes[-1].get("outputs",[])
    )
    if "Stage 23 scientific quality gate: PASSED" not in gate:
        raise ValueError("Stage 23 saved scientific Quality Gate is missing.")
    return True
