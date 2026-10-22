"""Verify the original Stage 23 notebook through pinned Git history.

The original saved execution is not duplicated in the current file tree.
A full Git checkout is required to retrieve its immutable historical bytes.
"""
import hashlib
from pathlib import Path
import subprocess


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
    current = json.loads(Path(active_path).read_text(encoding="utf-8"))
    old_code = [c for c in archived["cells"] if c.get("cell_type") == "code"]
    new_code = [c for c in current["cells"] if c.get("cell_type") == "code"]
    if len(old_code) != 13 or len(new_code) != 12:
        raise ValueError("Stage 24 code-cell counts must be 13 original/12 current.")
    setup = joined(old_code[0].get("source"))
    if "%cd /content" not in setup or "!git clone " not in setup:
        raise ValueError("Original Stage 24 setup cell does not match protocol.")

    def executable_lines(cell):
        return [
            line for line in joined(cell.get("source")).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    missing_counts = []
    for index, (old, new) in enumerate(zip(old_code[1:], new_code), start=1):
        if old.get("execution_count") is None:
            raise ValueError("Original Stage 24 lacks execution metadata.")
        if executable_lines(old) != executable_lines(new):
            raise ValueError(f"Stage 24 scientific code changed in cell {index}.")
        if new.get("execution_count") is None:
            if not new.get("outputs"):
                raise ValueError(f"Stage 24 code cell {index} has no saved evidence.")
            missing_counts.append(index)
        elif new["execution_count"] != old["execution_count"]:
            raise ValueError(f"Stage 24 execution count changed in cell {index}.")
        originals, replayed = old.get("outputs", []), new.get("outputs", [])
        if len(originals) != len(replayed):
            raise ValueError(f"Stage 24 output count changed in cell {index}.")
        for output_index, (before, after) in enumerate(
            zip(originals, replayed), start=1
        ):
            if before.get("output_type") != after.get("output_type"):
                raise ValueError("Stage 24 output type changed.")
            typ = before.get("output_type")
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
    replay = json.loads(Path(active_path).read_text(encoding="utf-8"))
    if len(original["cells"]) != 25 or len(replay["cells"]) != 25:
        raise ValueError("Stage 23 expected 25 original/current cells.")
    old_codes = [c for c in original["cells"] if c.get("cell_type")=="code"]
    new_codes = [c for c in replay["cells"] if c.get("cell_type")=="code"]
    if len(old_codes) != 12 or len(new_codes) != 12:
        raise ValueError("Stage 23 expected twelve scientific code cells.")
    for index,(old,new) in enumerate(zip(old_codes,new_codes),start=1):
        if old.get("execution_count") != new.get("execution_count"):
            raise ValueError(f"Stage 23 code cell {index} execution count changed.")
        left = [
            line for line in joined(old.get("source")).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        right = [
            line for line in joined(new.get("source")).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if left != right:
            raise ValueError(f"Stage 23 executable code changed at cell {index}.")
        older, newer = old.get("outputs", []),new.get("outputs", [])
        if len(older) != len(newer):
            raise ValueError(f"Stage 23 saved output count changed in cell {index}.")
        for output_index,(before,after) in enumerate(zip(older,newer),start=1):
            if before.get("output_type") != after.get("output_type"):
                raise ValueError("Stage 23 changed output type.")
            typ=before.get("output_type")
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
