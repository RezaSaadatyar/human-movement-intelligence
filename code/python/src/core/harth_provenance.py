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
        """Extract actual HTML table cells, ignoring Colab's random wrapper ids."""
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
            for row in re.findall(r"<tr\b[^>]*>(.*?)</tr>", match.group(1), re.DOTALL):
                values = [
                    html.unescape(re.sub(r"<[^>]+>", "", entry)).strip()
                    for entry in re.findall(
                        r"<(?:th|td)\b[^>]*>(.*?)</(?:th|td)>", row, re.DOTALL
                    )
                ]
                rows.append(values)
            return rows

        headings = section_rows("thead")
        body = section_rows("tbody")
        if not headings or not body:
            raise ValueError("Stage 24 displayed HTML table is incomplete.")
        return headings, body

    def _display_equal(historic, replayed):
        old_data = historic.get("data", {})
        new_data = replayed.get("data", {})
        if set(old_data) != set(new_data):
            return False

        old_html = joined(old_data.get("text/html"))
        new_html = joined(new_data.get("text/html"))
        if old_html or new_html:
            old_table = _table_values(old_html)
            new_table = _table_values(new_html)
            if (old_table is None) != (new_table is None):
                return False
            if old_table is not None:
                headings, old_rows = old_table
                newer_headings, new_rows = new_table
                if headings != newer_headings or len(old_rows) != len(new_rows):
                    return False
                columns = headings[-1]
                timing_columns = {
                    "train_seconds", "total_train_s", "total_inference_s",
                    "total_training_s",
                }
                keep = [
                    index for index, column in enumerate(columns)
                    if column not in timing_columns
                ]
                if not keep or any(
                    len(old_row) != len(columns) or len(new_row) != len(columns)
                    for old_row, new_row in zip(old_rows, new_rows)
                ):
                    return False
                if any(
                    [row[index] for index in keep]
                    != [other[index] for index in keep]
                    for row, other in zip(old_rows, new_rows)
                ):
                    return False
            elif old_html != new_html:
                return False

        for key in old_data:
            if key == "text/html":
                continue
            if key == "text/plain" and old_html and new_html:
                # Every data cell is compared in the structured HTML table.
                continue
            if joined(old_data[key]) != joined(new_data[key]):
                return False
        return True

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
        for before, after in zip(originals, replayed):
            if before.get("output_type") != after.get("output_type"):
                raise ValueError("Stage 24 output type changed.")
            typ = before.get("output_type")
            if typ == "error":
                raise ValueError("Stage 24 contains a recorded execution error.")
            if typ == "display_data":
                if not _display_equal(before, after):
                    raise ValueError(
                        f"Stage 24 scientific displayed values changed in cell {index}."
                    )
            elif typ == "stream":
                first, second = joined(before.get("text")), joined(after.get("text"))
                if first != second and not (
                    index == 2 and _cached_download_equivalent(first, second)
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
                if a!=b:
                    raise ValueError(f"Stage 23 logged scientific values changed in cell {index}.")
            elif before!=after:
                raise ValueError(f"Stage 23 unsupported output changed in cell {index}.")
    gate = "".join(
        joined(o.get("text")) for o in new_codes[-1].get("outputs",[])
    )
    if "Stage 23 scientific quality gate: PASSED" not in gate:
        raise ValueError("Stage 23 saved scientific Quality Gate is missing.")
    return True
