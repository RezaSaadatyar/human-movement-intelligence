# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Inspect Camargo's subject/date/mode/sensor/trial source organisation.

Only original source metadata are assumed: sensor folders (imu/fp), subject IDs
and a 'Header' timestamp field. MATLAB table/MCOS decoding is not universal;
unsupported source layouts fail explicitly instead of inventing data columns.
"""
from __future__ import annotations

from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

SENSOR_NAMES = {"imu", "fp", "conditions", "gcRight", "gcLeft"}
SUPPORTED = {".csv", ".parquet", ".mat"}

# A bounded, original-version cohort. Its suitability remains subject to QA.
PART1_V1_SUBJECT_ARCHIVES = tuple(
    f"AB{subject:02d}.zip" for subject in range(6, 15)
)


# =========================== 01. Source Discovery ===========================
def inventory_files(source_root: Path) -> pd.DataFrame:
    """List original sensor files without reading entire trial recordings."""
    source_root = Path(source_root)
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Camargo source directory is missing: {source_root}. "
            "Extract one ABxx participant archive from the official source."
        )
    rows = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        rel = path.relative_to(source_root)
        parts = rel.parts
        subject = next((x for x in parts if re.fullmatch(r"AB\d+", x, re.I)), "")
        found = [(i, x) for i, x in enumerate(parts[:-1]) if x in SENSOR_NAMES]
        if not subject or not found:
            continue
        sensor_i, sensor = found[-1]
        mode = parts[sensor_i - 1] if sensor_i > 0 else ""
        date = parts[sensor_i - 2] if sensor_i > 1 else ""
        rows.append({
            "subject_id": subject.upper(), "session_id": date,
            "condition": mode, "sensor": sensor, "trial_id": path.stem,
            "relative_path": rel.as_posix(), "format": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
        })
    return pd.DataFrame(rows, columns=[
        "subject_id", "session_id", "condition", "sensor", "trial_id",
        "relative_path", "format", "size_bytes",
    ])


def list_archives(source_root: Path) -> pd.DataFrame:
    """List local subject ZIPs without downloading or extracting them."""
    rows = []
    for path in sorted(Path(source_root).glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
        rows.append({"archive_name": path.name, "member_count": len(members),
                     "size_bytes": path.stat().st_size})
    return pd.DataFrame(rows, columns=["archive_name", "member_count", "size_bytes"])


def paired_trials(inventory: pd.DataFrame) -> pd.DataFrame:
    """Match same-subject/session/condition/trial IMU and FP file paths."""
    fields = ["subject_id", "session_id", "condition", "trial_id"]
    if inventory.empty:
        return pd.DataFrame(columns=fields + [
            "imu_path", "fp_path", "candidate_pair", "pair_issue"
        ])
    rows = []
    for keys, group in inventory.groupby(fields, dropna=False, sort=True):
        imu = group.loc[group.sensor == "imu", "relative_path"].tolist()
        fp = group.loc[group.sensor == "fp", "relative_path"].tolist()
        issue = "" if len(imu) == len(fp) == 1 else (
            f"expected exactly one IMU and one FP file; found {len(imu)}/{len(fp)}"
        )
        rows.append(dict(zip(fields, keys), imu_path=imu[0] if len(imu) == 1 else "",
                         fp_path=fp[0] if len(fp) == 1 else "",
                         candidate_pair=not bool(issue), pair_issue=issue))
    return pd.DataFrame(rows)


# ========================== 02. Signal Table Reader =========================
def _dict_to_dataframe(data: dict) -> pd.DataFrame:
    """Convert an explicit numeric MATLAB field mapping without guessing axes."""
    if "Header" not in data and "time_s" not in data:
        raise ValueError("No 'Header' or 'time_s' field in numeric mapping.")
    values = {}
    for key, value in data.items():
        if str(key).startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
            values[str(key)] = arr.astype(float)
    time_col = "Header" if "Header" in data else "time_s"
    if time_col not in values:
        raise ValueError(f"Timestamp {time_col} is not a numeric vector.")
    if len(set(map(len, values.values()))) != 1:
        raise ValueError("MATLAB numeric field vectors have different lengths.")
    return pd.DataFrame(values)


def _find_source_table(value, source_name="root", depth=0):
    """Identify a single real named Camargo table in decoded MATLAB content.

    SciPy handles plain numeric structs; mat-io additionally decodes MATLAB
    table/MCOS objects into pandas DataFrames. Source column names and row
    order are kept unchanged. A second eligible table is ambiguous and
    must not be silently chosen.
    """
    if depth > 5:
        return []
    if isinstance(value, pd.DataFrame):
        if "Header" in value.columns or "time_s" in value.columns:
            if value.columns.has_duplicates:
                raise ValueError("MATLAB table has duplicate source columns.")
            return [(source_name, value.copy())]
        return []
    if isinstance(value, dict):
        if "Header" in value or "time_s" in value:
            try:
                table = _dict_to_dataframe(value)
            except (TypeError, ValueError):
                pass
            else:
                return [(source_name, table)]
        results = []
        for name, nested in value.items():
            if str(name).startswith("__"):
                continue
            results.extend(
                _find_source_table(nested, f"{source_name}.{name}", depth + 1)
            )
        return results
    if isinstance(value, (tuple, list)) and len(value) <= 12:
        results = []
        for idx, nested in enumerate(value):
            results.extend(
                _find_source_table(nested, f"{source_name}[{idx}]", depth + 1)
            )
        return results
    return []


def _unambiguous_source_table(decoded, path: Path, reader: str):
    candidates = _find_source_table(decoded)
    if not candidates:
        if isinstance(decoded, dict):
            names = [str(k) for k in decoded if not str(k).startswith("__")]
        else:
            names = [type(decoded).__name__]
        raise ValueError(
            f"{reader}: no named numeric time table was decoded from "
            f"{path.name}; decoded top-level fields: {names[:12]}"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"{reader}: ambiguous multiple time tables in {path.name}: "
            f"{[name for name, _ in candidates]}"
        )
    return candidates[0][1]


def read_sensor_table(path: Path) -> pd.DataFrame:
    """Load source-named sensor columns from CSV, Parquet or MATLAB tables.

    Many original Camargo MAT files store a MATLAB classdef/MCOS table.
    scipy.io.loadmat does not reconstruct the table object, even when the
    outer MAT-file is a standard v7 file. Use mat-io only if the simpler
    scipy path cannot expose an unambiguous numeric time table.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        data = pd.read_csv(path)
    elif suffix == ".parquet":
        data = pd.read_parquet(path)
    elif suffix == ".mat":
        from scipy.io import loadmat

        scipy_issue = None
        try:
            decoded = loadmat(path, simplify_cells=True)
            data = _unambiguous_source_table(decoded, path, "scipy")
        except (NotImplementedError, TypeError, ValueError) as error:
            scipy_issue = error
            try:
                from matio import load_from_mat
            except ImportError as dependency_error:
                raise ImportError(
                    "The original Camargo file contains a MATLAB table/MCOS "
                    "object rather than plain numeric fields. Install "
                    "'mat-io==1.0.2' in this Colab runtime, or rerun the "
                    "Notebook 01 Imports and Research Environment cell. "
                    f"SciPy detail: {scipy_issue}"
                ) from dependency_error
            try:
                decoded = load_from_mat(str(path), raw_data=False)
                data = _unambiguous_source_table(decoded, path, "mat-io")
            except (TypeError, ValueError, OSError, RuntimeError) as error:
                raise ValueError(
                    "Failed to decode the original Camargo MATLAB table. "
                    "Do not synthesise a timestamp or rename an unknown "
                    "sensor channel. Inspect the original MAT variables and "
                    "table type before modifying the scientific mapping. "
                    f"SciPy detail: {scipy_issue}; mat-io detail: {error}"
                ) from error
    else:
        raise ValueError(f"Unsupported sensor file: {path.name}")
    if data.empty:
        raise ValueError(f"Empty sensor table: {path}")
    if "Header" not in data.columns and "time_s" not in data.columns:
        raise ValueError(
            f"{path.name}: decoded table has no original Header/time_s column."
        )
    return data


def validate_timebase(table: pd.DataFrame, source: str = "sensor"):
    """Return seconds if stored timestamps are finite and strictly increasing."""
    candidates = [x for x in ["Header", "time_s"] if x in table.columns]
    if not candidates:
        raise ValueError(f"{source}: explicit Header/time_s timestamp missing.")
    t = pd.to_numeric(table[candidates[0]], errors="coerce").to_numpy(float)
    if len(t) < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f"{source}: invalid/non-monotonic source timestamps.")
    return t


def inspect_columns(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Report observed fields without inferring axis, side or vertical force."""
    rows = []
    for column in table.columns:
        numeric = pd.to_numeric(table[column], errors="coerce")
        rows.append({
            "relative_file": str(path), "column": str(column),
            "dtype": str(table[column].dtype),
            "numeric_fraction": float(numeric.notna().mean()),
            "min_observed": float(numeric.min()) if numeric.notna().any() else np.nan,
            "max_observed": float(numeric.max()) if numeric.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


# ======================= 03. Single-Trial Pilot Access =======================
def extract_pilot_trial_from_zip(
    archive_path: Path, destination: Path, subject_id="AB06",
    condition="treadmill", max_member_bytes=250_000_000,
) -> dict:
    """Extract only one matching original IMU/FP trial, not a whole ZIP.

    The archive must contain the original <ABxx>/<session>/<mode>/<sensor>/
    <trial> layout. No path outside destination is written. This prepares
    physical source files, not a validated force/IMU reference alignment.
    """
    archive_path, destination = Path(archive_path), Path(destination)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Camargo participant ZIP not found: {archive_path}")
    subject_id = str(subject_id).upper()
    choices = {}
    with zipfile.ZipFile(archive_path) as bundle:
        for entry in bundle.infolist():
            if entry.is_dir():
                continue
            parts = Path(entry.filename.replace("\\", "/")).parts
            if (".." in parts or not parts or
                    Path(entry.filename).is_absolute()):
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            matched_subject = next(
                (i for i, part in enumerate(parts)
                 if part.upper() == subject_id), None
            )
            if matched_subject is None:
                continue
            relative = parts[matched_subject:]
            if len(relative) < 5:
                continue
            if (relative[2].lower() != condition.lower() or
                    relative[3].lower() not in {"imu", "fp"} or
                    Path(relative[-1]).suffix.lower() not in SUPPORTED):
                continue
            if entry.file_size > max_member_bytes:
                raise ValueError(
                    f"Pilot file is unexpectedly large: {entry.filename} "
                    f"({entry.file_size} bytes). Verify the source ZIP."
                )
            key = (relative[0].upper(), relative[1],
                   relative[2].lower(), Path(relative[-1]).stem)
            choices.setdefault(key, {})[relative[3].lower()] = (
                entry, Path(*relative)
            )
        complete = [(key, pair) for key, pair in choices.items()
                    if {"imu", "fp"}.issubset(pair)]
        if not complete:
            raise ValueError(
                f"No same-trial original {subject_id}/{condition}/imu+fp "
                "pair exists inside this ZIP. Use a single-participant "
                "archive (e.g. AB06.zip), not the combined 9+ GB Part 1 v2 "
                "archive, and check its internal folder structure."
            )
        key, pair = sorted(complete, key=lambda x: x[0])[0]
        out = {}
        for sensor in ("imu", "fp"):
            member, relative = pair[sensor]
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_file() or target.stat().st_size != member.file_size:
                import shutil
                import tempfile
                with bundle.open(member) as source:
                    with tempfile.NamedTemporaryFile(
                        dir=target.parent, delete=False
                    ) as tmp:
                        temp = Path(tmp.name)
                        shutil.copyfileobj(source, tmp)
                temp.replace(target)
            out[sensor] = target
    return {
        "subject_id": key[0], "session_id": key[1],
        "condition": key[2], "trial_id": key[3],
        "imu_path": str(out["imu"]), "fp_path": str(out["fp"]),
    }


def locate_pilot_source_archive(source_root: Path, subject_id="AB06",
                                explicit_archive=None) -> Path | None:
    """Prefer an explicit path; otherwise only check a few known local spots."""
    if explicit_archive:
        archive = Path(explicit_archive).expanduser()
        if not archive.is_file():
            raise FileNotFoundError(
                f"Specified Camargo archive does not exist: {archive}"
            )
        return archive
    name = f"{str(subject_id).upper()}.zip"
    candidates = [
        Path(source_root) / name,
        Path(source_root).parent / name,
        Path("/content") / name,
    ]
    return next((p for p in candidates if p.is_file()), None)


# =================== 04. Official Mendeley Pilot Acquisition =================
def _mendeley_public_file_metadata(
    filename="AB06.zip", dataset_id="fcgm3chfff", version=1,
    opener=None,
) -> dict:
    """Find one file in a pinned published Mendeley dataset version.

    This reads official public metadata only; it does not download the ZIP.
    No unverified hard-coded file UUID or time-limited signed URL is used.
    """
    import json
    from urllib.request import Request, urlopen

    opener = opener or urlopen
    # The public-api route supports anonymous access to a pinned version.
    # Prefer it to authenticated APIs; the latter remain fallbacks.
    urls = [
        f"https://data.mendeley.com/public-api/datasets/{dataset_id}/files"
        f"?folder_id=root&version={int(version)}",
        f"https://api.data.mendeley.com/datasets/publics/{dataset_id}/files"
        f"?version={int(version)}",
        f"https://api.data.mendeley.com/datasets/{dataset_id}/files"
        f"?version={int(version)}",
    ]
    errors = []
    for url in urls:
        request = Request(
            url, headers={"Accept": "application/json",
                          "User-Agent": "HumanMovementIntelligence/1.0"}
        )
        try:
            with opener(request, timeout=35) as response:
                payload = json.load(response)
        except (OSError, ValueError) as error:
            errors.append(f"{url}: {type(error).__name__}: {error}")
            continue
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = next(
                (payload[key] for key in (
                    "files", "items", "results", "content", "data"
                ) if isinstance(payload.get(key), list)),
                [],
            )
        else:
            records = []
        for file in records:
            if (isinstance(file, dict) and
                    file.get("filename", file.get("name")) == filename):
                return file
        errors.append(f"{url}: {filename} absent from public response")
    raise RuntimeError(
        f"Mendeley API could not locate {filename} in published dataset "
        f"{dataset_id} version {version}. Network access or the public "
        "API may be restricted. Download this file manually from "
        f"https://data.mendeley.com/datasets/{dataset_id}/{version} "
        "and put it in /content/, or set PILOT_ZIP_PATH to its mounted "
        "Google Drive location. API diagnostics: " + " | ".join(errors)
    )


def download_mendeley_pilot_archive(
    destination: Path, filename="AB06.zip",
    dataset_id="fcgm3chfff", version=1,
    max_bytes=1_600_000_000, opener=None,
) -> Path:
    """Download exactly one official public participant ZIP to Colab.

    Streams into a partial file; retries may resume only if HTTP Range is
    actually honoured. Verifies archive format, expected size and SHA256 when
    Mendeley provides it. Never downloads the dataset-wide combined archive.
    The caller must explicitly enable this transfer in the notebook.
    """
    import hashlib
    import shutil
    from urllib.request import Request, urlopen

    destination = Path(destination)
    if (filename not in PART1_V1_SUBJECT_ARCHIVES or
            dataset_id != "fcgm3chfff" or version != 1):
        raise ValueError(
            "The bounded source downloader only accepts Camargo "
            "Part 1 Version 1, AB06.zip through AB14.zip."
        )
    if max_bytes > 1_600_000_000 or max_bytes <= 0:
        raise ValueError("Invalid maximum allowed pilot archive size.")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    if target.is_file() and zipfile.is_zipfile(target):
        print("Reusing existing valid Camargo ZIP:", target)
        return target
    if target.is_file():
        raise ValueError(
            f"{target} exists but is not a valid ZIP; inspect before retrying."
        )
    metadata = _mendeley_public_file_metadata(
        filename, dataset_id, version, opener=opener
    )
    size = metadata.get("size") or (
        metadata.get("content_details") or {}).get("size")
    expected_size = int(size) if size else None
    if expected_size is not None and expected_size > max_bytes:
        raise ValueError(
            f"Official {filename} is {expected_size} bytes, larger "
            "than the allowed single-participant pilot limit."
        )
    details = metadata.get("content_details") or {}
    digest = details.get("sha256_hash") or metadata.get("sha256_hash")
    file_id = metadata.get("id")
    if not file_id:
        raise ValueError("Mendeley response lacks an identifiable file UUID.")
    # Public dataset metadata may include a fresh, short-lived download URL.
    # Otherwise request the canonical file redirect from the official API.
    from urllib.parse import urlsplit
    direct_url = details.get("download_url") or metadata.get("download_url")
    if direct_url:
        parsed = urlsplit(direct_url)
        if parsed.scheme != "https" or not parsed.hostname or (
            parsed.username or parsed.password
        ):
            raise ValueError("Mendeley metadata contained an unsafe file URL.")
    redirect_url = (
        f"https://api.data.mendeley.com/datasets/{dataset_id}/files/"
        f"{file_id}/file_downloaded?version={version}"
    )
    urls = [direct_url, redirect_url] if direct_url else [redirect_url]
    opener = opener or urlopen
    partial = destination / f"{filename}.part"
    start = partial.stat().st_size if partial.exists() else 0
    if start > max_bytes:
        raise ValueError(f"Unexpectedly large partial download: {partial}")
    headers = {"User-Agent": "HumanMovementIntelligence/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    attempts = []
    downloaded = False
    for url in urls:
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "HumanMovementIntelligence/1.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            with opener(Request(url, headers=headers), timeout=90) as response:
                code = response.getcode()
                content_range = response.headers.get("Content-Range", "")
                resume = (
                    existing > 0 and code == 206 and
                    content_range.startswith(f"bytes {existing}-")
                )
                mode = "ab" if resume else "wb"
                count = existing if resume else 0
                next_report = count + 100 * 1024 * 1024
                print(f"Fetching Camargo {filename} (version {version}) "
                      "directly from the official public source.")
                with partial.open(mode) as stream:
                    while True:
                        block = response.read(2 * 1024 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > max_bytes:
                            raise ValueError(
                                "Single-participant ZIP exceeded the "
                                "configured maximum size."
                            )
                        stream.write(block)
                        if count >= next_report:
                            if expected_size:
                                print(f"Received {count / 1e6:.0f} MB / "
                                      f"{expected_size / 1e6:.0f} MB")
                            else:
                                print(f"Received {count / 1e6:.0f} MB")
                            next_report = count + 100 * 1024 * 1024
            if expected_size is not None and count != expected_size:
                attempts.append(
                    f"Incomplete response: {count} of {expected_size} bytes"
                )
                continue
            if not zipfile.is_zipfile(partial):
                attempts.append("Response was not a valid ZIP archive")
                partial.unlink(missing_ok=True)
                continue
            downloaded = True
            break
        except OSError as error:
            attempts.append(f"{type(error).__name__}: {error}")
    if not downloaded:
        raise RuntimeError(
            f"Automatic retrieval of {filename} from Mendeley failed. "
            "Both the public file URL (when provided) and the official "
            "file redirect were attempted. No full-dataset download "
            "was attempted. Check the Colab network and these diagnostics: "
            + " | ".join(attempts)
        )
    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"Incomplete participant download: received "
            f"{partial.stat().st_size} of {expected_size} bytes. "
            "Run the cell again to resume if the provider supports ranges."
        )
    if not zipfile.is_zipfile(partial):
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            "Mendeley did not return a valid ZIP archive. "
            "It may have returned an access page or proxy error; "
            "download AB06.zip manually from the original Version 1 page."
        )
    if digest and len(str(digest)) == 64:
        sha = hashlib.sha256()
        with partial.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                sha.update(block)
        if sha.hexdigest().lower() != str(digest).lower():
            raise RuntimeError(
                "Downloaded AB06.zip does not match the source SHA256. "
                "Do not use this file; verify or obtain a fresh copy."
            )
    partial.replace(target)
    print("Validated original Camargo ZIP:", target)
    return target


# ===================== 05. Bounded Paired Trial Selection ====================
def extract_selected_trial_pairs_from_zip(
    archive_path: Path, destination: Path, subject_id: str,
    condition="treadmill", max_pairs=3,
    max_member_bytes=250_000_000, max_extract_bytes=750_000_000,
) -> list[dict]:
    """Extract a bounded, reproducible set of original IMU/FP trial pairs.

    Selection spans the sorted available trial identifiers without assigning
    walking speed or evaluating labels. All selected paired files retain the
    original subject/session/mode/source names. Outputs are source files,
    not proof that the force axes or clocks are scientifically eligible.
    """
    import shutil
    import tempfile

    subject_id = str(subject_id).upper()
    if not re.fullmatch(r"AB\d{2}", subject_id):
        raise ValueError("Expected a single ABxx participant ID.")
    if max_pairs < 1 or max_pairs > 5:
        raise ValueError("Select between one and five paired trials per person.")
    root = Path(destination)
    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing original ZIP: {archive_path}")
    grouped = {}
    with zipfile.ZipFile(archive_path) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            original = info.filename.replace("\\", "/")
            parts = Path(original).parts
            if not parts or ".." in parts or Path(original).is_absolute():
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            positions = [i for i, part in enumerate(parts)
                         if part.upper() == subject_id]
            if len(positions) != 1:
                continue
            rel = parts[positions[0]:]
            if (len(rel) != 5 or rel[0].upper() != subject_id
                    or rel[2].lower() != condition.lower()
                    or rel[3].lower() not in {"imu", "fp"}
                    or Path(rel[4]).suffix.lower() not in SUPPORTED):
                continue
            if info.file_size > max_member_bytes:
                raise ValueError(
                    f"Source member exceeds single-file size limit: {original}"
                )
            key = (rel[0].upper(), rel[1], rel[2].lower(),
                   Path(rel[4]).stem)
            sensor = rel[3].lower()
            if sensor in grouped.setdefault(key, {}):
                raise ValueError(f"Duplicate {sensor} entry in archive: {key}")
            grouped[key][sensor] = (info, Path(*rel))

        options = sorted((key, group) for key, group in grouped.items()
                         if len(group) == 2)
        if not options:
            raise ValueError(
                f"No exact original treadmill IMU/force pairs for {subject_id}."
            )
        count = min(max_pairs, len(options))
        indices = np.linspace(0, len(options) - 1, count, dtype=int)
        selected = [options[int(i)] for i in indices]
        total = sum(info.file_size for _, group in selected
                    for info, _ in group.values())
        if total > max_extract_bytes:
            raise ValueError(
                f"Selected trial files require {total} unpacked bytes; "
                "reduce max_pairs or review the original archive."
            )
        extracted = []
        for key, group in selected:
            row = {"subject_id": key[0], "session_id": key[1],
                   "condition": key[2], "trial_id": key[3]}
            for sensor in ("imu", "fp"):
                info, rel = group[sensor]
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                if (not target.is_file()
                        or target.stat().st_size != info.file_size):
                    with bundle.open(info) as source:
                        with tempfile.NamedTemporaryFile(
                            dir=target.parent, delete=False
                        ) as handle:
                            temporary = Path(handle.name)
                            shutil.copyfileobj(source, handle)
                    temporary.replace(target)
                row[f"{sensor}_path"] = str(target)
            extracted.append(row)
        return extracted


def participant_pair_counts(source_root: Path,
                            subject_ids: list[str] | tuple[str, ...]
                            ) -> dict[str, int]:
    """Count completed original source pairs by participant, not files."""
    allowed = {str(p).upper() for p in subject_ids}
    inventory = inventory_files(source_root) if Path(source_root).is_dir() \
        else pd.DataFrame()
    matched = paired_trials(inventory)
    counts = {p: 0 for p in sorted(allowed)}
    if not matched.empty:
        complete = matched.loc[
            matched.candidate_pair &
            matched.subject_id.isin(allowed) &
            matched.condition.eq("treadmill")
        ]
        for name, count in complete.subject_id.value_counts().items():
            counts[str(name)] = int(count)
    return counts


# =================== 08. Same-Trial Auxiliary Source Extraction =============
def extract_selected_auxiliary_sources_from_zip(
    archive_path: Path, source_root: Path, selected_pairs,
    *, sensors=("gcRight", "gcLeft", "conditions"),
    max_member_bytes=250_000_000,
    max_extract_bytes=750_000_000,
) -> pd.DataFrame:
    """Extract optional original same-trial streams for selected IMU/FP pairs.

    Selection is established by Notebook 01's original IMU/FP pair IDs.
    Extra streams never determine the cohort or become reference labels.
    Missing optional streams are reported rather than fabricated; existing
    source files are not overwritten when they match original ZIP size.
    """
    import shutil
    import tempfile

    archive_path, source_root = Path(archive_path), Path(source_root)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Original subject archive missing: {archive_path}")
    allowed = {"gcRight", "gcLeft", "conditions"}
    requested = tuple(sensors)
    if not requested or len(set(requested)) != len(requested) or (
        set(requested) - allowed
    ):
        raise ValueError("Auxiliary sensors must be unique original stream names")
    keys = ("subject_id", "session_id", "condition", "trial_id")
    if isinstance(selected_pairs, pd.DataFrame):
        if set(keys).difference(selected_pairs.columns):
            raise ValueError("Selected pair identifiers missing")
        entries = selected_pairs.to_dict("records")
    else:
        entries = list(selected_pairs)
    identities = {
        tuple(str(row[k]) for k in keys) for row in entries
    }
    if not identities or len(identities) != len(entries):
        raise ValueError("Selected pairs missing or contain duplicate trial IDs")
    if any(name != "treadmill" for _, _, name, _ in identities):
        raise ValueError("Only the selected original treadmill pairs are supported")

    original = {}
    with zipfile.ZipFile(archive_path) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            raw_name = info.filename.replace("\\", "/")
            parts = Path(raw_name).parts
            if not parts or ".." in parts or Path(raw_name).is_absolute():
                raise ValueError(f"Unsafe ZIP member: {raw_name}")
            subjects = [i for i, part in enumerate(parts)
                        if re.fullmatch(r"AB\d{2}", part, re.I)]
            if len(subjects) != 1:
                continue
            rel = parts[subjects[0]:]
            if len(rel) != 5:
                continue
            subject, session, condition, sensor, filename = rel
            identity = (subject.upper(), session, condition.lower(),
                        Path(filename).stem)
            if (identity not in identities or sensor not in requested
                    or Path(filename).suffix.lower() not in SUPPORTED):
                continue
            key = (identity, sensor)
            if key in original:
                raise ValueError(f"Duplicate original auxiliary stream: {key}")
            if info.file_size > max_member_bytes:
                raise ValueError(f"Oversize source auxiliary member: {raw_name}")
            original[key] = (info, Path(*rel))
        selected_size = sum(info.file_size for info, _ in original.values())
        if selected_size > max_extract_bytes:
            raise ValueError(
                "Selected original auxiliary members exceed the "
                "configured extraction budget"
            )
        results = []
        for identity in sorted(identities):
            for sensor in requested:
                item = original.get((identity, sensor))
                row = dict(zip(keys, identity))
                row["sensor"] = sensor
                row["source_status"] = "not_in_original_archive"
                row["relative_path"] = ""
                if item:
                    info, relative = item
                    destination = source_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if (destination.is_file() and
                            destination.stat().st_size == info.file_size):
                        row["source_status"] = "already_present"
                    else:
                        with bundle.open(info) as source:
                            with tempfile.NamedTemporaryFile(
                                dir=destination.parent, delete=False
                            ) as handle:
                                temporary = Path(handle.name)
                                try:
                                    shutil.copyfileobj(source, handle)
                                except BaseException:
                                    temporary.unlink(missing_ok=True)
                                    raise
                        if temporary.stat().st_size != info.file_size:
                            temporary.unlink(missing_ok=True)
                            raise ValueError(
                                f"Incomplete original source file: {relative}"
                            )
                        temporary.replace(destination)
                        row["source_status"] = "extracted"
                    row["relative_path"] = relative.as_posix()
                results.append(row)
    return pd.DataFrame(results, columns=list(keys) + [
        "sensor", "source_status", "relative_path",
    ])
