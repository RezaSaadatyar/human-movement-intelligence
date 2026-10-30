# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Acquire and audit pinned Essex Figshare MATLAB records without modelling.

Public sources are downloaded into ignored local raw-data folders. This module
does not infer unknown MATLAB-object payloads or subject-to-label mappings.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat, whosmat

ARTICLE_ID = 30225739
ARTICLE_VERSION = 13
API_ROOT = "https://api.figshare.com/v2"
MAX_DOWNLOAD_BYTES = 1024 ** 3
MAX_EXTRACTED_BYTES = 1024 ** 3


# ================ 01. Figshare Metadata and File Acquisition =================
def fetch_json(url):
    """Retrieve a public HTTPS Figshare JSON document without credentials."""
    if not url.startswith("https://api.figshare.com/"):
        raise ValueError("Only the official Figshare API is supported.")
    request = Request(url, headers={"User-Agent": "HMI-Essex-Research/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_version_manifest(article_id=ARTICLE_ID, version=ARTICLE_VERSION):
    """Return verified article-version metadata and its published file list."""
    endpoint = f"{API_ROOT}/articles/{article_id}/versions/{version}"
    article = fetch_json(endpoint)
    files = fetch_json(f"{endpoint}/files")
    if int(article.get("id", -1)) != int(article_id):
        raise ValueError("Figshare article identity differs from the pinned ID.")
    if int(article.get("version", -1)) != int(version):
        raise ValueError("Figshare version differs from the pinned version.")
    if not isinstance(files, list) or not files:
        raise ValueError("Figshare returned no files for the pinned version.")
    names = [item.get("name") for item in files]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate file names in version manifest.")
    return article, files


def find_published_file(files, filename):
    """Locate one unambiguous published file by its exact, case-insensitive name."""
    matches = [item for item in files if item.get("name", "").lower() == filename.lower()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {filename}; found {len(matches)}.")
    item = matches[0]
    if not item.get("download_url", "").startswith("https://"):
        raise ValueError("Expected an HTTPS URL from the official file manifest.")
    if not 0 < int(item.get("size", 0)) <= MAX_DOWNLOAD_BYTES:
        raise ValueError("Unexpected Figshare source file size.")
    return item


def file_hashes(path):
    """Compute SHA256 and MD5 over the exact source bytes without loading them."""
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024 ** 2), b""):
            sha256.update(block)
            md5.update(block)
    return {"sha256": sha256.hexdigest(), "md5": md5.hexdigest()}


def ensure_source_file(item, target_dir):
    """Download once with bounds, validate source size and any published MD5.

    Existing data are never overwritten silently. New downloads are moved
    into place only after all available identity checks have succeeded.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    name = item["name"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Manifest filename must be a simple filename.")
    destination = target_dir / name
    part = target_dir / f"{name}.part"
    expected_size = int(item["size"])
    if not destination.exists():
        if part.exists():
            raise FileExistsError(f"Unfinished download exists: {part}")
        request = Request(
            item["download_url"], headers={"User-Agent": "HMI-Essex-Research/1.0"}
        )
        try:
            total = 0
            with urlopen(request, timeout=120) as response, part.open("xb") as out:
                for block in iter(lambda: response.read(2 * 1024 ** 2), b""):
                    total += len(block)
                    if total > expected_size or total > MAX_DOWNLOAD_BYTES:
                        raise ValueError("Source download exceeded expected size.")
                    out.write(block)
            if total != expected_size:
                raise ValueError("Incomplete download according to Figshare size.")
            hashes = file_hashes(part)
            published_md5 = item.get("computed_md5") or item.get("supplied_md5")
            if published_md5 and hashes["md5"].lower() != published_md5.lower():
                raise ValueError("Downloaded file MD5 differs from Figshare.")
            part.rename(destination)
        except Exception:
            part.unlink(missing_ok=True)
            raise
    if not destination.is_file() or destination.stat().st_size != expected_size:
        raise ValueError(f"Cached source size differs from Figshare: {destination}")
    hashes = file_hashes(destination)
    published_md5 = item.get("computed_md5") or item.get("supplied_md5")
    if published_md5 and hashes["md5"].lower() != published_md5.lower():
        raise ValueError("Cached file MD5 differs from official Figshare metadata.")
    return destination, hashes


# ================ 02. ZIP Integrity and Controlled Extraction ================
def inspect_matlab_archive(zip_path, expected_mat_files=40):
    """Validate archive paths, members, CRC and total uncompressed size."""
    inventory = []
    seen = set()
    with ZipFile(zip_path) as archive:
        members = archive.infolist()
        total = 0
        for item in members:
            path = PurePosixPath(item.filename)
            if (
                path.is_absolute() or ".." in path.parts
                or "\\" in item.filename or ":" in item.filename
            ):
                raise ValueError(f"Unsafe ZIP entry: {item.filename}")
            if item.filename in seen:
                raise ValueError(f"Duplicate ZIP member: {item.filename}")
            seen.add(item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"ZIP symlink not permitted: {item.filename}")
            if item.file_size > MAX_EXTRACTED_BYTES:
                raise ValueError("Individual ZIP entry exceeds safety limit.")
            total += item.file_size
            if total > MAX_EXTRACTED_BYTES:
                raise ValueError("Total extracted archive size exceeds 1 GiB.")
            inventory.append({
                "filename": item.filename, "size_bytes": item.file_size,
                "compressed_bytes": item.compress_size,
                "is_matlab_file": item.filename.lower().endswith(".mat"),
            })
        mat_entries = [x for x in inventory if x["is_matlab_file"]]
        if len(mat_entries) != expected_mat_files:
            raise ValueError(
                f"Expected {expected_mat_files} MAT files, found {len(mat_entries)}."
            )
        mat_names = [PurePosixPath(item["filename"]).name for item in mat_entries]
        if len(set(mat_names)) != len(mat_names):
            raise ValueError("Two MATLAB files would overwrite the same local name.")
        corrupted = archive.testzip()
        if corrupted:
            raise ValueError(f"Archive CRC failed at {corrupted}.")
    return inventory


def extract_matlab_files(zip_path, destination, inventory):
    """Extract only validated MAT files; never overwrite different raw files."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    selected = [row for row in inventory if row["is_matlab_file"]]
    extracted = []
    with ZipFile(zip_path) as archive:
        for row in selected:
            name = PurePosixPath(row["filename"]).name
            target = destination / name
            if target.exists():
                if not target.is_file() or target.stat().st_size != row["size_bytes"]:
                    raise ValueError(f"Unexpected existing MATLAB file: {target}")
            else:
                temporary = destination / f"{name}.extracting"
                if temporary.exists():
                    raise FileExistsError(f"Unfinished extraction exists: {temporary}")
                try:
                    with archive.open(row["filename"]) as source, temporary.open("xb") as out:
                        shutil.copyfileobj(source, out, length=2 * 1024 ** 2)
                    if temporary.stat().st_size != row["size_bytes"]:
                        raise ValueError(f"Extraction size mismatch: {name}")
                    temporary.rename(target)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
            extracted.append(target)
    return sorted(extracted, key=lambda x: x.name.lower())


# ================== 03. MATLAB Record Representation Audit ===================
def _record_field(record, name):
    """Read a named MATLAB record field without manufacturing a numeric value."""
    if hasattr(record, "_fieldnames"):
        return getattr(record, name, None)
    if isinstance(record, np.void) and record.dtype.names and name in record.dtype.names:
        return record[name]
    if isinstance(record, dict):
        return record.get(name)
    return None


def _record_names(record):
    """Return source field names or an empty list for opaque MATLAB objects."""
    if hasattr(record, "_fieldnames"):
        return list(record._fieldnames)
    if isinstance(record, np.void) and record.dtype.names:
        return list(record.dtype.names)
    if isinstance(record, dict):
        return list(record)
    return []


def _target_status(value):
    """Classify FE target representation without assuming MATLAB decoding."""
    if value is None:
        return "missing_field"
    values = np.asarray(value)
    if values.size == 0:
        return "empty"
    if np.issubdtype(values.dtype, np.number):
        return "numeric_candidate" if np.isfinite(values).all() else "nonfinite_numeric"
    if values.dtype.names or values.dtype == object:
        return "matlab_object_unresolved"
    return "non_numeric_unresolved"


def inspect_subject_mat(path):
    """Audit top-level Data records and target representations, not waveforms.

    Do not claim that a nonempty object is an independently verified strain.
    """
    path = Path(path)
    variables = whosmat(path)
    matches = [v for v in variables if v[0] == "Data"]
    if len(matches) != 1:
        raise ValueError(f"Missing unique Data variable in {path.name}")
    matrix = loadmat(path, variable_names=["Data"], squeeze_me=False,
                     struct_as_record=False)["Data"]
    records = np.asarray(matrix).reshape(-1)
    if not len(records):
        raise ValueError(f"No Data records in {path.name}")
    first_fields = _record_names(records[0])
    statuses = [_target_status(_record_field(record, "e1e3")) for record in records]
    counts = {key: statuses.count(key) for key in (
        "numeric_candidate", "empty", "missing_field", "nonfinite_numeric",
        "matlab_object_unresolved", "non_numeric_unresolved"
    )}
    return {
        "subject_id": path.stem.lower(),
        "source_file": path.name,
        "record_count": int(len(records)),
        "source_data_shape": str(tuple(matrix.shape)),
        "first_record_type": type(records[0]).__name__,
        "field_names_first_record": " | ".join(first_fields),
        "source_sha256": file_hashes(path)["sha256"],
        **counts,
    }
