# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Download and inspect Essex Figshare MATLAB records.

Public sources are downloaded into ignored local raw-data folders. This module
does not infer unknown MATLAB-object payloads or subject-to-label mappings.
"""

from __future__ import annotations

import csv
import json
import shutil
import stat
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat, whosmat

ARTICLE_ID = 30225739
ARTICLE_VERSION = 13
API_ROOT = "https://api.figshare.com/v2"
MAX_DOWNLOAD_BYTES = 1024**3
MAX_EXTRACTED_BYTES = 1024**3
MATLAB_ARCHIVE_FILE_ID = 63190690
MATLAB_ARCHIVE_SIZE = 235590610
MATLAB_RELEASE_URL = (
    "https://github.com/RezaSaadatyar/human-movement-intelligence/"
    "releases/download/data-essex-v13/MATLAB.zip"
)


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
    matches = [
        item for item in files if item.get("name", "").lower() == filename.lower()
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {filename}; found {len(matches)}.")
    item = matches[0]
    if not item.get("download_url", "").startswith("https://"):
        raise ValueError("Expected an HTTPS URL from the official file manifest.")
    if not 0 < int(item.get("size", 0)) <= MAX_DOWNLOAD_BYTES:
        raise ValueError("Unexpected Figshare source file size.")
    return item


def ensure_source_file(item, target_dir):
    """Reuse the current file, or download it once to the raw-data folder."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    name = item["name"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Expected a filename without directories.")
    destination = target_dir / name
    if destination.is_file():
        return destination
    part = destination.with_suffix(destination.suffix + ".part")
    request = Request(item["download_url"], headers={"User-Agent": "HMI-Essex/1.0"})
    try:
        total = 0
        with urlopen(request, timeout=120) as response, part.open("wb") as out:
            for block in iter(lambda: response.read(2 * 1024**2), b""):
                total += len(block)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Download exceeds the supported archive size.")
                out.write(block)
        expected = item.get("size")
        if not total or (expected and total != int(expected)):
            raise ValueError("Download is incomplete; retry this cell.")
        part.replace(destination)
    finally:
        part.unlink(missing_ok=True)
    return destination


# =============== 01A. Dataset Acquisition ======================
def load_source_manifest(output_dir):
    """Read the current source-file table, or obtain it from Figshare."""
    csv_path = Path(output_dir) / "01_dataset_figshare_file_manifest.csv"
    if not csv_path.is_file():
        return fetch_version_manifest()
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    files = [
        {
            "id": int(row["file_id"]),
            "name": row["filename"],
            "size": int(row["size_bytes"]),
            "download_url": row["download_url"],
        }
        for row in rows
    ]
    article = {
        "id": ARTICLE_ID,
        "version": ARTICLE_VERSION,
        "title": "Essex wearable IMU and femoral-neck finite-element dataset",
        "doi": "10.6084/m9.figshare.30225739.v13",
    }
    return article, files


def ensure_matlab_archive(target_dir):
    """Locate MATLAB.zip locally or download the published Essex v13 archive."""
    destination = Path(target_dir) / "MATLAB.zip"
    if destination.is_file():
        return destination
    endpoints = [
        MATLAB_RELEASE_URL,
        f"https://ndownloader.figshare.com/files/{MATLAB_ARCHIVE_FILE_ID}",
        f"https://figshare.com/ndownloader/files/{MATLAB_ARCHIVE_FILE_ID}",
    ]
    failures = []
    for uri in endpoints:
        try:
            return ensure_source_file(
                {
                    "name": "MATLAB.zip",
                    "size": MATLAB_ARCHIVE_SIZE,
                    "download_url": uri,
                },
                target_dir,
            )
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            failures.append(type(exc).__name__)
    raise RuntimeError("Could not download MATLAB.zip: " + ", ".join(failures))


# ================ 02. Archive Inventory and Extraction ================
def inspect_matlab_archive(zip_path):
    """Validate archive paths, members, CRC and total uncompressed size."""
    inventory = []
    seen = set()
    with ZipFile(zip_path) as archive:
        members = archive.infolist()
        total = 0
        for item in members:
            path = PurePosixPath(item.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in item.filename
                or ":" in item.filename
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
            inventory.append(
                {
                    "filename": item.filename,
                    "size_bytes": item.file_size,
                    "compressed_bytes": item.compress_size,
                    "is_matlab_file": item.filename.lower().endswith(".mat"),
                }
            )
        mat_entries = [x for x in inventory if x["is_matlab_file"]]
        mat_names = [PurePosixPath(item["filename"]).name for item in mat_entries]
        if len(set(mat_names)) != len(mat_names):
            raise ValueError("Two MATLAB files would overwrite the same local name.")
        corrupted = archive.testzip()
        if corrupted:
            raise ValueError(f"Archive CRC failed at {corrupted}.")
    return inventory


def extract_matlab_files(zip_path, destination, inventory):
    """Extract current MAT members, replacing prior extracted copies."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    extracted = []
    with ZipFile(zip_path) as archive:
        for row in inventory:
            if not row["is_matlab_file"]:
                continue
            # Validate here as well: callers may supply their own inventory.
            member = PurePosixPath(row["filename"])
            if member.is_absolute() or ".." in member.parts or "\\" in row["filename"]:
                raise ValueError("Unsafe archive member path.")
            target = destination / member.name
            temporary = target.with_suffix(target.suffix + ".extracting")
            try:
                with (
                    archive.open(row["filename"]) as source,
                    temporary.open("wb") as out,
                ):
                    shutil.copyfileobj(source, out, length=2 * 1024**2)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            extracted.append(target)
    return sorted(extracted, key=lambda path: path.name.lower())


# ================== 03. MATLAB Record Representation Audit ===================
def _record_field(record, name):
    """Read a named MATLAB record field without manufacturing a numeric value."""
    if hasattr(record, "_fieldnames"):
        return getattr(record, name, None)
    if (
        isinstance(record, np.void)
        and record.dtype.names
        and name in record.dtype.names
    ):
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
    matrix = loadmat(
        path, variable_names=["Data"], squeeze_me=False, struct_as_record=False
    )["Data"]
    records = np.asarray(matrix).reshape(-1)
    if not len(records):
        raise ValueError(f"No Data records in {path.name}")
    first_fields = _record_names(records[0])
    statuses = [_target_status(_record_field(record, "e1e3")) for record in records]
    counts = {
        key: statuses.count(key)
        for key in (
            "numeric_candidate",
            "empty",
            "missing_field",
            "nonfinite_numeric",
            "matlab_object_unresolved",
            "non_numeric_unresolved",
        )
    }
    return {
        "subject_id": path.stem.lower(),
        "source_file": path.name,
        "record_count": int(len(records)),
        "source_data_shape": str(tuple(matrix.shape)),
        "first_record_type": type(records[0]).__name__,
        "field_names_first_record": " | ".join(first_fields),
        **counts,
    }
