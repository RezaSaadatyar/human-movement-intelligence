# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Read paired NinaPro DB2 EMG and acceleration from official source archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen
import io
import os
import re
import shutil
import tempfile
import zipfile

import numpy as np
from scipy.io import loadmat


# ======================== 01. Source and Data Contracts ======================
SOURCE_PAGE = "https://ninapro.hevs.ch/instructions/DB2.html"
ARCHIVE_URL = "https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s{subject}.zip"
RELEASED_RATE_HZ = 2000.0
EMG_CHANNELS = 12
ACC_CHANNELS = 36
REQUIRED_FIELDS = ("emg", "acc", "restimulus", "rerepetition")


@dataclass(frozen=True)
class ExerciseRecord:
    """Arrays sharing source-sample rows for one person and exercise."""

    subject_id: str
    exercise_id: int
    source_name: str
    emg: np.ndarray
    acc: np.ndarray
    stimulus: np.ndarray
    restimulus: np.ndarray
    repetition: np.ndarray
    rerepetition: np.ndarray


# ====================== 02. Archive Acquisition and Lookup ===================
def archive_path(raw_dir: Path, subject: int) -> Path:
    """Return one stable local name for an official per-subject ZIP."""
    return Path(raw_dir) / f"DB2_s{int(subject)}.zip"


def locate_subject(raw_dir: Path, subject: int) -> Path | None:
    """Find an official ZIP or an extracted E1 MATLAB file without guessing IDs."""
    raw_dir = Path(raw_dir)
    exact = archive_path(raw_dir, subject)
    if exact.is_file():
        return exact
    choices = [raw_dir / f"s{subject}.zip", raw_dir / f"S{subject}.zip"]
    for path in choices:
        if path.is_file():
            return path
    patterns = (f"S{subject}_E1_*.mat", f"s{subject}_E1_*.mat")
    candidates = [path for pattern in patterns for path in raw_dir.rglob(pattern)]
    return sorted(candidates)[0] if candidates else None


def ensure_subject_archive(raw_dir: Path, subject: int, *, download: bool) -> Path:
    """Reuse a local source; otherwise stream an official ZIP atomically.

    This is called explicitly by a notebook, never as an import side effect.
    A failed or incomplete transfer is not promoted to the canonical filename.
    """
    source = locate_subject(raw_dir, subject)
    if source is not None:
        return source
    if not download:
        raise FileNotFoundError(
            f"NinaPro DB2 subject {subject} is missing from {raw_dir}. "
            f"Obtain its archive from {SOURCE_PAGE}, or enable source download."
        )
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_path(raw_dir, subject)
    partial = destination.with_suffix(".zip.part")
    if partial.exists():
        partial.unlink()  # Only an incomplete download owned by this function.
    request = Request(
        ARCHIVE_URL.format(subject=subject),
        headers={"User-Agent": "HumanMovementIntelligenceResearch/1.0"},
    )
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        with zipfile.ZipFile(partial) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Corrupt ZIP member: {bad_member}")
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _safe_mat_members(archive: zipfile.ZipFile, exercise: int) -> list[str]:
    """Select exactly one E-file and reject unexpected archive paths."""
    names = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = Path(info.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in source archive: {info.filename}")
        if path.suffix.lower() != ".mat":
            continue
        if re.search(rf"(?:^|_)E0?{exercise}(?:_|\.)", path.stem, re.I):
            names.append(info.filename)
    return sorted(names)


def list_mat_members(source: Path) -> list[str]:
    """Inspect archive entries or a locally extracted MATLAB file."""
    source = Path(source)
    if source.suffix.lower() == ".mat":
        return [source.name]
    with zipfile.ZipFile(source) as archive:
        members = [info.filename for info in archive.infolist() if info.filename.lower().endswith(".mat")]
        for name in members:
            p = Path(name.replace("\\", "/"))
            if p.is_absolute() or ".." in p.parts:
                raise ValueError(f"Unsafe MATLAB entry: {name}")
        return members


# ========================= 03. MATLAB Field Reading ==========================
def _load_mat_file(source: Path, exercise: int) -> tuple[dict, str]:
    """Load E1/E2/E3 without extracting arbitrary archive contents."""
    source = Path(source)
    if source.suffix.lower() == ".mat":
        if re.search(rf"(?:^|_)E0?{exercise}(?:_|\.)", source.stem, re.I) is None:
            raise ValueError(f"Requested exercise E{exercise} is not in {source.name}")
        return _load_mat(source), source.name
    if source.suffix.lower() != ".zip":
        raise ValueError(f"Unsupported source type: {source}")
    with zipfile.ZipFile(source) as archive:
        members = _safe_mat_members(archive, exercise)
        if len(members) != 1:
            raise ValueError(
                f"Expected exactly one exercise E{exercise} MATLAB file in "
                f"{source.name}; found {members}. Inspect source naming."
            )
        member = members[0]
        with archive.open(member) as stream, tempfile.NamedTemporaryFile(suffix=".mat") as tmp:
            shutil.copyfileobj(stream, tmp, length=1024 * 1024)
            tmp.flush()
            return _load_mat(Path(tmp.name)), member


def _load_mat(path: Path) -> dict:
    try:
        return loadmat(str(path), squeeze_me=False)
    except NotImplementedError as exc:
        try:
            import h5py
        except ImportError as optional_exc:
            raise ImportError("MATLAB v7.3 source requires h5py.") from optional_exc
        with h5py.File(path, "r") as handle:
            return {
                key: np.asarray(value).T
                for key, value in handle.items()
                if hasattr(value, "shape") and value.dtype.kind not in "OSU"
            }


def _vector(data: dict, name: str, rows: int) -> np.ndarray:
    """Read an integer annotation without accepting silent shape changes."""
    if name not in data:
        if name in ("stimulus", "repetition"):
            return np.full(rows, -1, dtype=np.int32)
        raise KeyError(f"Required NinaPro annotation '{name}' was not found.")
    raw = np.asarray(data[name]).reshape(-1)
    if len(raw) != rows or not np.isfinite(raw).all():
        raise ValueError(f"{name} has missing/non-finite or misaligned entries.")
    if not np.all(raw == np.rint(raw)):
        raise ValueError(f"{name} contains non-integral annotation codes.")
    return raw.astype(np.int32)


def load_exercise(source: Path, subject: int, exercise: int = 1) -> ExerciseRecord:
    """Read a paired exercise, preserving source rows, values and class codes.

    Source arrays are interpreted as a released common row grid. This function
    does not invent native accelerometer timestamps or perform extra filtering.
    """
    data, member = _load_mat_file(source, exercise)
    missing = sorted(set(REQUIRED_FIELDS).difference(data))
    if missing:
        raise KeyError(f"NinaPro MATLAB file {member} lacks: {missing}")
    emg = np.asarray(data["emg"], dtype=np.float32)
    acc = np.asarray(data["acc"], dtype=np.float32)
    if emg.ndim != 2 or emg.shape[1] != EMG_CHANNELS:
        raise ValueError(f"Expected (samples, 12) EMG in {member}; observed {emg.shape}")
    if acc.ndim != 2 or acc.shape[1] != ACC_CHANNELS:
        raise ValueError(f"Expected (samples, 36) acceleration in {member}; observed {acc.shape}")
    if len(emg) != len(acc):
        raise ValueError(f"Different EMG/acceleration row counts in {member}")
    if len(emg) < 2:
        raise ValueError(f"No usable source samples in {member}")
    for field, expected in (("subject", int(subject)), ("exercise", int(exercise))):
        if field in data:
            values = np.unique(np.asarray(data[field]).reshape(-1))
            if (len(values) != 1 or not np.isfinite(values).all()
                    or not np.isclose(float(values[0]), expected)):
                raise ValueError(
                    f"Source {field} metadata in {member} disagrees with "
                    f"the requested identity {expected}: {values[:10]}"
                )
    return ExerciseRecord(
        subject_id=f"S{int(subject):02d}", exercise_id=exercise,
        source_name=member, emg=emg, acc=acc,
        stimulus=_vector(data, "stimulus", len(emg)),
        restimulus=_vector(data, "restimulus", len(emg)),
        repetition=_vector(data, "repetition", len(emg)),
        rerepetition=_vector(data, "rerepetition", len(emg)),
    )


# ========================= 04. Segments and Audits ===========================
def contiguous_segments(labels: np.ndarray, repetition: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return half-open corrected-label/repetition segments in source order."""
    labels = np.asarray(labels)
    repetition = np.asarray(repetition)
    if labels.ndim != 1 or repetition.shape != labels.shape:
        raise ValueError("Labels and repetitions must be aligned 1D vectors.")
    if len(labels) == 0:
        return []
    edges = np.r_[0, np.flatnonzero((labels[1:] != labels[:-1]) | (repetition[1:] != repetition[:-1])) + 1, len(labels)]
    return [(int(a), int(b), int(labels[a]), int(repetition[a])) for a, b in zip(edges[:-1], edges[1:])]


def inspect_record(record: ExerciseRecord) -> dict:
    """Summarise observed source provenance without inferring native timing."""
    labels, counts = np.unique(record.restimulus, return_counts=True)
    segments = contiguous_segments(record.restimulus, record.rerepetition)
    return {
        "subject_id": record.subject_id,
        "exercise_id": record.exercise_id,
        "source_file": record.source_name,
        "samples": len(record.emg),
        "duration_s_nominal": len(record.emg) / RELEASED_RATE_HZ,
        "emg_channels": record.emg.shape[1],
        "acc_columns": record.acc.shape[1],
        "emg_finite_fraction": float(np.isfinite(record.emg).mean()),
        "acc_finite_fraction": float(np.isfinite(record.acc).mean()),
        "corrected_labels": ",".join(map(str, labels.tolist())),
        "corrected_repetitions": ",".join(map(str, sorted(set(record.rerepetition.tolist())))),
        "contiguous_segments": len(segments),
        "rest_samples": int(counts[labels == 0].sum()) if 0 in labels else 0,
        "nominal_rate_hz": RELEASED_RATE_HZ,
    }
