"""Dataset-independent data contract for biomedical research experiments.

Presented by: Reza Saadatyar (2026).

The leading axis always denotes an observation. Additional axes may describe
channels, time, spatial dimensions, targets or other modality-specific data.
File decoding and scientific interpretation belong in dataset adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class DatasetBundle:
    """Hold aligned observation arrays with explicit research provenance.

    Args:
        dataset_id: Stable identifier for the source dataset or derivative.
        inputs: Named NumPy arrays with observations on axis zero. Supports
            tabular (N, F), time series (N, C, T) and images (N, ...).
        subject_ids: One subject identifier per observation; repeated IDs
            are expected when one person contributes multiple observations.
        targets: Optional named target arrays with observations on axis zero.
        sample_ids: Unique observation identifiers. They may be local row IDs
            when true source trial identifiers have not been verified.
        input_axes: Optional axis names, including 'sample' as first axis.
        target_axes: Optional target axis names; 'sample' must come first.
        units: Optional mapping from input/target names to documented units.
        provenance: Dataset/version/source notes; values should not contain
            participant-level sensitive attributes.

    Notes:
        This container checks structural alignment, not scientific validity.
        In particular, it cannot prove that external labels correctly match
        rows, or that a time series is free from target leakage.
    """

    dataset_id: str
    inputs: Mapping[str, np.ndarray]
    subject_ids: Sequence[str]
    sample_ids: Sequence[str]
    targets: Mapping[str, np.ndarray] = field(default_factory=dict)
    input_axes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    target_axes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fail early on inconsistent IDs, axes and array dimensions."""
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string")
        if not self.inputs:
            raise ValueError("At least one input modality is required")

        subjects = tuple(str(value).strip() for value in self.subject_ids)
        samples = tuple(str(value).strip() for value in self.sample_ids)
        if not subjects or any(not value for value in subjects):
            raise ValueError("subject_ids must contain non-empty identifiers")
        if len(samples) != len(subjects) or any(not value for value in samples):
            raise ValueError("sample_ids and subject_ids must align and be non-empty")
        if len(set(samples)) != len(samples):
            raise ValueError("sample_ids must be unique within a dataset")
        self.subject_ids = subjects
        self.sample_ids = samples

        self.inputs = self._normalise_arrays(self.inputs, len(subjects), "inputs")
        self.targets = self._normalise_arrays(self.targets, len(subjects), "targets")
        self._validate_axes(self.input_axes, self.inputs, "input_axes")
        self._validate_axes(self.target_axes, self.targets, "target_axes")
        names = set(self.inputs) | set(self.targets)
        if any(name not in names for name in self.units):
            raise ValueError("units refer to an unknown input or target")

    @staticmethod
    def _normalise_arrays(
        arrays: Mapping[str, np.ndarray], n_samples: int, section: str
    ) -> dict[str, np.ndarray]:
        """Convert supported arrays to NumPy without reshaping observations."""
        output: dict[str, np.ndarray] = {}
        for name, raw in arrays.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{section}: modality names must be non-empty strings")
            arr = np.asarray(raw)
            if arr.ndim < 1 or arr.shape[0] != n_samples:
                raise ValueError(
                    f"{section}.{name}: expected {n_samples} observations on axis 0, "
                    f"got shape={arr.shape}"
                )
            if arr.size == 0:
                raise ValueError(f"{section}.{name}: empty array")
            output[name] = arr
        return output

    @staticmethod
    def _validate_axes(
        axes: Mapping[str, tuple[str, ...]],
        arrays: Mapping[str, np.ndarray],
        section: str,
    ) -> None:
        """Validate optional explicit axis descriptions against actual arrays."""
        for name, names in axes.items():
            if name not in arrays:
                raise ValueError(f"{section}: unknown modality '{name}'")
            if len(names) != arrays[name].ndim or not names or names[0] != "sample":
                raise ValueError(
                    f"{section}.{name}: axis names must match shape "
                    "and start with 'sample'"
                )
            if len(set(names)) != len(names) or any(not a for a in names):
                raise ValueError(f"{section}.{name}: axis names must be distinct")

    @property
    def n_samples(self) -> int:
        """Number of observations; not the number of independent subjects."""
        return len(self.sample_ids)

    @property
    def n_subjects(self) -> int:
        """Number of distinct recorded subject IDs."""
        return len(set(self.subject_ids))

    def summary(self) -> dict[str, Any]:
        """Return a safe structural profile, not individual participant data."""
        def describe(arr: np.ndarray) -> dict[str, Any]:
            info: dict[str, Any] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            if np.issubdtype(arr.dtype, np.number):
                info["nonfinite_count"] = int(np.count_nonzero(~np.isfinite(arr)))
            else:
                info["nonfinite_count"] = None
            return info

        return {
            "dataset_id": self.dataset_id,
            "samples": self.n_samples,
            "subjects": self.n_subjects,
            "inputs": {name: describe(arr) for name, arr in self.inputs.items()},
            "targets": {name: describe(arr) for name, arr in self.targets.items()},
            "input_axes": {key: list(value) for key, value in self.input_axes.items()},
            "target_axes": {key: list(value) for key, value in self.target_axes.items()},
            "units": dict(self.units),
            "provenance": dict(self.provenance),
        }
