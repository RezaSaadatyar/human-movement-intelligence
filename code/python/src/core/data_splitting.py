# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Deterministic, group-disjoint Train/Validation/Test splitting."""

import numpy as np
import pandas as pd


# ========================= 1. Group-Independent Splitting =========================
def split_group_indices(
    group_ids,
    train_fraction=0.50,
    validation_fraction=0.25,
    test_fraction=0.25,
    random_state=42,
):
    """
    Split observations without allowing any subject/group in multiple sets.

    Args:
        group_ids (array-like): One participant/session/group ID per row.
            Repeated IDs are expected when individuals contribute many windows.
        train_fraction (float): Fraction of independent groups in Train.
        validation_fraction (float): Fraction of groups in Validation.
        test_fraction (float): Fraction of groups in Test.
        random_state (int): Fixed seed for reproducible group assignments.

    Returns:
        dict[str, np.ndarray]: Original row indices for 'train',
            'validation' and 'test'.

    Notes:
        Fractions describe group counts rather than sample counts. This
        method is NOT class-stratified. Fit preprocessing on Train only.
        All windows from any one subject must stay in the same partition.
    """
    # 1. Validate and standardise subject/group identifiers.
    ids = np.asarray(group_ids, dtype=object)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("group_ids must be a non-empty 1D array.")
    if any(pd.isna(value) or not str(value).strip() for value in ids):
        raise ValueError("group_ids contain empty or missing identifiers.")
    ids = np.asarray([str(value).strip().casefold() for value in ids])
    groups = np.unique(ids)
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required.")

    # 2. Validate the group fractions and random seed.
    fractions = np.asarray(
        [train_fraction, validation_fraction, test_fraction], dtype=float
    )
    if not np.isfinite(fractions).all() or np.any(fractions <= 0):
        raise ValueError("All split fractions must be finite and positive.")
    if not np.isclose(fractions.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("Split fractions must sum to 1.")
    if isinstance(random_state, bool) or not isinstance(
        random_state, (int, np.integer)
    ) or random_state < 0:
        raise ValueError("random_state must be a non-negative integer.")

    # 3. Convert fractional targets into actual integer subject counts.
    ideal = fractions * len(groups)
    counts = np.floor(ideal).astype(int)
    remainder_order = np.argsort(-(ideal - counts), kind="stable")
    for part in remainder_order[:len(groups) - counts.sum()]:
        counts[part] += 1
    for missing_part in np.flatnonzero(counts == 0):
        candidates = np.flatnonzero(counts > 1)
        donor = candidates[np.argmax(counts[candidates])]
        counts[donor] -= 1
        counts[missing_part] += 1

    # 4. Randomise GROUPS rather than windows to prevent participant leakage.
    shuffled = np.random.default_rng(int(random_state)).permutation(groups)
    train_end = counts[0]
    validation_end = train_end + counts[1]
    partitions = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }

    # 5. Restore original row indices for applying one split to X and y.
    indices = {
        name: np.flatnonzero(np.isin(ids, members))
        for name, members in partitions.items()
    }
    membership = [set(values) for values in partitions.values()]
    if any(len(index) == 0 for index in indices.values()):
        raise RuntimeError("A split has no observations.")
    if (membership[0] & membership[1]
            or membership[0] & membership[2]
            or membership[1] & membership[2]):
        raise RuntimeError("Subject overlap between partitions.")

    print(f"Independent groups: {len(groups)}")
    for name, members in partitions.items():
        print(f"{name}: {len(members)} groups | {len(indices[name]):,} rows")
    return indices
