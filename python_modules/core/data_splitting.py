# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Deterministic, group-disjoint Train/Validation/Test splitting."""

import numpy as np
import pandas as pd


# ====================== 01. Group-Independent Splitting ======================
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


# ===================== 02. Subject/Class Coverage Audit ======================
def audit_group_class_coverage(labels, group_ids, classes, n_splits=5):
    """
    Audit label availability in independent subjects and GroupKFold partitions.

    Args:
        labels (array-like): One documented class label per observation.
        group_ids (array-like): One independent subject/group ID per observation.
        classes (sequence): Fixed class codes for the scientific task.
        n_splits (int): Number of participant-disjoint folds to inspect.

    Returns:
        participant_counts (pd.DataFrame): Per-person counts of each class.
            Columns preserve the order of the declared classes.
        fold_audit (pd.DataFrame): Train/validation group membership, window
            counts, class coverage and missing-class codes for each fold.

    Notes:
        This is a READ-ONLY study-design audit. It does not fit a model,
        inspect the reserved Test set, stratify labels or modify samples.
        GroupKFold balances observations approximately, not class support.
    """
    from sklearn.model_selection import GroupKFold

    # 1. Validate aligned input labels and independent participant IDs.
    y = np.asarray(labels)
    raw_groups = np.asarray(group_ids, dtype=object)
    codes = list(classes)
    if y.ndim != 1 or raw_groups.ndim != 1 or len(y) == 0:
        raise ValueError("labels and group_ids must be non-empty 1D arrays.")
    if len(y) != len(raw_groups):
        raise ValueError("labels and group_ids must have the same length.")
    if pd.isna(y).any() or any(
        pd.isna(value) or not str(value).strip() for value in raw_groups
    ):
        raise ValueError("labels or group_ids contain missing values.")
    if len(codes) < 2 or len(set(codes)) != len(codes):
        raise ValueError("Provide at least two distinct declared class codes.")
    if not np.isin(y, codes).all():
        raise ValueError("Observed labels outside the declared classes.")
    if isinstance(n_splits, bool) or not isinstance(
        n_splits, (int, np.integer)
    ):
        raise ValueError("n_splits must be an integer.")

    # 2. Normalise group identity but retain original spelling in the report.
    original_ids = [str(value).strip() for value in raw_groups]
    groups = np.asarray([value.casefold() for value in original_ids])
    display_names = {}
    for canonical, original in zip(groups, original_ids):
        display_names.setdefault(canonical, original)
    unique_groups = np.unique(groups)
    if n_splits < 2 or n_splits > len(unique_groups):
        raise ValueError("n_splits must be between 2 and the subject count.")

    # 3. Count all declared classes for every independent participant.
    participant_counts = pd.crosstab(
        pd.Series(groups, name="subject_id"),
        pd.Series(y, name="class_code"),
    ).reindex(index=unique_groups, columns=codes, fill_value=0)
    participant_counts.index = [
        display_names[value] for value in participant_counts.index
    ]

    # 4. Audit each possible fold BEFORE fitting any classifier.
    splitter = GroupKFold(n_splits=n_splits)
    fold_records = []
    held_out = []
    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(np.zeros(len(y)), y, groups), start=1
    ):
        train_group_ids = set(groups[train_indices])
        valid_group_ids = set(groups[valid_indices])
        if train_group_ids & valid_group_ids:
            raise RuntimeError("Subject leakage in grouped fold audit.")
        held_out.extend(valid_indices.tolist())

        train_classes = set(np.unique(y[train_indices]).tolist())
        valid_classes = set(np.unique(y[valid_indices]).tolist())
        fold_records.append({
            "fold": fold,
            "train_subjects": ", ".join(
                sorted(display_names[g] for g in train_group_ids)
            ),
            "validation_subjects": ", ".join(
                sorted(display_names[g] for g in valid_group_ids)
            ),
            "train_windows": len(train_indices),
            "validation_windows": len(valid_indices),
            "training_classes_present": len(train_classes),
            "validation_classes_present": len(valid_classes),
            "missing_train_classes": tuple(
                code for code in codes if code not in train_classes
            ),
            "missing_validation_classes": tuple(
                code for code in codes if code not in valid_classes
            ),
        })

    if sorted(held_out) != list(range(len(y))):
        raise RuntimeError("Validation folds do not cover each row exactly once.")

    fold_audit = pd.DataFrame(fold_records)
    print(
        f"Group audit: {len(unique_groups)} subjects, {len(y):,} windows, "
        f"{len(codes)} declared classes, {n_splits} folds"
    )
    print(
        "Folds missing training classes:",
        int(fold_audit["missing_train_classes"].map(bool).sum()),
    )
    return participant_counts, fold_audit
