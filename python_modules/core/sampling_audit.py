# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Deterministic temporal-window sampling audits without model fitting."""

import numpy as np
import pandas as pd


# ============================ 1. Subsampling Policy =============================
def select_evenly_spaced_indices(eligible_count, maximum_windows):
    """
    Select indices evenly along an ordered eligible recording.

    Args:
        eligible_count (int): Number of eligible, already-segmented windows.
        maximum_windows (int): Maximum retained windows for this participant.

    Returns:
        np.ndarray: Strictly increasing local window indices.

    Notes:
        This duplicates the Stage 10-13 time-order selection rule exactly.
        Activity labels never determine which positions are retained.
    """
    if (
        isinstance(eligible_count, (bool, np.bool_))
        or not isinstance(eligible_count, (int, np.integer))
        or eligible_count < 1
    ):
        raise ValueError("eligible_count must be a positive integer.")
    if (
        isinstance(maximum_windows, (bool, np.bool_))
        or not isinstance(maximum_windows, (int, np.integer))
        or maximum_windows < 1
    ):
        raise ValueError("maximum_windows must be a positive integer.")
    number = min(int(eligible_count), int(maximum_windows))
    return np.linspace(0, eligible_count - 1, num=number, dtype=np.int64)


# ======================= 2. Participant/Class Coverage Audit =====================
def audit_temporal_sampling(
    metadata,
    caps=(250, 500, 1000),
    classes=tuple(range(1, 9)),
    subject_col="subject_id",
    label_col="label",
    forbidden_subjects=(),
):
    """
    Audit chronological window capping against the complete eligible cohort.

    Args:
        metadata (pd.DataFrame): One chronologically ordered eligible-window
            record per row. Within each person rows must preserve source
            window order. The input must not include protected Test subjects.
        caps (sequence[int]): Positive, unique per-person window limits.
        classes (sequence): All predefined documented target class codes.
        subject_col (str): Participant ID field.
        label_col (str): Original source activity-code field.
        forbidden_subjects (sequence): IDs never permitted in development.

    Returns:
        selected (pd.DataFrame): Selected original metadata row IDs for each
            policy ("full", "cap_250", ...), subject, label and source position.
            Each source row can appear under multiple *different* policies.
        per_subject (pd.DataFrame): Total windows and observed class counts
            per person and policy.
        class_counts (pd.DataFrame): Count of windows by policy and class,
            including explicit zeros for missing classes.

    Notes:
        - Works only on windows already checked for annotation purity and
          timestamp continuity; never subsample continuous raw rows first.
        - Selection is even in the eligible-window order, NOT class-balanced.
        - This audits sample representativeness; it fits no scaler/model.
        - Original Test IDs are prohibited, including case variants.
    """
    # 1. Check all source/target invariants before making sampling decisions.
    if not isinstance(metadata, pd.DataFrame) or metadata.empty:
        raise ValueError("metadata must be a non-empty DataFrame.")
    if not {subject_col, label_col}.issubset(metadata.columns):
        raise ValueError("metadata must include subject and label columns.")
    codes = list(classes)
    cap_values = list(caps)
    if len(codes) < 2 or len(set(codes)) != len(codes):
        raise ValueError("classes must have at least two distinct labels.")
    if not cap_values or len(set(cap_values)) != len(cap_values):
        raise ValueError("caps must contain distinct positive integers.")
    if any(
        isinstance(cap, (bool, np.bool_))
        or not isinstance(cap, (int, np.integer))
        or cap < 1
        for cap in cap_values
    ):
        raise ValueError("caps must be positive integers.")
    labels = metadata[label_col].to_numpy()
    if pd.isna(labels).any() or not np.isin(labels, codes).all():
        raise ValueError("metadata contains undocumented/out-of-task labels.")
    raw_subjects = metadata[subject_col].to_numpy(dtype=object)
    if any(pd.isna(value) or not str(value).strip() for value in raw_subjects):
        raise ValueError("metadata contains empty or missing subject IDs.")
    source_names = np.asarray([str(value).strip() for value in raw_subjects])
    canonical_names = np.asarray([name.casefold() for name in source_names])
    protected = {str(value).strip().casefold() for value in forbidden_subjects}
    if protected & set(canonical_names):
        raise ValueError("Protected Test subject present in metadata.")

    # 2. Preserve chronological source row identity for each subject.
    #    Reset only the external index, not the actual order of windows.
    original = metadata.reset_index(drop=True)
    all_policy_records = []
    original_display_names = {}
    for key, label in zip(canonical_names, source_names):
        original_display_names.setdefault(key, label)

    for subject in pd.unique(canonical_names):
        global_rows = np.flatnonzero(canonical_names == subject)
        subject_name = original_display_names[subject]
        n = len(global_rows)
        for policy, positions in [
            ("full", np.arange(n, dtype=np.int64)),
            *[
                (f"cap_{int(cap)}", select_evenly_spaced_indices(n, cap))
                for cap in cap_values
            ],
        ]:
            chosen = global_rows[positions]
            all_policy_records.append(pd.DataFrame({
                "policy": policy,
                "subject_id": subject_name,
                "source_row": chosen,
                "local_window_position": positions,
                "label": labels[chosen],
            }))

    selected = pd.concat(all_policy_records, ignore_index=True)
    if not selected.groupby(["policy", "subject_id"])["source_row"].apply(
        lambda values: values.is_unique
    ).all():
        raise RuntimeError("A sampling policy has duplicated a source row.")

    # 3. Derive person-wise and class-wise counts with explicit zero support.
    per_subject = (
        selected.groupby(["policy", "subject_id"], sort=False)
        .agg(windows=("label", "size"), observed_classes=("label", "nunique"))
        .reset_index()
    )
    class_counts = (
        selected.groupby(["policy", "label"], sort=False)
        .size().rename("windows").reset_index()
    )
    all_names = ["full", *(f"cap_{int(cap)}" for cap in cap_values)]
    full_index = pd.MultiIndex.from_product(
        [all_names, codes], names=["policy", "label"]
    )
    class_counts = (
        class_counts.set_index(["policy", "label"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # 4. Ensure the uncapped policy restores precisely the original corpus.
    full_rows = selected.loc[selected["policy"] == "full", "source_row"]
    if sorted(full_rows.tolist()) != list(range(len(original))):
        raise RuntimeError("The complete eligible corpus was not preserved.")
    return selected, per_subject, class_counts


# ======================== 3. Nested Training-Data Caps ===========================
def select_nested_window_indices(eligible_count, base_cap=500, expanded_cap=1000):
    """
    Build two deterministic sampling policies with STRICT source-row nesting.

    Args:
        eligible_count (int): Number of time-ordered eligible source windows.
        base_cap (int): Original maximum windows per person (e.g. 500).
        expanded_cap (int): Larger maximum windows per person (e.g. 1000).

    Returns:
        base (np.ndarray): Original evenly spaced source-local window positions.
        expanded (np.ndarray): Sorted source-local positions containing EVERY
            member of base, plus new positions evenly selected from the
            otherwise unselected eligible windows.

    Notes:
        - base reproduces Stages 10-13 EXACTLY for identical source windows.
        - expanded differs from a separate, non-nested 1000-point linspace:
          it preserves the original 500 windows, then adds new windows.
        - Source labels are not examined, so sampling is not class-stratified.
        - This helper operates on already-windowed, eligible source records.
    """
    base = select_evenly_spaced_indices(eligible_count, base_cap)
    if (
        isinstance(expanded_cap, (bool, np.bool_))
        or not isinstance(expanded_cap, (int, np.integer))
        or expanded_cap <= base_cap
    ):
        raise ValueError("expanded_cap must be an integer greater than base_cap.")

    # 1. Select extra eligible positions from those absent in the base.
    target_count = min(int(eligible_count), int(expanded_cap))
    remaining_needed = target_count - len(base)
    if remaining_needed == 0:
        return base, base.copy()
    remaining = np.setdiff1d(
        np.arange(eligible_count, dtype=np.int64),
        base, assume_unique=True
    )
    new_locations = select_evenly_spaced_indices(
        len(remaining), remaining_needed
    )

    # 2. Keep chronology and prohibit duplicated source-window positions.
    expanded = np.sort(np.concatenate([base, remaining[new_locations]]))
    if len(expanded) != target_count or not np.isin(base, expanded).all():
        raise RuntimeError("Expanded sampling lost base rows or repeated windows.")
    return base, expanded
