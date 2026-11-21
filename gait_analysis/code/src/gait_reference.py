# ==================== Presented by: Reza Saadatyar (2026) ====================
"""Convert independently measured vertical force to contact event times."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ========================== 01. Force Contacts ==============================
def force_contact_events(time_s, vertical_force_n, threshold_n=20.0,
                         min_contact_s=0.08, min_swing_s=0.06):
    """Detect entry/exit of contact from force; discard edge-censored events.

    Threshold is a predeclared planning value, not a verified optimum.
    Raw force must have a reviewed side assignment and an upright-positive
    vertical sign convention. Fail for possible sign mistakes.
    """
    t = np.asarray(time_s, float)
    f = np.asarray(vertical_force_n, float)
    if len(t) != len(f) or len(t) < 4:
        raise ValueError("Force and time lengths must match.")
    if not np.isfinite(t).all() or not np.isfinite(f).all():
        raise ValueError("Force/time contain non-finite samples.")
    if np.any(np.diff(t) <= 0) or threshold_n <= 0:
        raise ValueError("Invalid force timestamp or positive threshold.")
    if np.percentile(f, 95) <= threshold_n:
        raise ValueError("No detectable upward force contact above threshold.")
    contact = f >= threshold_n
    cuts = np.r_[0, np.flatnonzero(np.diff(contact.astype(int))) + 1, len(t)]
    segments = [(int(a), int(b), bool(contact[a]))
                for a, b in zip(cuts[:-1], cuts[1:])]
    events = []
    for i, (start, stop, is_contact) in enumerate(segments):
        if not is_contact or i == 0 or i == len(segments) - 1:
            continue
        duration = t[stop - 1] - t[start]
        previous = segments[i - 1]
        following = segments[i + 1]
        pre_s = t[start] - t[previous[0]]
        post_s = t[following[1] - 1] - t[following[0]]
        if duration < min_contact_s or min(pre_s, post_s) < min_swing_s:
            continue
        events += [{"event_type": "IC", "t_reference_s": float(t[start])},
                   {"event_type": "TC", "t_reference_s": float(t[stop])}]
    return pd.DataFrame(events, columns=["event_type", "t_reference_s"])


# ========================= 02. Reference Strides ============================
def reference_strides(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate IC-to-IC stride durations with physical seconds."""
    if events.empty:
        return pd.DataFrame(columns=["stride_index", "start_s", "end_s",
                                     "stride_time_s", "cadence_steps_per_min"])
    ic = np.sort(events.loc[events.event_type == "IC", "t_reference_s"].to_numpy(float))
    if len(ic) < 2:
        return pd.DataFrame(columns=["stride_index", "start_s", "end_s",
                                     "stride_time_s", "cadence_steps_per_min"])
    intervals = np.diff(ic)
    good = (intervals >= 0.35) & (intervals <= 3.0)
    return pd.DataFrame({
        "stride_index": np.flatnonzero(good), "start_s": ic[:-1][good],
        "end_s": ic[1:][good], "stride_time_s": intervals[good],
        # Same-foot IC→IC is a stride; two steps occur per stride.
        "cadence_steps_per_min": 120.0 / intervals[good],
    })


def stride_summary(strides: pd.DataFrame) -> dict:
    """Return descriptive reference-only stride measures for a trial."""
    x = strides.stride_time_s.to_numpy(float)
    if not len(x):
        return {"n_strides": 0, "mean_stride_time_s": np.nan,
                "stride_time_cv": np.nan, "cadence_steps_per_min": np.nan}
    mean = float(x.mean())
    return {"n_strides": len(x), "mean_stride_time_s": mean,
            "stride_time_cv": (float(x.std(ddof=1) / mean)
                               if len(x) >= 2 and mean > 0 else np.nan),
            "cadence_steps_per_min": float((120.0 / x).mean())}
