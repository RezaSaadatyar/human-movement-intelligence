# ======================== Presented by: Reza Saadatyar (2026) ========================
"""Multiclass temperature scaling with subject-disjoint calibration labels."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


# ====================== 1. Probability Validation ======================
def _check_probabilities(probabilities, classes=None):
    """Validate finite, class-aligned, normalised multiclass softmax values."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not len(p) or p.shape[1] < 2:
        raise ValueError("Expected non-empty (windows, >=2 classes) probabilities.")
    if not np.isfinite(p).all() or (p < -1e-7).any() or (p > 1 + 1e-7).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Each probability row must sum to one.")
    if classes is not None:
        codes = list(classes)
        if len(codes) != p.shape[1] or len(set(codes)) != len(codes):
            raise ValueError("Classes must match probability columns exactly.")
    return np.clip(p, 1e-12, 1)


# ====================== 2. Calibrate Scores without Refitting the Model ======================
def apply_temperature(probabilities, temperature):
    """
    Apply a positive scalar softmax temperature to already-fitted model scores.

    Args:
        probabilities (array-like): N x C original softmax values.
        temperature (float): Scalar T. T=1 is the unmodified model.

    Returns:
        np.ndarray: N x C class probabilities after temperature scaling.

    Notes:
        log(probabilities)/T is equivalent to scaling the original model
        logits by 1/T, up to an arbitrary per-row additive constant.
        Probability clipping protects logs against softmax underflow.
        The top-1 predicted class is mathematically unchanged for T>0.
    """
    p = _check_probabilities(probabilities)
    if not np.isscalar(temperature) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite, positive scalar.")
    log_scores = np.log(p) / float(temperature)
    return np.exp(log_scores - logsumexp(log_scores, axis=1, keepdims=True))


# ====================== 3. Calibration-Only Temperature Fitting ======================
def fit_temperature(calibration_probabilities, calibration_labels, classes,
                    lower_bound=0.25, upper_bound=4.0):
    """
    Fit one multiclass temperature using ONLY separate Calibration labels.

    Args:
        calibration_probabilities (array-like): Calibration-only softmax NxC.
        calibration_labels (array-like): Original activity codes for those rows.
        classes (sequence): Original documented class codes in column order.
        lower_bound (float): Lower positive temperature search boundary.
        upper_bound (float): Upper positive temperature search boundary.

    Returns:
        dict: Fitted temperature, calibration baseline and calibrated NLL,
            and number of calibration windows.

    Notes:
        Optimises mean multiclass NLL in log-temperature space. It does not
        use outer-fold Validation or reserved Test labels. Results on the
        fitted calibration cohort are NOT unbiased improvement estimates.
    """
    codes = list(classes)
    p = _check_probabilities(calibration_probabilities, codes)
    y = np.asarray(calibration_labels)
    if y.ndim != 1 or len(y) != len(p) or not np.isin(y, codes).all():
        raise ValueError("Calibration labels must align with declared classes.")
    if (not np.isfinite(lower_bound) or not np.isfinite(upper_bound)
            or lower_bound <= 0 or upper_bound <= lower_bound):
        raise ValueError("Invalid positive temperature search bounds.")
    lookup = {value: index for index, value in enumerate(codes)}
    indices = np.asarray([lookup[value] for value in y], dtype=np.int64)
    log_p = np.log(p)

    def nll(log_t):
        scaled = log_p / np.exp(log_t)
        norm = logsumexp(scaled, axis=1)
        return float((norm - scaled[np.arange(len(y)), indices]).mean())

    result = minimize_scalar(nll, bounds=(np.log(lower_bound),
                             np.log(upper_bound)), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimisation failed.")
    fitted = float(np.exp(result.x))
    return {
        "temperature": fitted,
        "calibration_windows": int(len(p)),
        "calibration_nll_before": nll(0.0),
        "calibration_nll_after": nll(result.x),
    }


# ====================== 4. Held-Out Multiclass NLL ======================
def multiclass_nll(y_true, probabilities, classes):
    """
    Compute mean negative log likelihood on class-aligned labelled windows.

    Args:
        y_true (array-like): True original activity codes.
        probabilities (array-like): NxC softmax scores.
        classes (sequence): Class-column order.

    Returns:
        float: Mean multiclass NLL in natural log units.
    """
    codes = list(classes)
    p = _check_probabilities(probabilities, codes)
    labels = np.asarray(y_true)
    if labels.ndim != 1 or len(labels) != len(p) or not np.isin(labels, codes).all():
        raise ValueError("True labels must align with probability rows/classes.")
    lookup = {value: index for index, value in enumerate(codes)}
    indices = np.asarray([lookup[value] for value in labels], dtype=np.int64)
    return float(-np.log(p[np.arange(len(p)), indices]).mean())
