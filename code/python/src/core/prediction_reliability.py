"""Descriptive out-of-fold confidence diagnostics for multiclass activity models."""
import numpy as np
import pandas as pd

def audit_prediction_confidence(y_true, probabilities, subjects, classes, n_bins=10):
    """
    Analyse softmax confidence and errors without calibrating or refitting.

    Args:
        y_true (array-like): Held-out source class codes, in row order.
        probabilities (array-like): Held-out N x C softmax values.
        subjects (array-like): One original participant ID per held-out row.
        classes (sequence): Ordered original activity codes.
        n_bins (int): Number of fixed-width confidence intervals.

    Returns:
        details (pd.DataFrame): Label, predicted class, confidence, entropy,
            margin, correctness and original participant for every window.
        reliability (pd.DataFrame): Confidence-bin size, mean confidence
            and observed accuracy; empty bins retained explicitly.
        by_subject (pd.DataFrame): Per-person errors and wrong-prediction
            confidence.
        summary (dict): Window-level Brier score, ECE, mean confidence,
            accuracy and count of high-confidence errors.

    Notes:
        These are uncalibrated descriptive diagnostics. Overlapping windows
        are correlated; ECE is sensitive to binning. No threshold selection,
        clinical decision-making or independent calibration is performed.
    """
    p=np.asarray(probabilities,dtype=float)
    truth=np.asarray(y_true)
    ids=np.asarray(subjects,dtype=object)
    codes=np.asarray(list(classes))
    if p.ndim!=2 or not len(p) or p.shape[1]!=len(codes):
        raise ValueError("Probability array must have N rows and one column per class.")
    if truth.ndim!=1 or ids.ndim!=1 or len(truth)!=len(p) or len(ids)!=len(p):
        raise ValueError("Labels, probabilities and participant IDs must align.")
    if len(codes)<2 or len(set(codes.tolist()))!=len(codes):
        raise ValueError("Task class codes must be distinct.")
    if not np.isin(truth,codes).all() or not np.isfinite(p).all():
        raise ValueError("Invalid true labels or nonfinite probabilities.")
    if (p < -1e-7).any() or (p > 1+1e-7).any() or not np.allclose(p.sum(axis=1),1,atol=1e-5):
        raise ValueError("Probabilities must lie in [0,1] and sum to one.")
    if any(pd.isna(x) or not str(x).strip() for x in ids):
        raise ValueError("Missing subject identity.")
    if isinstance(n_bins,bool) or not isinstance(n_bins,(int,np.integer)) or n_bins<2:
        raise ValueError("n_bins must be an integer >= 2.")
    idx=p.argmax(axis=1)
    pred=codes[idx]
    conf=p[np.arange(len(p)),idx]
    top_two=np.sort(p,axis=1)[:,-2:]
    margin=top_two[:,1]-top_two[:,0]
    entropy=-(p*np.log(np.clip(p,1e-12,None))).sum(axis=1)/np.log(len(codes))
    correct=pred==truth
    details=pd.DataFrame({
        "subject_id":[str(x).strip() for x in ids],
        "true_label":truth,"predicted_label":pred,
        "confidence":conf,"normalised_entropy":entropy,
        "top2_margin":margin,"correct":correct,
    })
    bins=np.minimum((conf*n_bins).astype(int),n_bins-1)
    rows=[]
    for b in range(n_bins):
        mask=bins==b
        rows.append({
            "bin":b,"lower":b/n_bins,"upper":(b+1)/n_bins,
            "windows":int(mask.sum()),
            "mean_confidence":float(conf[mask].mean()) if mask.any() else np.nan,
            "observed_accuracy":float(correct[mask].mean()) if mask.any() else np.nan,
        })
    reliability=pd.DataFrame(rows)
    ece=sum((row.windows/len(p))*abs(row.mean_confidence-row.observed_accuracy)
            for row in reliability.itertuples() if row.windows>0)
    encoded=(truth[:,None]==codes[None,:]).astype(float)
    brier=float(np.mean(np.sum((p-encoded)**2,axis=1)))
    people=[]
    for subject,part in details.groupby("subject_id",sort=True):
        wrong=part.loc[~part["correct"]]
        people.append({
            "subject_id":subject,"windows":len(part),
            "accuracy":float(part["correct"].mean()),
            "errors":len(wrong),
            "mean_error_confidence":float(wrong["confidence"].mean()) if len(wrong) else np.nan,
            "high_confidence_errors":int((wrong["confidence"]>=0.9).sum()),
        })
    summary={
        "windows":len(p),"accuracy":float(correct.mean()),
        "mean_confidence":float(conf.mean()),
        "ece_10_bins" if n_bins==10 else "ece":float(ece),
        "multiclass_brier":brier,
        "high_confidence_errors_0_90":int(((~correct)&(conf>=0.9)).sum()),
    }
    return details,reliability,pd.DataFrame(people),summary
