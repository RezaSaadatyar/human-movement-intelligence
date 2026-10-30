# HARTH v2.0 | Research Reproducibility and Evidence Boundaries

This guide describes what can be inspected and verified in the completed
**Wearable Activity Recognition** study in
[Human Movement Intelligence](../README.md). The research consists of
[26 numbered notebooks](../wearable_activity_recognition/code/notebooks/)
and a separately executed
[FINAL evidence audit](../wearable_activity_recognition/code/notebooks/final_harth_research_audit_and_closure.ipynb).
The study results, scientific limitations and recorded outcomes are detailed in
[Research Methods and Experimental Evidence](../wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md).

The completed research is **not** an independent external validation, a
prospective deployment study or a guarantee of identical GPU training results
across environments.

## 1. Source and experimental boundary

- **Dataset:** Official [NTNU AI Lab HARTH v2.0](https://github.com/ntnu-ai-lab/harth-ml-experiments),
  source revision `dad2cfbe89a26f72f19770419469ac037de200df`;
  this is distinct from UCI HARTH v1.2.
- **Inputs:** Six lower-back and thigh acceleration channels, nominal 50 Hz;
  label-pure 100-sample windows with 50-sample stride.
- **Task:** Eight documented activity labels (source codes 1–8).
- **Development:** 29 participants, 132,397 eligible task windows and a
  14,201-window bounded model-comparison cohort.
- **Qualified held-out assessment:** S023 and S032, 500 selected windows each.
  These two participants were descriptively examined in exploratory work.
  The assessment therefore is same-study and not fully blind.

The source recordings and trained model-weight files are **not committed**
to the public repository. Source windows overlap, and rare-activity support
in the qualified assessment is extremely limited: Shuffling **3**, stairs
ascending **10**, stairs descending **6** windows.

## 2. Recorded outputs available in GitHub

| Evidence | Versioned record | What it establishes |
| --- | --- | --- |
| Seven-model Development benchmark | [Stage 22 notebook](../wearable_activity_recognition/code/notebooks/22_unified_matched_model_benchmark.ipynb) and [model summary CSV](../wearable_activity_recognition/output/22_matched_cv_model_summary.csv) | Participant-disjoint matched comparison under the bounded Development protocol |
| Multiseed and imbalance robustness | [Stage 23 notebook](../wearable_activity_recognition/code/notebooks/23_multiseed_and_imbalance_robustness.ipynb) and [robustness summary](../wearable_activity_recognition/output/23_robustness_model_summary.csv) | Three seeds and three folds across four model/loss conditions |
| Frozen model selection | [Stage 24 notebook](../wearable_activity_recognition/code/notebooks/24_model_selection_and_protocol_freeze.ipynb) | Development-only selection and documented source lineage |
| Qualified held-out evaluation | [Stage 25 notebook](../wearable_activity_recognition/code/notebooks/25_final_development_training_and_inference.ipynb) and [saved model metrics](../wearable_activity_recognition/output/25_qualified_model_summary.csv) | One separately approved same-study evaluation; not external validation |
| Final evidence synthesis | [Stage 26 notebook](../wearable_activity_recognition/code/notebooks/26_final_research_evidence_and_report.ipynb) and [report provenance](../wearable_activity_recognition/output/26_report_provenance.json) | Read-only synthesis of saved source CSVs and their SHA256 fingerprints |
| Cross-stage research closure | [FINAL notebook](../wearable_activity_recognition/code/notebooks/final_harth_research_audit_and_closure.ipynb) and [FINAL provenance](../wearable_activity_recognition/output/final_research_audit_provenance.json) | Saved audit with 11 executed code cells and Quality Gate PASSED |

The FINAL audit checks **20 recorded CSV inputs**, the Stage 26 report
fingerprint and the frozen protocol documents. Its committed outputs include
eight audit tables, a figure and a provenance JSON. The saved Quality Gate
reports `FINAL HARTH | CROSS-STAGE EVIDENCE AUDIT: PASSED`.
This establishes **consistency of available recorded files**, not independent
replication of all training runs, model weights or external Test performance.

## 3. Frozen scientific approval and lineage

The research protocol records remain under
[`wearable_activity_recognition/evidence/`](../wearable_activity_recognition/evidence/):

- [Initial proposal](../wearable_activity_recognition/evidence/stage_25_protocol_proposal.json)
- [Reconciled proposal](../wearable_activity_recognition/evidence/stage_25_protocol_proposal_reconciled.json)
- [Approved qualified assessment](../wearable_activity_recognition/evidence/stage_25_protocol_approved_qualified_test.json)

The historical Stage 25 labels and SHA256 identifiers are retained for
traceability. Approval of the recorded qualified evaluation is **not**
authorisation to repeat Test scoring for model selection. The original
scientific executions and earlier file states remain recoverable in Git
history; renaming a stage must not overwrite an approved fingerprint.

The [Stage 26 provenance JSON](../wearable_activity_recognition/output/26_report_provenance.json)
contains SHA256 values for its seven source CSVs and the protocol
manifests. The
[FINAL provenance JSON](../wearable_activity_recognition/output/final_research_audit_provenance.json)
contains 20 input-CSV fingerprints and the report/manifest digests.

## 4. What a fresh checkout can verify

A GitHub checkout is sufficient to review the notebook code and saved
outputs, inspect the committed result tables and compare the recorded
source-file SHA256 digests. No model retraining or Test re-evaluation is
needed to read these results.

From the repository root, to install the standard declared dependencies:

```bash
python -m pip install -r requirements.txt
```

The optional [deep learning requirements](../requirements-dl.txt)
support PyTorch-based methods but are not a complete, exact environment
lockfile. Independent end-to-end reproduction additionally requires
the original source recordings, approved preprocessing conditions,
suitable computation and separately preserved trusted model artefacts.

The repository retains [general shared-library tests](../tests/) for
reusable utilities and other projects. HARTH-specific release-readiness
and notebook-provenance test files have been retired from the current tree
following study closure; they are recoverable from Git history. **Running
the remaining tests is not a substitute for the recorded HARTH FINAL
scientific audit.** No automatic pass result is claimed here.

## 5. Interpretation and safety limits

- The two final participants originate from the same dataset and were
  inspected descriptively before the qualified assessment.
- The 1,000 selected final windows overlap and are not 1,000 independent
  participants or real-world evaluation trials.
- Minority activity samples are insufficient for robust population-level
  class-performance claims.
- Random Forest and neural models differ in representation and fitting
  objectives; some exploratory experiments use non-equivalent protocols.
- Calibration outcomes from Development were mixed and were not included
  in the selected final model.
- Model uncertainty, prospective clinical validity, model superiority
  across independent populations and deployment readiness are **not
  established**.
- Do not deserialize unknown `joblib` or pickle model files: loading
  untrusted model artefacts may execute arbitrary code.

The repository documents a **completed experimental research portfolio
study**. Publishing a versioned GitHub Release or conducting new external
validation would be separate, explicitly scoped actions.
