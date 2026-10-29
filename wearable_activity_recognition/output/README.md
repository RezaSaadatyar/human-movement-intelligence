# HARTH research output files

**Numbering update:** Historical displayed-precision CSV filenames and associated approval digests retain their original stage numbers. The active research notebooks have been renumbered 21–27 → 20–26; see [the mapping](../STAGE_NUMBERING.md). The cited original source notebook paths below refer to versions in Git history rather than to the new active filenames.


This project stores reusable experiment results, figures, evaluation tables, and selected checkpoints in the study-level `output/` directory. It is a sibling of `code/`, not a notebook subfolder. Notebook outputs are written here on execution. Local files do **not** automatically sync from Google Colab to a laptop or GitHub.

## Current active Stage 22–26 files and final audit

The active Stage 22, 23 and 25 notebooks have now exported **full-precision
numeric CSV files** under their current `22_*`, `23_*` and `25_*` filename
prefixes. The executed Stage 26 notebook reads **seven current CSV files**
from those stages, validates the frozen pending/approved manifests, and writes
eight `26_*` report products. It does not retrain or rerun the qualified Test.
The historical `*_display.csv` snapshots described below are legacy
transcriptions and are **not** needed by the active Stage 26 reader.

The new `code/notebooks/final_harth_research_audit_and_closure.ipynb`
performs a complementary cross-stage audit of earlier experiments and the
Stage 26 input-file SHA256 ledger. It is **not yet executed**; its intended
`final_*` exports must not be described as existing results until the user
runs and commits the notebook and outputs.

## Historical Stage 23 and Stage 24 exports

The `23_*_display.csv` and `24_*_display.csv` tables were **transcribed one time from the stored, completed HTML outputs** of the pinned research notebooks. No training, inference, held-out Test scoring, or notebook re-execution was performed during this conversion.

- Stage 23 source: `code/notebooks/23_unified_matched_model_benchmark.ipynb`, reviewed Git blob `9d58028051a382b2f5f8ad70ed8885cc12fa796b`.
- Stage 24 source: `code/notebooks/24_multiseed_and_imbalance_robustness.ipynb`, reviewed Git blob `3c37a397d101003039cc64b96981bcc326c137b6`.
- The immutable approved historical Stage 23/24 execution SHA256 values remain recorded in `evidence/stage_25_protocol_approved_qualified_test.json`.

**Precision limit:** the imported CSV files contain the **displayed precision** of their source tables, generally four decimal places. They are sufficient for the displayed tables, descriptive plots and Stage 25 development-model selection review, but they are **not** full-precision arrays, raw per-window predictions, or an authoritative replacement for the original executed notebook evidence. In particular, the complete source-level out-of-fold prediction arrays and all per-participant/per-class experiment rows were not recoverable from these displayed tables alone. Do not report more precision than the CSV provides.

The Stage 23 and 24 source notebooks have been kept byte-for-byte unchanged because their exact committed identities participate in historical evidence checks. Any later export of full-precision results should come from existing verified experiment artefacts, not from rerunning the protected final Test.

## Git and local storage

The default project policy permits review of individual, publishable files at or below **50,000,000 bytes** for Git Commit/Push. Files larger than that stay local. Sensitivity, permissions and privacy take precedence over size. Some model/binary extensions are globally ignored by `.gitignore` and require a deliberately reviewed exception before being tracked. Colab `/content` is ephemeral; copy important artefacts to durable storage before ending a session.

## Historical Stage 26 qualified held-out display exports

The four `26_qualified_*_display.csv` tables were transcribed **once** from
the saved, completed HTML tables of the previously approved qualified held-out
evaluation. Source was the committed
`code/notebooks/26_final_development_training_and_inference.ipynb` Git blob
`1fe6b6d15925dd48a8bbd05f6180ca8b89b26653` (before the subsequent
presentation/export-only edits). The exported tables are:

- `26_qualified_subject_coverage_display.csv`: S023 and S032, 500 windows each.
- `26_qualified_model_summary_display.csv`: both frozen models, 2 rows.
- `26_qualified_per_subject_display.csv`: model/subject results, 4 rows.
- `26_qualified_per_class_display.csv`: eight source-code classes for each
  model, 16 rows; support sums to 1,000 per model.

No new Test inference or modelling was performed during transcription.
These are four-decimal **display snapshots**, not a replacement for the
original higher-precision aligned predictions or protected model artefacts.
The source notebook's saved execution results remain unchanged.

When the original approved Notebook 26 is deliberately executed in an
authorised environment, its existing local files under
`outputs/harth_stage26_qualified_test/` are maintained for compatibility,
and reusable full-precision result copies are additionally written to this
study-level `output/` folder without a `_display` suffix. The frozen model
files remain in `outputs/harth_stage26_final/` to preserve trusted reload
paths; this directory is ignored by Git.

## Active Notebook 26 file-based reporting

`26_final_research_evidence_and_report.ipynb` reads seven exact source CSVs:
`22_matched_cv_model_summary.csv`, `23_robustness_model_summary.csv`,
`23_robustness_fold_metrics.csv`, and four `25_qualified_*.csv` files
(subject coverage, model summary, per-subject and per-class outcomes). It
checks the existing pending and approved protocol manifests, computes the
SHA256 of each input CSV and manifest, and exports four figures, three small
CSV tables and `26_report_provenance.json`. The executed notebook and these
eight products are committed. It does **not** parse previous notebook HTML,
reconstruct predictions, train models or rescore S023/S032.

The historic Stage 27 numbering remains identifiable from Git history and
the frozen provenance register; the current reporting notebook is Stage 26.
