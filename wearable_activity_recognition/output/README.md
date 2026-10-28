# HARTH research output files

This project stores reusable experiment results, figures, evaluation tables, and selected checkpoints in the study-level `output/` directory. It is a sibling of `code/`, not a notebook subfolder. Notebook outputs are written here on execution. Local files do **not** automatically sync from Google Colab to a laptop or GitHub.

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

## Notebook 27 file-based reporting

`27_final_research_evidence_and_report.ipynb` reads seven named CSV
snapshots (three Stage 23/24 tables and four Stage 26 tables) and the
unchanged pending/approved protocol manifests. It does **not** parse saved
Notebook 23/24/25/26 cells or HTML. Its report products are named
`27_*.csv`, `27_*.png` and `27_report_provenance.json` in this directory.
The provenance JSON records SHA256 digests of the input CSV files and
manifests. Re-running the report does not rerun any model or the Test.
