# Essex notebook workflow simplification

The eight notebooks now use one current set of outputs. Each execution replaces that notebook's files; the next notebook reads the current upstream data. Existing outputs remain usable until their producing notebook is rerun. There is no runtime dependency on earlier runs or administrative approval reports.

## Removed or replaced components

| Component | Change and reason |
| --- | --- |
| `essex_output_layout.py` | Removed the module that migrated old output folders and compared earlier runs. Outputs are written directly to the current `output/` folder. No `_consolidation_drafts` folder was present in the checkout. |
| `essex_native_conversion.py` | Removed the optional conversion-report validation branch. Notebook 02 now follows SciPy inspection and the Python table reader. The standalone MATLAB conversion script remains a functional export utility, outside this notebook sequence. |
| Source and paired-array hashes | Removed `file_hashes`, `sha256_file`, `paired_source_digest`, file fingerprints and comparisons with stored source digests. New downloads still detect incomplete transfer and unsafe ZIP members. |
| Repeated registry checks | Removed `check_record_registry`, `assert_source_fe_status_matches` and notebook assertions against expected cohort counts. The current dataset is described with counts and tables. |
| Workbook alignment approval | Replaced `audit_workbook_ids` with `load_participant_ids`, which reads recognised IDs and reports duplicates and summary rows without an approval flag. |
| Export approval and readiness reports | Removed `pilot_allows_expansion`, unused `export_full_source`, `summarise_alignment` and `summarise_temporal_evidence`. Actual numeric readers, source-coordinate matching and decoding summaries remain. |
| Stage 04 cache refusal | Replaced `save_ignored_cache` with `save_paired_data`; rerunning Stage 04 atomically replaces the current NPZ. |
| Stage 07 fingerprinted resumption | Replaced `run_cached_fold` with `run_training_fold`. Every execution trains the requested fold and replaces its result file. Rerunning the full training cell performs all 75 fits. |
| Stage 08 approval and release lock | Removed `load_frozen_protocol`, `release_one_time_test` and release-token/receipt checks. `final_model_settings()` contains the same RF and TCN recipes previously stored in JSON. Rerunning Stage 08 refits those models and overwrites its outputs. |
| Administrative artifacts | Deleted 12 JSON reports, six `02_native_*.csv` reports, `02_python_full_export_decision.csv` and the separate evidence-register CSV. |
| Administrative CSV columns | Removed `published_md5`, `source_sha256` and `experiment_fingerprint` from three retained tables. All other CSV field strings are unchanged. |
| Obsolete diagrams | Deleted seven unreferenced `01_workflow.svg`–`07_workflow.svg` diagrams from the earlier stage layout. Updated the eight current notebook diagrams and study/final overviews. |

## Preserved calculations and useful checks

Original MATLAB reading, trial identification, IMU–FE pairing, missing-target exclusions, native time positions and unitless targets remain. The original 29/4/4 participant allocation is unchanged. Model architectures, hyperparameters, Training-only scaling, scoring and participant-level uncertainty calculations remain; bootstrap sample counts now come from the supplied participant IDs rather than requiring exactly four people.

Checks remain where needed to compute valid results: missing required columns, malformed/nonfinite arrays, ambiguous trial keys, incompatible time positions and participant overlap. These are local calculation requirements, not comparisons against a historical report.

Current arrays are aligned to trial-table order by key. Stages 06 and 07 also align the current Stage 05 fold labels by key and keep each participant in one fold. When paired arrays are missing or older than their current inputs, they are reconstructed from the current registry, mapping and source archive.

The full numerical result CSVs and PNG plots were not recalculated or altered. The final RF and TCN model settings were compared with the previous configuration and are identical. No Test-driven model selection was introduced.

## Verification and execution limits

- The complete offline unit suite passed: **134 tests**. Offline tests exercise overwrite/reuse behaviour, incomplete downloads, archive extraction, a synthetic source-to-numeric-audit-to-time-mapping-to-pairing flow, trial reordering, participant allocation, fold alignment and stage-07 reruns.
- Model tests exercise original FE-grid outputs, participant isolation, bootstrap calculations and a small CPU training example.
- Saved Test predictions reproduce the reported equal-participant-weighted MAE. Numerical result files are retained from the previous completed run.
- All eight notebooks pass syntax/schema checks; removed imports and undefined references were checked. Updated SVG diagrams were rendered and visually inspected.
- CI now checks notebook syntax and current result tables instead of requiring deleted JSON reports or claiming that edited notebook cells were already executed.

The revised notebooks have **not** been run end to end on the full Essex dataset or in hosted Colab. Edited code-cell outputs were cleared to avoid presenting old execution as evidence of the new code. Re-executing the notebooks will replace current results; repeated held-out evaluation is not a new independent validation cohort.
