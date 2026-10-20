# Stage 25 | Development-Only Model Selection and Proposed Protocol Freeze

## Evidence basis

Stage 23 ran seven classifiers across three subject-disjoint HARTH development folds on precisely 14,201 genuine source windows from 29 people, with no use of originally reserved S023/S032 in model fitting or scoring. Stage 24 ran 36 GPU jobs over three seeds (42, 123, 2026) and three folds. Both fully executed source notebooks are read by SHA256 and source-row fingerprint. Rounded metrics in displayed notebook tables are linked to the saved sources, not recomputed from missing or inaccessible local Colab caches.

Stage 24 development mean Macro F1 across three seed-level fold averages:
- **1D-ResNet / inverse-square-root Weighted CE: 0.8286**, across-seed SD 0.0052.
- TCN / same Weighted CE: 0.8105, across-seed SD 0.0079.
- ResNet / Focal Loss: 0.8058, across-seed SD 0.0053.
- ResNet / Class-Balanced CE: 0.7915, across-seed SD 0.0191.

Stage 23 Random Forest is a fixed classical reference, not a multi-seed replication. It reported Shuffling pooled F1 0.5032 versus 1D-ResNet 0.3962 and TCN 0.3392. Differences are development-dependent, not proof of universal superiority. Other four temporal models remain fully documented in Stage 23.

## Proposed final training procedure, pending explicit review

**Primary candidate:** 1D-ResNet (`in_channels=6`, `n_classes=8`, `hidden_channels=32`) using inverse-square-root weighted cross-entropy, Adam, 0.001 learning rate, batch size 128, 400 optimizer updates, source-only Train-normalisation and class weights, and reproducibility seed 42 (fixed as a convention, not optimised from Test). Do not tune final checkpoints on reserved Test.

**Predeclared classical reference:** Random Forest, 120 trees, minimum leaf 3, `balanced_subsample` loss weights, seed 42, deterministic 30 statistical features from six channels.

**Source:** pinned HARTH v2.0 revision `dad2cfbe89a26f72f19770419469ac037de200df`, nominal 50 Hz, 100-sample / 50-stride windows, timestamp continuity and 100% single-label purity. Eight original activity source codes (1–8), no reinterpretation of undocumented source codes; exclude out-of-task windows only after proper segmentation.

**Training cohort:** all 29 Development people, chronological label-independent cap of up to 500 eligible windows per person (expected 14,201), rather than a silent switch to the complete 132,397-window development cohort. Fit model/scaler/weights on Development alone.

**Qualified held-out report proposal:** one held-out scoring run for S023/S032 using identical post-window cap 500, after explicit user approval; include both individual results and class-level supports. Earlier descriptive examination of these IDs means they must not be described as strictly blind or externally unseen. Dataset originates from one source collection; no independent external generalisation is established.

**Metrics:** fixed eight-class Macro F1 as primary, overall accuracy, balanced accuracy, confusion matrix, per-class precision/recall/F1 and true support, each held-out subject's fixed-task Macro F1, compute and reproducibility. No Test-driven hyperparameter, architecture, preprocessing, loss or threshold changes. Stage 19 temperature calibration is not included due to mixed Development outcomes.

## Outputs and review workflow

- `notebooks/25_model_selection_and_protocol_freeze.ipynb`: eleven complete, explanatory sections with Bold Markdown and numbered code comments. It **does not load raw HARTH recordings or train a model**.
- `code/python/src/core/harth_selection_evidence.py`: strict read-only parser for original executed Stage 23/24 notebook HTML results, including source SHA256 alignment and expected model/Fold evidence completeness.
- `tests/test_harth_selection_evidence.py`: offline tests for saved HTML parsing and the **actual committed executed research notebooks**.
- The Notebook creates `studies/wearable_activity_recognition/research_outputs/stage_25_protocol_proposal.json` with immutable source-module digests and `PENDING_USER_REVIEW` status. No overwrite if an existing JSON differs.

## Running Stage 25

Stage 25 requires neither GPU nor downloaded raw source files. In the repository, install classical requirements and run:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_harth_selection_evidence.py" -v
```

Run Notebook 25 in order, inspect its figures, numerical tables and approval-pending manifest, and commit/push actual executed outputs and the generated proposal JSON if reviewed. Stage 26 is **not** authorised merely by a passing code quality gate. The user reviews the proposed source/sample/test scope and explicitly approves (or requests changes) before any held-out Test access.

## Interpretation

This is a **development-informed model choice**, not pre-registered independent validation or a finding of statistical significance. The selected one-seed final fit is proposed for reproducibility; Stage 24's multi-seed results characterise variability. The subject-disjoint experiments are still from one study, with overlapping and correlated IMU windows. Claims of clinical validity, deployment readiness and external generalisation are unsupported.
