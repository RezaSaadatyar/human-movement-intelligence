# Stage 27 | Final HARTH Study Report and Evidence Review

> **As-executed status:** The [current Stage 27 notebook](../../notebooks/27_final_research_evidence_and_report.ipynb) has **11 executed code cells**: one Google Colab setup cell and ten numbered read-only research sections. Its source-grounded synthesis Quality Gate is saved as `PASSED`. References below to an unexecuted *first commit* describe historical planning, not the present notebook. The Stage 26 approved results are read as existing evidence, not re-scored.

## Purpose and current status

Stage 26's executed notebook has thirteen executed code cells and the final **qualified held-out Quality Gate PASSED**. Stage 27 is an independent **read-only synthesis** of existing saved results. It will NOT redownload the HARTH data, load S023/S032, retrain models, re-score the held-out people, or select the primary model again.

At the initial implementation checkpoint, Stage 27 existed only as a prepared notebook. **That historical checkpoint has now been superseded:** the current notebook's 11 code cells have saved execution counts, no saved errors and a `PASSED` final evidence-synthesis gate. A versioned GitHub release remains a separate, user-approved action.

## Verified fixed results (from saved executed Stage 26)

| Model | Qualified pooled fixed-eight-class Macro F1 | Accuracy | Balanced accuracy across observed true classes |
| --- | ---: | ---: | ---: |
| Preselected 1D-ResNet Weighted CE | 0.7182 | 0.9020 | 0.9234 |
| Predeclared 120-tree Random Forest | 0.6817 | 0.9270 | 0.7599 |

- Original reserved participants: **S023 and S032**, 500 Test windows each.
- Individual fixed-eight-class Macro F1: S023 **0.7384** (ResNet), **0.7391** (RF); S032 **0.5662** (ResNet), **0.4597** (RF).
- True test class supports: Walking 538, Running 86, Shuffling 3, Stairs ascending 10, Stairs descending 6, Standing 127, Sitting 115, Lying 115.
- The three rare activity categories (Shuffling, ascending stairs, descending stairs) do not support broad performance conclusions. The number of source windows is not the number of independent participant observations.
- Original source activity codes 10 and 11 were detected on S032 and excluded after time-safe segmentation without invented labels.
- Both reserved people had been seen descriptively in earlier exploration. **The qualified results are neither historically blind nor external validation.** No hyperparameter choice was made after Test observation.
- These metrics are rounded figures **transcribed from saved Notebook 26 displays**, not recalculated from raw prediction files here.

## Upstream Development evidence

Stage 23 evaluated **seven models** across **three participant-disjoint Development folds**, using 14,201 source-window rows from 29 participants. Stage 24 completed **36 model/fold/seed fits**. Stage 25 selected the ResNet primary and RF comparator **before** qualified Test evaluation. The notebook reads actual Stage 23 and Stage 24 saved tables with enforced notebook fingerprints; it does not imply performance improvements beyond the reported evidence.

## Stage 27 deliverables

- `code/python/src/core/harth_study_report.py`: read-only saved-notebook and frozen-approval manifest validator.
- `notebooks/27_final_research_evidence_and_report.ipynb`: one Colab setup cell and 10 numbered evidence-reporting sections (bold Markdown headings, numbered English explanatory code comments, tables, plots, research limitations and final Quality Gate).
- `tests/test_harth_study_report.py`: offline source integrity, saved output and report structure checks.
- Ignored local `outputs/harth_stage27_report/`: CSV summaries (rounded display values) and evidence provenance JSON.

## Execution procedure

From the latest repository checkout, start a Python kernel in Cursor or Jupyter; **GPU and raw HARTH CSVs are not required**. The notebook's first code cell contains `%cd /content` and is **Google Colab-specific**. In Colab, run that setup cell to clone or safely fast-forward the existing checkout. On local Windows, Cursor or standard Jupyter, **skip the Colab-only setup cell**, open the repository root as the working directory, verify a complete and up-to-date Git checkout and run the ten research sections. Do not edit execution counts or overwrite frozen evidence to compensate for a skipped setup cell.

```bash
git pull origin main
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_harth_study_report.py" -v
```

Open `studies/wearable_activity_recognition/notebooks/27_final_research_evidence_and_report.ipynb` and run the ten **research** code cells in order, after the Colab-specific setup if using Colab. Carefully inspect every figure, table and the final printed Quality Gate. Save the executed notebook and commit/push it; the `outputs/` files are intentionally Git-ignored and should not be accidentally published.

## Release review (only after executed Stage 27 notebook is reviewed)

1. Verify saved Stage 27 code execution counts, absence of errors and its exact final `PASSED` marker.
2. Check exported summaries and references against source notebooks; do not treat saved rounded tables as exact full-precision predictions.
3. Update the active HARTH README and roadmap to link the real executed Stage 27 outputs and set prior historical 'pending' descriptions straight, **without rewriting previous research evidence**.
4. Draft the manuscript-style final Markdown report with Methods, Results, Discussion, Limitations and Reproducibility. Use actual numbers only, label Development and qualified Test comparisons separately.
5. Only then consider an appropriately versioned GitHub release, avoiding any claim of external generalisation or clinical readiness.

## Important restrictions

This is a reproducible research code portfolio and exploratory HARTH benchmark; it is not clinical deployment, an independent external validation study, or evidence of statistically significant cross-population generalisation. Stage 27 does not create additional outcome evidence.
