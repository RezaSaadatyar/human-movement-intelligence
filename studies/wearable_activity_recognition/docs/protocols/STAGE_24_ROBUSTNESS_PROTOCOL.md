# Stage 24 | Multi-Seed Stability and Class-Imbalance Loss Ablation

## Research question and empirical motivation

The completed Stage 23 evaluation used 29 development participants, 14,201 genuine annotation-pure HARTH v2.0 windows and three subject-disjoint GroupKFold partitions. Mean fixed-eight-class Macro F1 was 1D-ResNet 0.8328, TCN 0.8138 and Random Forest 0.8094, while pooled Shuffling F1 was Random Forest 0.5032, ResNet 0.3962 and TCN 0.3392. This motivates a controlled robustness audit rather than an immediate final model selection.

Stage 24 has TWO separately interpreted studies in ONE detailed notebook:

1. **Multi-seed robustness:** 1D-ResNet versus TCN, inverse-square-root Weighted Cross-Entropy fixed, three seeds `42`, `123` and `2026`, across identical three group-held-out folds.
2. **Loss ablation:** 1D-ResNet alone, comparing original Weighted Cross-Entropy, Focal Loss (gamma=2.0 with identical inverse-square-root Train-only alpha weights) and Class-Balanced Cross-Entropy (effective-number beta=0.9999). The same three seeds and folds are used. Existing ResNet Weighted CE jobs are shared and NOT trained twice.

The unique workload is **12 architecture/loss/seed configurations × three folds = 36 GPU training jobs**: 18 from two-model multi-seed evaluation, plus 18 additional ResNet Focal/Class-Balanced jobs. Each neural fit receives exactly 400 Adam updates, batch size 128, learning rate 0.001 and last-step-only Validation scoring.

## Locked source and leakage policy

- Recreate the exact Stage 23 source-data protocol: official pinned HARTH v2.0 source, 29 Development people excluding S023/S032, 100-sample windows, 50-sample step, time continuity, perfect label purity, task source codes 1–8, then evenly spaced per-person cap 500.
- Use the same GroupKFold folds, 14,201 source-row IDs and eight-class denominator; all out-of-fold source rows have one label and prediction for each condition.
- New model weights and channel means/SD are computed ONLY from the Fold Train subset. All class/loss weights are based ONLY on Fold Train label counts.
- The original Test people were inspected in early descriptive stages, therefore should be described as *reserved from later model development*, not strictly preregistered-unseen.
- Out-of-task and undocumented source labels are retained for the source audit but not relabelled or used as targets.

## Interventions and technical definitions

- **Original weighted baseline:** `inverse_sqrt` class weights mean-normalised using Train labels; PyTorch weighted CE.
- **Focal Loss:** `alpha * (1-p_true)^gamma * (-log p_true)`, with `gamma=2.0` and alpha set to the same Train-only inverse-sqrt class weight. Aggregate using weighted sample mean (sum of weighted losses / sum of alpha). This reduces to weighted CE when gamma is zero.
- **Class-Balanced CE:** weighting from effective number `(1-beta)/(1-beta**n)`, with beta=0.9999; weights mean-normalised then used in weighted CE. Count-derived effective numbers are approximations because HARTH windows overlap.
- **No augmentation** is included in this experiment: augmentation would introduce a separate hyperparameter and semantic-validity assumption. It remains a documented future research option rather than an unverified extra intervention.

## Source implementation and notebook

- `code/python/src/core/robustness_experiments.py`: Train-only loss weighting, focal and class-balanced losses, common ResNet/TCN fitting, source/code-fingerprinted per-job JSON cache, strict OOF coverage verification and class/person metric aggregation.
- `tests/test_robustness_experiments.py`: Synthetic finite-loss/gradient checks, gamma-zero equivalence, source-aligned cached job reload and rejection of incomplete result grids.
- `studies/wearable_activity_recognition/notebooks/24_multiseed_and_imbalance_robustness.ipynb`: 12 fully documented **bold Markdown** sections with numbered code comments, reproducible setup, saved real GPU fit output, fold/seed grids, per-class minority results, subject-level metrics, paired loss differences, compute analysis and complete final Quality Gate.

## Resumable Google Colab GPU run

```bash
cd /content
git clone https://github.com/RezaSaadatyar/human-movement-intelligence.git
cd human-movement-intelligence
python -m pip install -r requirements-dl.txt
python -m unittest discover -s tests -p "test_robustness_experiments.py" -v
```

If the repo is already present, use `git pull origin main` instead of cloning again. Select **Runtime > Change runtime type > GPU**. Run the 24 notebook in order and **save the actual outputs**. The 36 jobs are saved as JSON under ignored `data/processed/harth_stage24_robustness/` and may be resumed while Colab runtime files persist. A full reset may delete local source caches and incomplete training progress. Keep notebook output as the compact public evidence rather than pushing raw source data or the full prediction cache.

## Interpretation and Stage 25 boundary

Fixed Fold assignments and repeated seeds support a better-informed model-selection decision, but cannot prove statistical superiority, model causality, robustness to genuinely new datasets or clinical validity. Eight-class Macro F1 is the primary model metric, accompanied by Shuffling/stairs class support and precision/recall/F1, equal-person scores and runtime. Avoid selecting the best seed as though it were independently evaluated. Stage 25 should freeze a model and all processing decisions using only the Development cohort before any qualified reserved-Test evaluation in Stage 26.
