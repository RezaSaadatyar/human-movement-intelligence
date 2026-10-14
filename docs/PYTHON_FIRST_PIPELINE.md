# Python-only data ingestion and baseline pilot

This is the default computational route for **Human Movement Intelligence**. Python is used for dataset access, schema inspection, quality assurance, subject-aware splitting, preprocessing, training and evaluation. MATLAB is **not** required to run this pilot.

## Why there is a separate numeric pilot

The original Figshare v13 subject files have MATLAB MCOS objects stored inside MAT v5 files. SciPy currently exposes many of these fields as opaque references rather than actual numerical time series. This does not indicate a problem with PyTorch or machine learning; it is a format interoperability issue.

Source-specific decoders are being considered but must be tested before being trusted. A Python-only generic MCOS decoder **has not** been validated on this dataset. In the meantime, the authors publish a separate `Data.mat` plus `labels.csv` in their BoneNet repository. This is a *derivative sample*, not the entire six-activity original dataset.

**Source:** https://github.com/ZainabAltai/BoneNet-A-Framework-for-Rapid-and-Accurate-Prediction-of-Femoral-Neck-strains-from-Exercise-Data

## Installation and first full Python run

From the project root in PowerShell:

```powershell
git pull origin main
python -m pip install -r requirements-pilot.txt
python scripts/bonenet_python_pilot.py --download
```

This performs, in one Python workflow:
1. Downloads the authors' `Data.mat` and `labels.csv` only if not already present.
2. Stores them under ignored `data/raw/bonenet_reference/`.
3. Records local SHA-256 checksums and MATLAB variable names.
4. Reads source labels and checks subject counts and duplicate `SubjID + Task + TrialNum` keys.
5. Attempts to build genuinely numeric feature and target arrays using the published notebook's axis-stacking approach.
6. Reports tensor shapes, feature/target names and nonfinite-value counts, or explicit blockers if the actual file differs from the notebook expectation.
7. Writes only safe structural metadata to `outputs/bonenet_python_pilot_audit.json`.

The script does not invoke MATLAB, use MATLAB Engine, or require uploading the 225 MiB Figshare archive.

## Baseline implementation (conditional)

A cautious training baseline is implemented, but **do not invoke it before inspecting the first audit**. The source's `labels.csv` has 444 rows from 37 subjects and duplicated `SubjID + Task + TrialNum` keys; row order is relied on by an author's example notebook but is not independently proven. It cannot establish intensity-level identifiers from that CSV alone.

If review finds the numeric contract valid, an explicitly opt-in exploratory reproduction is available:

```powershell
python scripts/bonenet_python_pilot.py --train --accept-order-based-label-alignment
```

This compares a training-set mean baseline against Ridge regression on per-channel temporal mean/std features. It uses GroupShuffleSplit with subjects as groups, a held-out subject test set and train-fitted standardisation. Results, if obtained, are exploratory and **must not** be presented as robust external validation, clinical predictions, or full-cohort evidence. Do not invoke this stage automatically.

## Original Figshare data remains in scope

The original `data/raw/essex_femoral_neck_loading/MATLAB/Processed Data/` files remain the intended primary source for the full biomechanical study. We retain the 40-file audit and will attempt Python-only MCOS decoding in a separate, explicitly evaluated adapter, without assuming unrelated `groundTruth` decoders work for this dataset.

## Test results

The initial Python-only pipeline was exercised locally against **synthetic** MAT files and artificial participant arrays. Five offline unit tests passed. It has **not** been executed against the source authors' `Data.mat` in this session. The GitHub CI status must be checked separately.

Next: inspect the one JSON audit from the first run, then make any source-specific adjustments, rather than asking the maintainer to repeatedly inspect MAT variables by hand.

## Automated public-source audit (GitHub Actions)

A metadata-only audit is configured under `.github/workflows/bonenet-source-audit.yml`. This workflow downloads the authors' **public** 14 MB derivative into a temporary GitHub Actions runner and executes `python scripts/bonenet_python_pilot.py --download`, without running any training. The raw MAT and label files are **not** uploaded to GitHub or workflow artifacts. Only the resulting JSON metadata report is uploaded. It triggers on changes to its workflow definition (and can also be triggered manually).

The output must be inspected before the opt-in `--train` stage is considered. If GitHub Actions is disabled, the same single command can be run locally. Workflow execution and success are not assumed merely from committing this file.
