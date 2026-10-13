# Data pipeline design: Biomechanical Load Estimation

**Stage:** Source-verified planning and inspection utilities. **No Figshare source archive has been downloaded, no model trained and no result validated.**

## 1. Research question and source boundary

Can wearable IMU signals estimate **model-derived** lower-limb joint moments and/or femoral-neck principal strain across unseen participants? The first study is a feasibility and evaluation exercise, not a clinical prediction system.

Primary source: Hollinger et al. (2026), *Scientific Data*, DOI https://doi.org/10.1038/s41597-026-07429-3  
Data DOI: https://doi.org/10.6084/m9.figshare.30225739

The publication describes 40 healthy participants, with three additional exclusions (subjects 16, 29 and 32) in the finite-element strain analysis, giving 37 FE-eligible participants. Source files for these individuals and targets must be checked before confirming a modelling cohort.

## 2. Reported source schema (from Table 3 in the publication)

| Component | Published variables | Time representation | Units / note |
| --- | ---: | --- | --- |
| IMU | 42 | 101 points per normalised movement cycle | Acceleration and angular rate; **verify acceleration units in files** |
| Joint angles (angles.csv) | 18 | 101 points | Degrees |
| ID / joint moments | 18 | 101 points | Nm/kg |
| Force plate | 6 | 101 points | N |
| JCF | 8 | 101 points | N; dominant-side data |
| MuscleF | 77 | 101 points | N; dominant-side data |
| e1/e3 | 2 | Every tenth point from the 101-point cycle | Dimensionless strain (sometimes presented in microstrain) |

The article reports recorded frequencies of IMU 225 Hz, angles 200 Hz and force plate 2000 Hz **before** normalisation. Time-normalised arrays should not be treated as raw 225/200/2000 Hz signals. The strain sequence may be sampled on an 11-point grid; **verify the actual count and indexes** before aligning with the 101-point predictors. Do not interpolate strain targets as if additional FE simulations had been conducted.

The article describes participant MATLAB files such as `subj04.mat`, `SubCharacteristics.xlsx`, and `angles.csv` under a processed-data layout. The exact downloaded directory structure and nested MATLAB keys are unverified.

## 3. Independently inspected secondary public material

Authors' GitHub repository: https://github.com/ZainabAltai/BoneNet-A-Framework-for-Rapid-and-Accurate-Prediction-of-Femoral-Neck-strains-from-Exercise-Data

Its `labels.csv` was read directly and has **444 rows from 37 distinct subject identifiers**, with **333 Running** and **111 Walking** records. The fields are `SubjID,Task,TrialNum`. Only **222 distinct combinations** exist for these three columns, so these columns **cannot** independently identify 444 separate training records. Running effort/intensity is not encoded in that CSV. Row order or another explicit source identifier must be validated before joining labels to MATLAB arrays.

The authors also publish a roughly 14 MB `Data.mat` and notebook code using `inputs`, `outputs`, `e1` and `e3`. **Data.mat was not opened to inspect binary tensor shape or confirm its correspondence with Figshare.** This GitHub derivative should not be silently treated as the full six-activity original archive.

## 4. Proposed task sequence

**Task A — Audit and descriptive movement analysis:** Inspect the file manifest, subject and trial hierarchy, field names, 101-point trajectories, missingness, unit consistency, exclusions and modality alignment. No model.

**Task B — Joint moment regression baseline:** Where confirmed, predict one defined component of ID joint moment from IMU-only predictors; report joint/axis/side and Nm/kg explicitly. Include a training-mean baseline and classical regression models.

**Task C — Principal strain regression:** Predict FE-derived `e1` or `e3` only after confirming target coverage, correct temporal grid and subject eligibility. Compare scalar peak and sequence targets as distinct experimental formulations.

**Task D — Representation learning:** Only after an auditable baseline, assess 1D-CNN, temporal architectures and optional masked reconstruction/transfer learning using identical group splits.

## 5. Canonical observation contract (planned)

A unit of analysis is a **participant × activity × condition/intensity × unique trial × cycle**. Record fields should include:

```text
dataset_id
dataset_version
source_file
source_record_id
subject_id
activity
condition_or_intensity
trial_id
cycle_id
input_modality
input_array_shape
input_units
target_name
target_array_shape
target_units
target_is_model_derived
quality_flags
```

The canonical identifier may only be constructed **after** the data files reveal a unique trial identifier. Never join solely on `SubjID + Task + TrialNum` from the derivative `labels.csv`.

## 6. Planned staged transformations

```text
Public record and licence
  -> versioned file manifest and checksums
  -> archive/file header inspection
  -> actual MATLAB field inventory
  -> subject/trial key and modality/target validation
  -> exclusion, missingness and units audit
  -> immutable subject-held-out split registry
  -> train-only transforms and time-grid handling
  -> tensor builder and task-specific targets
  -> baseline model and independent evaluation
  -> stored predictions, audit reports and limitations
```

**Split policy:** hold out entire participants (prefer a fixed untouched test group) and run grouped validation on training subjects. For the 37-subject FE cohort, avoid treating 444 cycles or 101 sampled points as independent participants. All samples associated with one participant must remain in the same partition. Do not copy the authors' simple stratification-by-activity split for unseen-subject generalisation.

**Leakage policy:** for the IMU-only question, never pass joint moments, FE strain, JCF, ground-reaction forces or their derived features to the model as predictors. Any experiment with additional input modalities is a **different task** and must be labelled separately. Fitting normalisation on the full cohort is prohibited.

## 7. Quality gates before modelling

- Exact Figshare article version, licence and downloadable file inventory recorded.
- File types/sizes and checksums recorded; archives inspected without unsafe extraction.
- Nested MATLAB schemas, shapes and axes confirmed on actual files.
- A deterministic unambiguous mapping exists from a source record to subject/task/condition/trial/target.
- Valid e1/e3 participants and missing targets documented.
- Input/target temporal correspondence and units confirmed.
- Participant split overlap is zero and distribution of activities reported.
- No test-set information is used in preprocessing or hyperparameter selection.

## 8. Next command sequence

From the repository root in a network-connected terminal:

```bash
python scripts/figshare_inventory.py --manifest outputs/figshare_30225739_inventory.json
```

This **lists** official files without downloading them. Then select a named file from the printed list:

```bash
python scripts/figshare_inventory.py --download-file "EXACT_FILENAME_FROM_LIST" --destination data/raw/essex_femoral_neck_loading
python -m pip install -r requirements-data.txt
python scripts/inspect_mat.py --root data/raw/essex_femoral_neck_loading --report outputs/essex_mat_headers.json
```

If the selected download is a ZIP archive, inspect its entries before extracting manually into the local raw folder. Do not assume these commands ran merely because they are documented.
