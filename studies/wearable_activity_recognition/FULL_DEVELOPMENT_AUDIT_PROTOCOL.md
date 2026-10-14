# HARTH Stage 8: full development-cohort data and fold audit

## Aim

Inspect **all 29 available HARTH v2.0 development subjects**, excluding
the two earlier reserved Test IDs `S023` and `S032`. This stage does
not train or evaluate machine-learning classifiers.

## Dataset and download

The official HARTH v2.0 repository has 31 per-person CSV files.
After excluding the two preselected Test IDs, 29 remain.

Ten development participants were previously acquired in Notebook 07.
The other **19 source CSV files total approximately 775 MiB**. Before
downloading, Notebook 08 prints which files are present locally. Its next
cell fetches any missing public CSV, then applies common, previously tested
processing. Source data are stored under Git-ignored `data/raw/harth_v2/`.

## Processing and scientific integrity

- Treat each person's recording independently. Use source timestamps,
  50 Hz nominal sample rate, 100-sample windows, stride 50, full class purity
  and the existing no-timing-gap policy.
- Compute five deterministic statistical/temporal features for each of the
  six accelerometer channels, yielding 30 features per eligible window.
- Define an eight-class pilot task using officially documented labels 1–8.
- Exclude source codes outside the target task **only after** time-safe
  windowing. Audit excluded windows by source code and person.
- Preserve undocumented codes in source inspection and do not invent class
  definitions for them.
- Exclude `S023` and `S032` from development loading and model fitting.

## General reusable audit

`audit_group_class_coverage()` is defined in
`code/python/src/core/data_splitting.py`. It returns:

1. A subject-by-class count table, including explicit zero counts.
2. A GroupKFold feasibility report: group memberships, group-disjoint
   train/validation row counts and absent classes for each partition.

The output **does not** guarantee class-stratified or statistically
independent activity windows. A fold with missing training classes needs
a revised scientifically justified evaluation protocol before fitting.

## Important caveat

Earlier project stages have already inspected the two reserved Test
participants' source labels and summary statistics. They are **reserved
from model fitting and validation selection**, but should not be described
as completely unseen in a strict preregistration sense. All development
decisions must continue to exclude them. Notebook 08 does not load or
inspect these two subjects.

## Run

1. `git pull origin main`
2. `python -m pip install -r requirements.txt`
3. Open `notebooks/08_full_development_audit.ipynb` in Cursor.
4. Run through **Section 02** first; review the missing-file inventory.
5. Proceed to **Section 03** only when ready for up to about 775 MiB
   in additional downloads and the subsequent processing.
6. Inspect coverage tables, fold diagnostics, charts and the final quality
   gate. Commit the executed Notebook after confirming outputs.

No final accuracy or clinical/generalisation conclusion is supported by
this stage. The model-training and full-cohort GroupKFold comparison will
be separate, later steps.
