# Essex data access: Figshare v13

**Source:** https://doi.org/10.6084/m9.figshare.30225739

## Verified locally on 13 October 2026
- Official file inventory: Figshare version 13, six files.
- Support files: `README.md` and `SubCharacteristics.xlsx`, downloaded and inspected.
- `MATLAB.zip`: downloaded and verified; 42 ZIP entries, including 40 MAT files, 224.79 MiB uncompressed, no archive-path warnings.
- Extracted MAT files: 40 under `data/raw/essex_femoral_neck_loading/MATLAB/Processed Data/`.
- Five top-level MAT headers: `subj04`, `subj05`, `subj06`, `subj08`, `subj09`. All report MAT v5-compatible files with `Data` and `__function_workspace__`.
- The bounded nested inspector has run on `subj04` and `subj16`. Each contains a `[48,1]` `Data` array; the **first record** in each has 13 fields: `SubjID`, `Task`, `SubTask`, `TrialNum`, `TrialNumUpdated`, `UpdatedTrialID`, `Cycle`, `IMU`, `ID`, `FP`, `e1e3`, `JCF`, `MuscleF`. This is *not* yet an audit of every element or target completeness.

See [verified v13 manifest](DATA_MANIFEST_ESSEX_V13.md), [metadata audit](../biomechanical_load_estimation/SUBJECT_METADATA_AUDIT.md) and [pipeline design](../biomechanical_load_estimation/DATA_PIPELINE_DESIGN.md).

## Next: inspect nested MATLAB structures

From the repository root on the maintainer's Windows system:

```powershell
git pull origin main
python scripts/inspect_mat_nested.py --root data/raw/essex_femoral_neck_loading --subjects subj04 subj16 --report outputs/essex_nested_mat_schema.json
```

The script loads **only** the named `Data` variable, not `__function_workspace__`. It records field names, array shapes and data types. Object arrays are sampled at their **first element**; this does **not** validate all trials. Numeric signal values are **not** included in the report.

If the first inspection terminates early because of node/depth limits, an explicitly deeper run may be appropriate after reviewing its initial structure:

```powershell
python scripts/inspect_mat_nested.py --subjects subj04 subj16 --max-depth 8 --max-nodes 800 --report outputs/essex_nested_mat_schema_deep.json
```

The console output has been received. Next, share `outputs/essex_nested_mat_schema.json` from the local machine so the first-element nested IMU, ID, JCF and e1e3 shape/dtype metadata can be examined. Only after this review should the all-record audit be finalised. These reports remain under ignored local `outputs/`. Do not upload raw MAT data, credentials or personal participant-level information to public GitHub.

## Data principles

- Do not assume any participant has an `e1/e3` field until checked.
- Cross-check actual subject identifiers with `SubCharacteristics.xlsx` before defining a cohort.
- 101-point movement trajectories and every-tenth-point FE outputs must not be treated as identical grids until verified.
- Baseline modelling must use subject-independent splits, training-only fitted transforms and an explicit target definition.
- Avoid downloading `Visual3D.zip` (approximately 18 GiB) for the first baseline.
