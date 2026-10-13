# MATLAB-native inspection instructions (Essex v13)

The Python audit has completed over 40 subject MAT files (1,897 `Data` elements). Most nonempty biomechanical fields are represented as MATLAB opaque object metadata by SciPy. We must identify their native MATLAB classes and actual numeric variable shapes before designing the Data Loader.

## Step 1 — Inspect two representative subject files

From Windows PowerShell:

```powershell
git pull origin main
```

If MATLAB is installed, launch MATLAB, select the repository root `D:\human-movement-intelligence` as **Current Folder**, and execute in its Command Window:

```matlab
addpath(fullfile(pwd,'code','matlab'));
inspect_essex_matlab_native
```

Alternatively, if the MATLAB executable is on PATH:

```powershell
matlab -batch "addpath('code/matlab'); inspect_essex_matlab_native"
```

The default subjects are `subj04` (nonempty e1e3 representations) and `subj16` (empty e1e3). The script reads the original `Data` variable, inspects all records within the two files and summarises MATLAB classes, counts, variable names and dimensions for `IMU`, `ID`, `FP`, `JCF`, `MuscleF`, `e1e3`, `Cycle`, and `TrialNumUpdated`.

It writes a **metadata-only** report to:

`outputs/essex_matlab_native_schema.json`

Do not upload raw participant MAT files into the public repository. Share the JSON report privately for review; it is generated under ignored `outputs/`.

## Step 2 — Only if the first inspection succeeds

A subsequent run can inspect all 40 files:

```matlab
inspect_essex_matlab_native('all')
```

Do not execute this all-files step yet. The two-file schema should first be reviewed for unexpected variable types or errors.

## Boundaries

- A nonempty `e1e3` object is not necessarily a valid numeric strain target.
- `Data` element count is not automatically the number of distinct biological trials.
- Variable names and sample shapes must be reviewed before joining inputs with targets.
- No model training, clinical inference or leakage-prone split is performed by this inspector.

**MATLAB availability is a prerequisite for these commands.** If MATLAB is not installed or licensed locally, stop and choose a separately validated alternative.
