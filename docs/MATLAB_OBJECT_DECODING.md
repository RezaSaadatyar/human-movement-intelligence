# MATLAB object decoding and full-record quality audit

**Based on:** uploaded `essex_nested_mat_schema.json` inspected on 13 October 2026.  
**Scope:** first records of `subj04.mat` and `subj16.mat`, each `Data` array of shape `[48,1]`. The first record has the same 13 named fields.

## Confirmed from the actual JSON

- Scalar-like textual identifiers: `SubjID`, `Task`, `SubTask`, `TrialNum`, `UpdatedTrialID`.
- `IMU`, `ID`, `FP`, `JCF`, `MuscleF`, `Cycle`, and `TrialNumUpdated` are represented in the SciPy report as structured arrays with `_TypeSystem`, `_Class`, and `_ObjectMetadata`. These are MATLAB opaque/object references, **not decoded sensor arrays**.
- In the first record of `subj04.mat`, `e1e3` is likewise an unresolved MATLAB object reference. This does not establish a valid strain waveform.
- In the first record of `subj16.mat`, `e1e3` is an actual empty `uint8` array of shape `[0,0]`. This agrees with the publication's note that the participant is excluded from FE strain calculations, but coverage over all records is unverified.
- None of the 48 records in either file has yet been reviewed comprehensively for target availability or unique observation identification.

## Why SciPy's initial result is insufficient

MAT v5-compatible files can include MATLAB class instances that `scipy.io.loadmat` exposes as opaque metadata references rather than their numerical properties. The six `uint32` values under `_ObjectMetadata` are **not sensor samples**. The actual contents may depend on MATLAB's internal object subsystem. Do not coerce those values to float tensors.

See SciPy documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html and related object-decoding discussion: https://github.com/scipy/scipy/issues/22736.

## Step 1 — Audit every record's representation (Python-only)

After pulling the new script, run from the repository root:

```powershell
python scripts/audit_mat_records.py --subjects subj04 subj16 --report outputs/essex_two_subject_record_audit.json
```

It will summarise `e1e3` as **empty vs opaque reference** per record, count activity and sub-activity labels, check reported IDs and detect duplicated `UpdatedTrialID` strings. It will **not** infer that an opaque object is a populated target. Send the JSON report for review.

After this succeeds, run an all-subject audit:

```powershell
python scripts/audit_mat_records.py --all --report outputs/essex_all_record_audit.json
```

This processes all 40 files locally and may take longer. No raw samples are exported; leave both reports in the ignored `outputs/` folder.

## Step 2 — Recover the actual sensor and target arrays

If MATLAB is available on the maintainer's computer, a controlled MATLAB-native export to ordinary numeric arrays and explicit field names is likely the most reliable route. First inspect `class(Data(1).IMU)`, `class(Data(1).e1e3)`, their variable names if tables, and `isempty(Data(1).e1e3)` in MATLAB. Do **not** modify source `.mat` files. A separate export script will be designed only after confirming the MATLAB classes and the exact table schema.

If MATLAB is not available, evaluate alternative compatible tools or a reduced derivative dataset separately and label the provenance. Do **not** assume SciPy, h5py, mat73, mat4py or GNU Octave can decode arbitrary MCOS objects in this particular MAT v5 file.

## Data contract gate

No dataset loader, normalisation, padding, temporal interpolation or supervised model training should be considered scientifically ready until `IMU`, `ID` and `e1e3` have resolved numeric shapes, trial-to-target mappings and units. Group splits must be subject-independent; `UpdatedTrialID` uniqueness is not yet established.
