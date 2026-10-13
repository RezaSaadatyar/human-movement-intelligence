# MAT file acquisition and schema audit log

**Dataset:** Hollinger et al. Figshare v13, article ID 30225739.  
**User-provided local results:** 13 October 2026.  
**Status (13 October 2026):** Archive extracted; a bounded first-record nested schema probe has completed on two files. Full 48-record variable/target coverage is not yet audited.

| Audit step | Observed finding |
| --- | --- |
| Remote Figshare version | 13 |
| Extracted MATLAB files | 40 |
| ZIP entries | 42 (40 MAT files + 2 directories) |
| Total uncompressed ZIP size | 224.79 MiB |
| ZIP safety warnings | None |
| Five MAT header checks | MAT v5-compatible; `Data` and `__function_workspace__` |
| Python for user's inspection run | Python 3.14; NumPy/SciPy installed, h5py installed |
| First-record nested schemas | Two files inspected: subj04 and subj16, both `Data` shape `[48, 1]`, 13 named fields in their first elements |
| First-element fields | `SubjID`, `Task`, `SubTask`, `TrialNum`, `TrialNumUpdated`, `UpdatedTrialID`, `Cycle`, `IMU`, `ID`, `FP`, `e1e3`, `JCF`, `MuscleF` |
| Strain target coverage | Not yet checked |
| Subject-level partitioning | Not yet done |

**Next:** inspect the actual `outputs/essex_nested_mat_schema.json` generated for `subj04.mat` and `subj16.mat` (uploaded by the maintainer, not yet provided here). Extract the nested shape/dtype metadata for the first record of each. Then design an all-record audit that checks all 48 `Data` elements within each file, target presence and empty/NaN coverage, and unique trial identification without publishing raw signals.

The first failed archive-inspection attempt resulted from concatenated command-line text; the subsequent proper inspection ran successfully and did not identify a ZIP problem.

**Scientific boundary:** The field names are confirmed for the **first record only** of two files. Field existence does **not** establish populated `e1e3`, identical shapes in all 48 elements, exact trial semantics, or 48 independent observations. No train/test split or output prediction has been generated.
