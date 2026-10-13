# MAT file acquisition and schema audit log

**Dataset:** Hollinger et al. Figshare v13, article ID 30225739.  
**User-provided local results:** 13 October 2026.  
**Status:** Source archive safely extracted; nested variables not yet audited.

| Audit step | Observed finding |
| --- | --- |
| Remote Figshare version | 13 |
| Extracted MATLAB files | 40 |
| ZIP entries | 42 (40 MAT files + 2 directories) |
| Total uncompressed ZIP size | 224.79 MiB |
| ZIP safety warnings | None |
| Five MAT header checks | MAT v5-compatible; `Data` and `__function_workspace__` |
| Python for user's inspection run | Python 3.14; NumPy/SciPy installed, h5py installed |
| Nested structures | Not yet checked |
| Strain target coverage | Not yet checked |
| Subject-level partitioning | Not yet done |

**Next:** examine two representative subjects, `subj04.mat` and `subj16.mat`, through the bounded `inspect_mat_nested.py` utility. These are selected for schema comparison, not used as the actual modelling cohort. The tool must not export signal values.

The first failed archive-inspection attempt resulted from concatenated command-line text; the subsequent proper inspection ran successfully and did not identify a ZIP problem.

**Scientific boundary:** the name `Data` alone does not establish the presence or shape of IMU, inverse dynamics, joint-contact-force or femoral-neck strain variables. No train/test split or output prediction has been generated.
