# Essex Figshare v13: Completed Python representation audit

**Data source:** Hollinger et al., Figshare article 30225739, version 13.  
**Evidence:** Maintainer's local `python scripts/audit_mat_records.py --all` console log, 13 October 2026. Detailed JSON reports were generated locally but are not attached to this record.

| Measure | Observed result |
| --- | ---: |
| Participant files | 40 |
| `Data` elements inspected | 1,897 |
| Explicitly empty `e1e3` | 168 |
| Nonempty MATLAB-object `e1e3` | 1,729 |
| Participants with 48 elements | 35 |
| Participants with fewer than 48 elements | 5 |

The 168 empty entries comprise **144** in the three fully FE-excluded participants (`subj16`, `subj29`, `subj32` with 48 each), plus **24** partial gaps (`subj06` 9; `subj08` 9; `subj11` 6).

Files with fewer than 48 elements: `subj09` (42), `subj11` (40), `subj12` (47), `subj14` (42), `subj36` (46). The remaining 35 files contain 48 each.

**Scientific limitation:** The 1,729 nonempty MATLAB object representations are **not yet verified finite-valued strain waveforms**. The console reports zero duplicate `UpdatedTrialID` values within files, but the readable-ID counts and cross-file uniqueness have not yet been checked in the JSON. This is a metadata audit, not an ML cohort or completed regression benchmark.

## Next: MATLAB-native representation check

Some fields appear as `_TypeSystem`, `_Class` and `_ObjectMetadata` when read using SciPy. To determine whether they are actual MATLAB tables, objects or other types, the next bounded audit uses `code/matlab/inspect_essex_matlab_native.m`.

If MATLAB is installed, open it in `D:\human-movement-intelligence`, then run:

```matlab
addpath(fullfile(pwd,'code','matlab'));
inspect_essex_matlab_native
```

It inspects two files by default and writes only class names, variable names, array dimensions and counts to `outputs/essex_matlab_native_schema.json`. This script has **not** been run by the assistant on source files.

If MATLAB is unavailable, report that before trying alternative object-decoding approaches. Do not coerce MATLAB metadata references into numeric tensors.
