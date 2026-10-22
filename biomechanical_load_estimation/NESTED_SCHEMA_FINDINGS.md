# Uploaded nested MATLAB schema analysis (13 October 2026)

The user-uploaded `essex_nested_mat_schema.json` was read directly. This record is a **schema analysis**, not extraction of raw signals.

| Finding | subj04 first record | subj16 first record |
| --- | --- | --- |
| `Data` shape | `[48,1]` | `[48,1]` |
| First-record MATLAB struct field count | 13 | 13 |
| `SubjID, Task, SubTask, TrialNum, UpdatedTrialID` | Unicode fields | Unicode fields |
| `IMU, ID, FP, JCF, MuscleF` | Unresolved MATLAB object references | Unresolved MATLAB object references |
| `e1e3` | Unresolved MATLAB object reference | Empty array: `[0,0]`, `uint8` |
| Inspectable numeric IMU/target shape | Not yet known | Not yet known |

The `TrialNumUpdated` and `Cycle` fields also appear as opaque references. The `_TypeSystem`, `_Class` and `_ObjectMetadata` fields reflect how SciPy represents the MATLAB object; they do **not** give the original IMU channel names or target values.

**Caveat:** The existing `inspect_mat_nested.py` sampled only the first element of each `Data` array. Consequently the table says nothing about the other 47 records, actual target completeness, distinct physical trials or modelling sample size.

**Next:** run `scripts/audit_mat_records.py` to inspect all elements' source representations and task keys; then establish a MATLAB-native or otherwise verified conversion pathway before implementing tensor assembly. See [MATLAB decoding guide](../../docs/MATLAB_OBJECT_DECODING.md).
