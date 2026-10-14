# BoneNet actual-data validation (14 October 2026)

**Status:** Successfully executed on a temporary GitHub Actions runner using the authors' real public `Data.mat` and `labels.csv` (Python only). This is a derivative walking/running cohort, not the full 40-participant six-exercise Figshare source.

- Source data audit: https://github.com/RezaSaadatyar/human-movement-intelligence/actions/runs/38558537259
- Offline test suite: https://github.com/RezaSaadatyar/human-movement-intelligence/actions/runs/38558537147

## Actual source metadata

| Item | Verified |
| --- | --- |
| Source MATLAB root variables | `inputs`, `outputs` |
| Input `X` | 444 records × 42 channels × 101 time points |
| Output `Y` | 444 records × 101 time points × 2 targets |
| Targets | `e1`, `e3` |
| Non-finite source input entries | 0 |
| Non-finite source target entries | 0 |
| Label rows | 444 |
| Subject IDs in CSV | 37 |
| Task counts | Running 333, Walking 111 |
| Duplicated `SubjID+Task+TrialNum` combinations | 222 |
| Independently verified sample-to-label association | **No** |

The numerical tensors were successfully recovered without MATLAB. The source's repeated composite keys mean that the row-order assumption still requires source provenance/validation. The successful numeric audit therefore **does not itself prove subject-independent generalisation validity**.

The Python test suite passed **14/14 offline tests** after correcting a synthetic MATLAB struct fixture in `tests/test_audit_mat_records.py`. The previously failed CI run was caused by that fixture, not the real source data.

## Exploratory modelling stage

A separate workflow, `.github/workflows/bonenet-baseline.yml`, will run a transparent training-mean and per-channel statistical-feature Ridge baseline using a subject-held-out split. It deliberately declares the author's sample-to-label row-order assumption, and its results must remain labelled **exploratory replication, not validated scientific evidence** until that mapping is independently checked. Only metadata and metrics are uploaded, not raw data or model weights.

Do not interpret finite-element strain targets as direct in-vivo measurements, and do not claim clinical validation.
