# Essex subject metadata audit (Figshare v13)

**Inspection date:** 13 October 2026  
**Local source files:** `README.md` and `SubCharacteristics.xlsx` downloaded and integrity-checked from Figshare record https://doi.org/10.6084/m9.figshare.30225739. The maintainer provided these two files for direct content inspection.  
**Status:** Participant metadata inspected. The MATLAB movement/strain archive has **not** been downloaded or inspected, and no model has run.

## 1. Workbook and participant cohort

The `SubCharacteristics.xlsx` workbook contains a single worksheet (`Sheet1`). The participant table occupies rows 2–41, with row 1 containing headers. Rows 42 and 43 contain aggregate mean and SD summaries, rows 46–50 are age-group headings, and rows 52–53 provide sex-count notes. **Do not import these summary rows as participant observations.**

The 40 participant records have 40 distinct non-contiguous `SubjXX` IDs (from Subj04 through Subj44, with gaps). Do **not** generate identifiers by numeric range, and do **not** assume 40 MATLAB files correspond to IDs 01–40.

| Characteristic | Aggregated workbook finding |
| --- | --- |
| Participant rows / distinct IDs | 40 / 40 |
| Recorded sex categories | 19 M / 21 F |
| Age, range | 19–69 years |
| Age, arithmetic mean | 40.30 years |
| Body mass, arithmetic mean | 68.4445 kg |
| Height, arithmetic mean | 1.709725 m |
| BMI, arithmetic mean | 23.2701 kg/m² |
| Limb dominance | 36 R / 4 L |
| Explicit health/pain-history notes | 5 |
| Placeholder value `…` in health/pain column | 35 |

Only aggregate, non-identifying findings are recorded here. The raw participant-level spreadsheet remains ignored under `data/raw/`.

## 2. Exact source headers and interpretation

| Exact workbook header | Proposed canonical column | Type / interpretation |
| --- | --- | --- |
| `Participent ID` | `subject_id` | Case-normalised string, primary group ID |
| `Gender` | `gender_recorded` | Source-coded categorical M/F; do not infer identity |
| `Age (year)` | `age_years` | Integer years |
| `Mass (kg)` | `mass_kg` | Numeric kilograms |
| `Height (m)` | `height_m` | Numeric metres |
| `Dominant limb` | `dominant_limb` | Source-coded R/L |
| `BMI` | `bmi_kg_m2` | Numeric; agrees approximately with mass/height² |
| `Any disease/ pain` | `health_history_note` | Free text or `…` placeholder; **not a clean diagnosis/healthy label** |
| `Date of data collection` | `collection_date` | Excel serial date; convert with a documented workbook epoch |
| `Sport ` | `sport_note` | Free text; not a validated activity outcome label |

Two additional column headers contain blanks/spaces and have no participant content. Exclude them.

All 40 participant rows contain nonempty cells in the ten source columns. **Nonempty does not mean clinically complete**: 35 records contain `…` under health/pain, which should be treated as an unspecified/placeholder value and audited separately, not converted automatically to `healthy` or `no disease`.

## 3. Cross-source inconsistencies requiring provenance notes

1. **Sex distribution mismatch:** the downloaded README describes 20 males and 20 females; the actual workbook and its embedded totals indicate 19 males and 21 females. Do not silently overwrite either source or claim the discrepancy is resolved.
2. **Participant range:** source IDs have gaps and include `Subj44`, so a generated `subj01`–`subj40` participant list would be incorrect. Source README folder examples are illustrative.
3. **Spreadsheet filenames:** the downloaded README refers to `SubjCharacteristics.csv` and `angles.csv`; Figshare v13 actually lists `SubCharacteristics.xlsx` and `angles.xlsx`. The format difference is recorded, not corrected in the source.
4. **Health status:** the README's cohort-level healthy/no-recent-lower-limb-injury description should not be interpreted as a per-participant negative diagnosis label; the workbook contains free-text historical or symptom-related notes.
5. **IMU units:** README displays `deg/s², deg/s` for accelerometer and gyroscope; do **not** standardise or convert accelerations from this description alone. Inspect the actual MATLAB field metadata and units.
6. **Temporal grids:** most time-normalised signals are described as 101 points, while FE strain is sampled at every tenth point. Determine the actual available time axes before defining sequence targets.

## 4. Implications for the planned data pipeline

- Load only the participant table, not the mean/SD or age-group summary rows.
- Preserve the original source headers in provenance, but expose validated canonical names for code.
- Normalise IDs consistently; **never infer missing IDs are records**.
- Treat the pain/history field as a sensitive free-text note. Exclude it from baseline prediction features; do not publish it in experiment artefacts or GitHub.
- Store only a minimal subject-key lookup when joining metadata to MATLAB trial records.
- Verify which participants have usable FE outputs; a source paper notes three FE exclusions, but file-level target coverage remains unknown.
- Check whether demographics are needed for a predeclared scientific analysis; do not add them to an IMU-only prediction without defining a different input contract.
- Split by `subject_id`; trial count or normalised cycle points cannot substitute for independent participants.

## 5. Next validation gate

Download the `MATLAB.zip` archive (~224.68 MiB), inspect ZIP member paths and total uncompressed size **before extraction**, then inspect a few real MAT file headers. Only then implement the dataset loader and define `e1/e3` or joint moment targets.

**Scope:** This audit establishes spreadsheet-level descriptive facts, not full dataset quality, clinical validity, or benchmark performance.
