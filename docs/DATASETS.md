# Dataset catalogue and access guide

**Status:** Sources and landing pages are recorded below. Raw files have **not** yet been downloaded or inspected in this repository. Source-page descriptions are not a substitute for file-level schema verification.

## 1. Dataset strategy

Use two acquisition modes:
1. **Programmatic retrieval:** for public repositories with a stable documented library/API (e.g., UCI through `ucimlrepo`). An initial manual download is not always necessary when the runtime has network access. Cache a versioned local copy for repeatable research.
2. **Manual archive or verified API download:** for larger study-specific archives (e.g., Figshare). Follow the DOI landing page and retain its exact version, licence and file manifest. Do not invent permanent direct-file URLs.

Raw, derived and sensitive data do **not** belong in Git. Store them under `data/raw/<dataset_id>/` and `data/processed/<dataset_id>/`, or configure an external folder. See [data instructions](../data/README.md).

## 2. Primary biomechanical dataset — candidate for first research study

**Title:** Dataset for Rapid Estimation of Femoral Neck Loading During Gait and Dynamic Exercises (2026).  
**Data DOI:** https://doi.org/10.6084/m9.figshare.30225739  
**Landing page:** https://figshare.com/articles/dataset/Dataset_for_Rapid_Estimation_of_Femoral_Neck_Loading_During_Gait_and_Dynamic_Exercises/30225739  
**Publication:** Hollinger et al., *Scientific Data*, https://doi.org/10.1038/s41597-026-07429-3  
**Contributor:** Bernard X. W. Liew (among co-authors).  
**Rights:** the NLM dataset catalogue reports CC BY 4.0; confirm the licence and version on the specific Figshare record before any redistribution.  
**Reported modalities:** IMU movement signals; MSK-derived joint kinematics and kinetics; FE-derived femoral-neck tensile/compressive strains.  
**Population:** healthy-volunteer exercise dataset; it is not a pain diagnosis or patient-treatment dataset.  
**Tasks that might be viable:** regression for selected joint-level/modelled biomechanical targets, movement-cycle feature analysis and label-efficiency experiments.

**Acquisition procedure**
1. Open the Figshare landing page or DOI.
2. Examine the current version, licence, total archive size, downloadable files and any usage documentation.
3. Download only the needed archive/files to `data/raw/essex_femoral_neck_loading/`. If a working official Figshare API endpoint is confirmed later, an optional fetch script can replace manual downloading.
4. Record the data record version, archive filenames, checksums, citation, included subjects, variable names, target availability and permitted use.
5. Examine MATLAB/CSV structure before designing a predictive model. The published paper refers to processed subject-level MATLAB files and tabular data; **do not assume exact keys or file paths** until inspected.

**Status:** Public source and paper verified; direct Figshare file inspection and download method still to be validated. The network view of the Figshare landing page was restricted during research; this is not evidence that the data are private.

## 3. Other biomechanics dataset

**Walking and running biomechanics with different step widths and speeds**  
https://auckland.figshare.com/projects/Dataset_of_walking_and_running_biomechanics_with_different_step_widths_across_different_speeds/220732

The University of Auckland project page reports C3D marker/ground-reaction-force materials and TRC/MOT/CSV formats. Check individual item licences and study-level limitations. This is a **candidate** for gait and mechanics analysis, not a substitute for co-recorded video or IMU unless the exact item provides those modalities.

Suggested local path: `data/raw/auckland_walk_run/`.

## 4. Public activity-recognition datasets

| Dataset | UCI ID | Description and role | Access | Licence |
| --- | --- | --- | --- | --- |
| PAMAP2 | 231 | 9 participants, 18 listed activities, wearable IMU and heart-rate signals; sequence classification | https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring | CC BY 4.0 |
| MHEALTH | 319 | Activity recognition from wearable movement and physiological sensors | https://archive.ics.uci.edu/dataset/319/mhealth+dataset | CC BY 4.0 |
| HARTH | 779 | 22 participants, thigh/back accelerometers in free-living activity recordings | https://archive.ics.uci.edu/dataset/779/harth | CC BY 4.0 |
| UCI HAR Smartphones | 240 | 30 participants, six activities, inertial features/windows | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones | CC BY 4.0 |

Dataset pages give licences and full citations. These are useful benchmarks for **activity classification**. They do not contain ground truth femoral-neck strain or clinical pain labels.

### Programmatic UCI access (example only — not yet executed)

```python
# Install separately: pip install ucimlrepo
from ucimlrepo import fetch_ucirepo

# UCI HAR Smartphones: public structured benchmark
dataset = fetch_ucirepo(id=240)
X = dataset.data.features
y = dataset.data.targets
print(dataset.metadata)
print(X.shape, y.shape)
```

This makes an external network request when executed; it does not automatically generate the project's audited study schema, reproduce the source's train/test protocol or guarantee stable immutable versioning. Add caching, subject ID validation and split governance before modelling.

**Large archives:** PAMAP2 is reported by UCI as about 656 MB and HARTH about 296 MB; prefer local caching rather than repeatedly fetching these during each run.

## 5. Rules before a dataset enters a study

- Confirm the source, persistent citation, version, licence and permitted reuse.
- Record whether the data consist of raw sensor streams, preprocessed windows, derived features or simulated/modelled quantities.
- Check units, sampling rates, timestamps, grouping IDs, expected file shapes and target definitions.
- Validate the relationship between predictors and targets to avoid target leakage.
- Partition by subject (or stronger independent unit) where the research question concerns generalisation to new people.
- Do not merge unrelated EEG, EMG, IMU and video datasets and imply subject-level multimodal alignment.
- Do not upload copyrighted or restricted raw datasets merely because they are publicly accessible.
- Describe sample limitations, participant selection, activity types and clinical applicability.

## 6. Data-inventory record template

For each downloaded dataset, record:
```text
dataset_id:
source_url:
source_doi:
version_or_release_date:
licence_and_attribution:
downloaded_at:
download_method:
local_root:
file_inventory_and_checksums:
unit_of_analysis:
participant_and_session_identifiers:
sensor_modalities_and_units:
available_targets_and_units:
exclusions_or_known_limitations:
privacy_or_access_restrictions:
```

**Next step:** inspect the real Figshare archive and decide which target can be predicted without using information derived from that target.

## 7. October 2026 source inspection update

The **primary publication's Table 3** and its subject exclusions have been checked; the authors' **secondary public GitHub** `labels.csv` was directly inspected (444 rows, 37 participants; 333 Running, 111 Walking). Its `SubjID + Task + TrialNum` combinations are not unique (222 unique combinations). This does **not** establish the nested structure of the main Figshare source files.

The Figshare host could not be reached from the execution environment, so its **live file inventory, latest version and binary data schemas are still pending**. The NLM catalogue refers to DOI version `v13`, which must not be assumed to remain the latest.

See [data-access commands](DATA_ACCESS_ESSEX.md), [the proposed pipeline](../studies/biomechanical_load_estimation/DATA_PIPELINE_DESIGN.md), and the read-only `scripts/figshare_inventory.py` metadata tool.
