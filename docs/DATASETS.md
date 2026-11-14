# Research Datasets and Access

The active studies use distinct original datasets, model inputs, targets and participant-level evaluation protocols. Follow each study's scientific documentation before downloading data or reproducing an analysis.

## 1. Datasets Used in the Research

| Study | Original source | Inputs and targets | Protocol and access |
| --- | --- | --- | --- |
| [Wearable Activity Recognition](../wearable_activity_recognition/README.md) | [NTNU AI Lab HARTH v2.0](https://github.com/ntnu-ai-lab/harth-ml-experiments) | Six lower-back/thigh acceleration channels; eight original activity codes | Original source revision and windowing rules are given in the [HARTH methods](../wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) |
| [Biomechanical Load Estimation](../biomechanical_load_estimation/README.md) | [Essex Figshare v13](https://doi.org/10.6084/m9.figshare.30225739.v13), described by [Hollinger et al.](https://doi.org/10.1038/s41597-026-07429-3) | Seven lower-body IMUs and model-derived femoral-neck principal-strain targets | Source files and participant/trial exclusions are documented in the [Essex methods](../biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md) |

The active HARTH study uses the **NTNU AI Lab v2.0 source**, not the [UCI HARTH release](https://archive.ics.uci.edu/dataset/779/harth). The Essex study uses **model-derived FE targets**, not direct in-vivo strain measurements or clinical outcomes.

## 2. Download and Local Storage

Use the data acquisition sections of the research notebooks. The expected local directories and their purposes are summarised in the [data README](../data/README.md).

For Essex, the [source inventory](DATA_MANIFEST_ESSEX_V13.md) identifies the files used by the v13 workflow and the [data-access guide](DATA_ACCESS_ESSEX.md) explains download and extraction. For HARTH, the pinned source revision and window eligibility rules are described in the [scientific methods](../wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md) and [execution guide](HARTH_REPRODUCIBILITY.md).

Keep the data version, sensor definitions, participant identities, original target units and time alignment consistent with the study protocol. Data access and reuse are governed by the source providers' terms.

## 3. Additional Public Collections

These collections may be relevant to future independent comparisons, but **they are not interchangeable with the current HARTH or Essex datasets**:

| Dataset | Source | General content |
| --- | --- | --- |
| PAMAP2 | [UCI dataset 231](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) | Activity monitoring using wearable signals |
| MHEALTH | [UCI dataset 319](https://archive.ics.uci.edu/dataset/319/mhealth+dataset) | Wearable movement and physiological signals |
| UCI HAR Smartphones | [UCI dataset 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | Smartphone inertial activity recognition |
| Auckland walking/running biomechanics | [Auckland Figshare project](https://auckland.figshare.com/projects/Dataset_of_walking_and_running_biomechanics_with_different_step_widths_across_different_speeds/220732) | Gait and running biomechanics at varied speeds and step widths |

Any future comparison would require its own source/label mapping, participant-disjoint design and evaluation plan. Separate sources must not be described as subject-linked multimodal observations without actual record linkage.

## 4. Data Integrity and Reuse

For a scientific reproduction, verify the dataset citation and version, licence, source units and channel meanings, subject/trial identifiers, sampling frequency, timestamp continuity, exclusion rules and target provenance. Establish participant separation before fitting transformations or selecting models, and document discrepancies rather than silently recoding labels.

Public availability of a dataset does not automatically grant unrestricted redistribution or establish clinical representativeness.
