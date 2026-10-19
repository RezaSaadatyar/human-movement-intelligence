# Research study catalogue

Each study will have its own question, data availability check, unit of analysis, modelling protocol, evaluation and limitations. **Implementation status is study-specific. HARTH has completed a saved-evidence synthesis; other studies must not be described as completed without their own verified execution.** Do not confuse a proposal with completed research.

**Study diagrams:** [HARTH scientific workflow](wearable_activity_recognition/README.md) · [BoneNet exploratory workflow](biomechanical_load_estimation/README.md). Each diagram reports its own scientific scope and verified status; planned directions do not have completed-result diagrams.

## 1. Wearable Activity Recognition
- **Question:** How well can wearable motion signals classify activities for people not seen during training?
- **Candidates:** UCI HAR, PAMAP2, HARTH and MHEALTH.
- **Models to compare:** feature-engineered classical ML, 1D-CNN, BiLSTM and TCN.
- **Protocol:** subject-held-out evaluation; per-class recall/F1, balanced accuracy, confusion matrix and group-level variability.
- **Status:** Executed HARTH v2.0 research Stages 01–27, including seven-model Development benchmark, multi-seed robustness, frozen-model fitting, qualified same-study two-person held-out evaluation and read-only evidence synthesis. Two Test participants had been descriptively inspected earlier; no blind external validation is established. See [study README](wearable_activity_recognition/README.md).

## 2. Gait and Movement Analysis
- **Question:** Which reproducible temporal and biomechanical descriptors characterise walking and running under varying conditions?
- **Candidates:** publicly accessible gait kinematics, TRC/MOT/C3D archives.
- **Methods:** gait-event identification, joint angles, stride-normalised trajectories and comparative statistics.
- **Protocol:** check units, temporal alignment, movement-cycle definition and participant-level independence.
- **Status:** Planned.

## 3. Biomechanical Load Estimation
- **Question:** Can movement signals estimate selected *model-derived* joint kinetics or femoral-neck strain in unseen people or movement conditions?
- **Candidate:** 2026 Hollinger et al. femoral-neck loading dataset.
- **Models:** transparent regression baseline, tree-based regression, 1D-CNN/BiLSTM, optional data-efficient adaptation.
- **Protocol:** no derived-target leakage; subject-held-out results, physical units, MAE/RMSE, target-specific curve and peak errors.
- **Status:** Public BoneNet derivative inspected and exploratory baseline documented. This is not a confirmed full-cohort femoral-neck loading benchmark, and source-to-label linkage limitations remain. See [study README and figure](biomechanical_load_estimation/README.md).

## 4. Markerless Motion Capture
- **Question:** How reliable are video-derived body keypoints and joint-angle trajectories under varying acquisition conditions?
- **Potential methods:** OpenCV, MediaPipe Pose, selected 3D reconstruction tools and matched reference comparisons.
- **Protocol:** use licensed motion video; do not equate estimated 3D keypoints with laboratory ground truth.
- **Status:** Planned; appropriate video/reference dataset not selected.

## 5. Neurorehabilitation and Motor Function
- **Question:** How transferable are movement-related physiological or neuromuscular representations across participants and sessions?
- **Possible signals:** EEG/EMG and movement-related data if permission and aligned labels exist.
- **Methods:** temporal models, transfer learning, calibration-efficiency experiments, physiological feature analysis.
- **Protocol:** participant/session-aware splits; clearly separate existing published work from new experiments.
- **Status:** Planned.

## 6. Multimodal Movement Learning
- **Question:** When do synchronised sensor modalities improve movement representation or prediction relative to single modalities?
- **Inputs:** co-recorded, time-aligned IMU, video, kinematics and/or EMG where publicly available.
- **Methods:** feature-level and decision-level fusion, missing-modality experiments and domain shift analysis.
- **Protocol:** explicit temporal alignment, modality provenance and group independence.
- **Status:** Planned; aligned multi-modal dataset not selected.

## Study template

Every implemented study will eventually contain:
- A problem statement and testable hypotheses
- Data citations, access steps, schema and limitations
- Predictor/target definitions and grouping rules
- Baseline and proposed model descriptions
- Executable configuration, tests and run instructions
- Real result tables and failure analysis
- Clear scope of any scientific and clinical claims

See [research standards](../docs/RESEARCH_STANDARDS.md), [visual documentation rules](../docs/RESEARCH_VISUAL_STANDARDS.md) and [roadmap](../docs/ROADMAP.md).
