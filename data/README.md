# Research data and local working directories

The studies use their original research datasets and generate local numerical arrays or fitted models during execution. Use the relevant study notebooks for acquisition, processing and file locations.

| Study | Local directory | Purpose |
| --- | --- | --- |
| [HARTH v2.0](../wearable_activity_recognition/README.md) | `data/raw/harth_v2/` | Source lower-back and thigh accelerometer recordings |
| [Essex IMU–FE](../biomechanical_load_estimation/README.md) | `data/raw/essex_femoral_neck_loading/` | Original Essex Figshare records |
| [NinaPro DB2](../multimodal_movement_learning/README.md) | `data/raw/ninapro_db2/` | Per-subject sEMG and acceleration source archives, acquired and reused within an available research runtime |
| BoneNet reference material | `data/raw/bonenet_reference/` | Reference derivative files used by the relevant source-loading workflow |
| HARTH final models | `data/processed/harth_final_models/` | Fitted ResNet, Random Forest and inference settings |
| Essex paired data | `data/processed/essex_femoral_neck_loading/` | Locally reconstructed aligned IMU–FE arrays |
| NinaPro DB2 paired features | `data/processed/ninapro_db2/` | Source-length aligned E1 sensor arrays and 48/96-feature matrices produced from selected paired windows |

For dataset provenance, file inventories and download locations, see [Research Datasets and Access](../docs/DATASETS.md). For the notebook execution sequence and saved outputs, see the [HARTH study README](../wearable_activity_recognition/README.md#4-running-and-reviewing-the-study) and [Essex study README](../biomechanical_load_estimation/README.md#4-running-and-reviewing-the-study). The [NinaPro study README](../multimodal_movement_learning/README.md#4-running-and-reviewing-the-study) documents the third study's source and cohort execution. Respect the source providers' dataset licences and fixed versions.
