# Research data and local working directories

The studies use their original research datasets and generate local numerical arrays or fitted models during execution. Use the relevant study notebooks for acquisition, processing and file locations.

| Study | Local directory | Purpose |
| --- | --- | --- |
| [HARTH v2.0](../wearable_activity_recognition/README.md) | `data/raw/harth_v2/` | Source lower-back and thigh accelerometer recordings |
| [Essex IMU–FE](../biomechanical_load_estimation/README.md) | `data/raw/essex_femoral_neck_loading/` | Original Essex Figshare records |
| BoneNet reference material | `data/raw/bonenet_reference/` | Reference derivative files used by the relevant source-loading workflow |
| HARTH final models | `data/processed/harth_final_models/` | Fitted ResNet, Random Forest and inference settings |
| Essex paired data | `data/processed/essex_femoral_neck_loading/` | Locally reconstructed aligned IMU–FE arrays |

For details, see the [HARTH execution guide](../docs/HARTH_REPRODUCIBILITY.md) and [Essex execution guide](../docs/BONENET_REPRODUCIBILITY.md). Respect the original dataset licences and use the stated source versions when reproducing an analysis.
