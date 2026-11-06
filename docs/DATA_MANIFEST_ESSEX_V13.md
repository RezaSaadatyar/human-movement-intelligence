# Essex Figshare v13 source inventory

**Dataset:** https://doi.org/10.6084/m9.figshare.30225739.v13

**Article ID:** 30225739 | **Version:** 13

The following inventory was recorded from the official version-13 file list. These are descriptive source details, not hard-coded execution checks.

| File | Size (MiB) | Figshare file ID | Recommended action |
| --- | ---: | ---: | --- |
| README.md | 0.01 | 63343417 | Download and inspect first |
| SubCharacteristics.xlsx | 0.03 | 63342400 | Download and inspect first |
| MATLAB.zip | 224.68 | 63190690 | Core data, obtain after README |
| angles.xlsx | 29.73 | 63342406 | Joint-angle analysis, later |
| OpenSim.zip | 10.40 | 63190519 | Optional MSK modelling |
| Visual3D.zip | 18414.04 | 63341380 | Defer, approximately 18 GiB |

Notebook 01 reads the current source manifest and obtains the required support files and `MATLAB.zip`. Existing downloads are reused under `data/raw/essex_femoral_neck_loading/`; extracted participant files are written to `matlab_records/` below that folder.

The original run observed 42 ZIP entries, including 40 participant MAT files under `MATLAB/Processed Data/`, and 224.79 MiB uncompressed. Source structure, numerical decoding and FE time alignment are handled in Notebooks 01–03. The current manifest table is [`01_dataset_figshare_file_manifest.csv`](../biomechanical_load_estimation/output/01_dataset_figshare_file_manifest.csv).

See [data access](DATA_ACCESS_ESSEX.md) and [notebook execution](BONENET_REPRODUCIBILITY.md) for the current workflow.
