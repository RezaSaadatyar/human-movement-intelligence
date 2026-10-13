# Local data layout

The repository stores **instructions and source references**, not raw research datasets.

## Recommended local layout

```text
data/
├── README.md                     # tracked
├── raw/                          # ignored by Git
│   ├── essex_femoral_neck_loading/
│   ├── uci_har/
│   ├── pamap2/
│   ├── mhealth/
│   ├── harth/
│   └── auckland_walk_run/
├── interim/                      # ignored
├── processed/                    # ignored
└── cache/                        # ignored
```

These subdirectories need to be created only when the first dataset is obtained.

For the Essex-linked study, open https://doi.org/10.6084/m9.figshare.30225739, inspect/download the required files and place them in `data/raw/essex_femoral_neck_loading/`. The final folder structure and MATLAB field names will be decided **after** examining the real files.

For UCI datasets, the documented `ucimlrepo` interface can fetch data directly during a connected Python run, so manual downloading may be unnecessary. For reproducibility, prefer a recorded dataset version and an ignored local cache. See [the full catalogue](../docs/DATASETS.md).

Do not commit `data/raw/`, `data/processed/`, archives, participant recordings, images, credentials or generated model artefacts. The MIT repository licence does **not** override a dataset's own rights.
