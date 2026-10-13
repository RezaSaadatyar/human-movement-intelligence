# Essex data access: verified v13

**Primary source:** https://doi.org/10.6084/m9.figshare.30225739  
**Live inventory verified:** 13 October 2026 from the maintainer's Windows terminal. Version 13; six files. See [file inventory](DATA_MANIFEST_ESSEX_V13.md).

## Step 1: Small supporting files

In PowerShell from the repository root:

```powershell
git pull origin main
python scripts/figshare_inventory.py --download-file "README.md" --destination data/raw/essex_femoral_neck_loading
python scripts/figshare_inventory.py --download-file "SubCharacteristics.xlsx" --destination data/raw/essex_femoral_neck_loading
```

They are about 0.01 and 0.03 MiB. Send the README text and metadata column names (without personal data), or share the two downloaded files privately for inspection.

## Step 2: MATLAB.zip after documentation review

```powershell
python scripts/figshare_inventory.py --download-file "MATLAB.zip" --destination data/raw/essex_femoral_neck_loading --max-mb 512
```

This file is about 224.68 MiB. Inspect ZIP member filenames and uncompressed size **before** extraction. Extract to a local folder under `data/raw/essex_femoral_neck_loading/`. Never commit raw data.

## Step 3: MATLAB header inspection

Once `.mat` files are extracted:

```powershell
python -m pip install -r requirements-data.txt
python scripts/inspect_mat.py --root data/raw/essex_femoral_neck_loading --report outputs/essex_mat_headers.json
```

This reports only top-level variable names and shapes. A dedicated loader comes after nested arrays, units and trial keys are confirmed.

## Not needed yet

- `angles.xlsx` (29.73 MiB): later kinematic analysis.
- `OpenSim.zip` (10.40 MiB): optional model files.
- `Visual3D.zip` (18,414.04 MiB): not needed for first baseline.

## Research integrity

The authors' separate GitHub `labels.csv` is a derivative, not the primary Figshare trial manifest. Its subject/activity/trial combinations are not unique and must not be joined naively. Subject-independent splitting, train-only preprocessing, explicit units and provenance are required. No model results are available yet.
