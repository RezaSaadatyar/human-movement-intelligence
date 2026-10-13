# Verified Essex Figshare v13 inventory

**Verified by maintainer:** 13 October 2026, using `scripts/figshare_inventory.py` on Windows.  
**Dataset:** https://doi.org/10.6084/m9.figshare.30225739  
**Article ID:** 30225739 | **Version:** 13 | **Files:** 6

| File | Size (MiB) | Figshare file ID | Recommended action |
| --- | ---: | ---: | --- |
| README.md | 0.01 | 63343417 | Download and inspect first |
| SubCharacteristics.xlsx | 0.03 | 63342400 | Download and inspect first |
| MATLAB.zip | 224.68 | 63190690 | Core data, obtain after README |
| angles.xlsx | 29.73 | 63342406 | Joint-angle analysis, later |
| OpenSim.zip | 10.40 | 63190519 | Optional MSK modelling |
| Visual3D.zip | 18414.04 | 63341380 | Defer, approximately 18 GiB |

The remote filename, version, size and file IDs were verified from the user's actual command output. Contents, licence at the file level, metadata columns, actual MATLAB array shapes and archive subdirectories **have not yet been inspected**.

## Next acquisition step

From the checked-out repository root on Windows:

```powershell
git pull origin main
python scripts/figshare_inventory.py --download-file "README.md" --destination data/raw/essex_femoral_neck_loading
python scripts/figshare_inventory.py --download-file "SubCharacteristics.xlsx" --destination data/raw/essex_femoral_neck_loading
```

The tool checks the official byte count and optional hash and will refuse to overwrite existing files. Keep downloaded data under ignored local `data/raw/`. Check README and spreadsheet contents before downloading the 225 MiB MATLAB archive. Avoid Visual3D.zip for the first baseline.
