# Essex 2026 data access: practical guide

## Source of truth and current verification

- Article: https://doi.org/10.1038/s41597-026-07429-3
- Dataset: https://doi.org/10.6084/m9.figshare.30225739
- Dataset record ID: `30225739`
- Figshare API (public): https://api.figshare.com/v2/articles/30225739
- Authors' separate code and derivative example: https://github.com/ZainabAltai/BoneNet-A-Framework-for-Rapid-and-Accurate-Prediction-of-Femoral-Neck-strains-from-Exercise-Data

The paper establishes the high-level dataset fields and exclusions. The Figshare file list **could not be retrieved from the assistant's execution environment** because the Figshare host was unreachable there. The public API and its schema are documented by Figshare. **Do not interpret this network restriction as access requiring a paid or private account.**

The NLM catalogue references dataset DOI version `v13`; the current live version must be confirmed by querying metadata, not assumed to remain `v13`.

## Step 1 — List files without downloading data

On Windows PowerShell, after cloning this repository:

```powershell
cd human-movement-intelligence
python scripts/figshare_inventory.py --manifest outputs/figshare_30225739_inventory.json
```

The script prints exact remote file names, sizes and IDs and saves a JSON report under the ignored `outputs/` directory. **No dataset file is downloaded by this command.** If the API fails on the user's network, open the Figshare DOI in the browser to inspect the files and record version, or download manually.

## Step 2 — Choose minimal files

Review the listing and prefer the smallest source bundle containing `SubCharacteristics.xlsx`, `angles.csv`, one or more `subjXX.mat` files, and any official README. This is a *selection criterion*, not a claim about the actual current archive layout. Record the version and licences before choosing.

Do **not** start with the entire C3D/OpenSim raw archive if smaller processed files suffice for an initial feasibility check.

## Step 3 — Download exactly one named file

```powershell
python scripts/figshare_inventory.py --download-file "EXACT_FILENAME_FROM_LIST" --destination data/raw/essex_femoral_neck_loading --max-mb 1024
```

The script streams a selected file, checks the Figshare-declared byte count and optional MD5, refuses to overwrite existing files, and enforces a size cap. Increase `--max-mb` only after checking storage capacity. No URL credentials or API tokens are needed for public data.

Alternatively use the Figshare browser **Download** control and place the resulting file in:

```text
human-movement-intelligence/data/raw/essex_femoral_neck_loading/
```

On a cloud notebook, use an approved mounted storage folder or a runtime-only cache. Never commit raw data to Git.

## Step 4 — Validate the local archive

If the download is a ZIP, review archive entries for unexpected paths before unpacking. After MATLAB files exist:

```powershell
python -m pip install -r requirements-data.txt
python scripts/inspect_mat.py --root data/raw/essex_femoral_neck_loading --max-files 5 --report outputs/essex_mat_headers.json
```

The header tool reports top-level variables and shapes without claiming to decode nested MATLAB trial fields. Next, an actual dataset-specific loader must be built to validate the nested values and construct trial keys.

## Step 5 — Share only safe inspection outputs

The file lists, schema summaries and error messages (without personal credentials or participant-level private data) are appropriate to share for the next coding step. Do not upload clinical or restricted datasets into a public GitHub repository.

**Current stage:** Figshare metadata inventory pending on a network-connected machine; authors' labels.csv inspected; dataset pipeline design recorded. No model has been trained.
