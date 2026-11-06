# Essex data access

**Source:** [Essex Figshare v13](https://doi.org/10.6084/m9.figshare.30225739.v13).

Start with Notebook 01 in `biomechanical_load_estimation/code/notebooks/`. It uses the current file manifest, the source README, participant workbook and MATLAB archive. No separate inspection scripts or JSON reports are required.

- Raw downloads: `data/raw/essex_femoral_neck_loading/`.
- Extracted participant files: `data/raw/essex_femoral_neck_loading/matlab_records/`.
- Public summaries and current tables: `biomechanical_load_estimation/output/`.
- Paired numerical arrays: `data/processed/essex_femoral_neck_loading/essex_imu_fe_pairs_v13.npz`.

Existing raw files are reused. If `MATLAB.zip` is absent, the downloader tries the project data release and official Figshare download endpoints for the same v13 archive. A new transfer must be complete before replacing its temporary file. ZIP path and CRC checks remain to prevent unsafe extraction or use of a damaged archive.

Notebooks 02–03 inspect the original nested tables, source labels and FE time positions. Notebook 04 builds pairs and assigns participants. FE targets are not assumed to occur at every tenth IMU position; the source time coordinates determine the actual matching indices. Training-only transforms and participant-disjoint evaluation are retained.

The original archive and full numerical arrays are not committed into the code tree. The much larger Visual3D archive is unnecessary for the notebook sequence.

See the [source inventory](DATA_MANIFEST_ESSEX_V13.md), [execution guide](BONENET_REPRODUCIBILITY.md) and [research report](../biomechanical_load_estimation/BONENET_RESEARCH_METHODS_AND_EVIDENCE.md).
