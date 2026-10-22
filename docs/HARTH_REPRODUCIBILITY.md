# HARTH v2.0 | Reproducibility and Release Readiness

## Scope and verified evidence

This is a practical pre-release audit for the completed HARTH activity-recognition study in `human-movement-intelligence`. **It does not add a training experiment, alter frozen model code, rerun original Test, or create a GitHub Release.** Executed Notebooks 23–27 contain seven-model Development comparisons, 36 multi-seed jobs, pre-Test model selection, one approved qualified Test analysis and a read-only evidence synthesis. The earlier 01–22 notebooks remain historical evidence with different objectives and subsets.

- Source: [NTNU AI Lab HARTH v2.0](https://github.com/ntnu-ai-lab/harth-ml-experiments), pinned revision `dad2cfbe89a26f72f19770419469ac037de200df`; **not** the older UCI HARTH v1.2 release.
- Original signals: back and thigh tri-axial acceleration (six channels), 50 Hz; window length 100 and stride 50 with time continuity and pure labels; original codes 1–8.
- Development: 29 people; 14,201 bounded windows. Participants S023 and S032 were reserved for the qualified final report, 500 windows each.
- Stage 23 development 1D-ResNet three-fold mean Macro F1: 0.8328 (fold SD 0.0435). Stage 24 repeated-seed weighted-CE ResNet mean Macro F1: 0.8286 (across-seed SD 0.0052).
- Stage 26 qualified pooled fixed-eight-class Macro F1: ResNet 0.7182; fixed Random Forest 0.6817. Accuracy 0.9020 / 0.9270. These are **not** an externally validated or historically blind result. S023/S032 were inspected descriptively during earlier exploration.
- Primary evidence: [Notebook 23](../studies/wearable_activity_recognition/notebooks/23_unified_matched_model_benchmark.ipynb), [24](../studies/wearable_activity_recognition/notebooks/24_multiseed_and_imbalance_robustness.ipynb), [25](../studies/wearable_activity_recognition/notebooks/25_model_selection_and_protocol_freeze.ipynb), [26](../studies/wearable_activity_recognition/notebooks/26_final_development_training_and_inference.ipynb), [27](../studies/wearable_activity_recognition/notebooks/27_final_research_evidence_and_report.ipynb).

## Historical Stage 23/24 research evidence

The current `notebooks/23_unified_matched_model_benchmark.ipynb` is a later, presentation-updated version. The original approved execution is preserved in pinned Git history and its scientific results are verified against that historical record. The current file SHA256 differs from the byte-level fingerprint retained in the approved Stage 25 manifest; this does not replace or invalidate the historical record.

The original executed Stage 23 notebook is retained in **pinned repository Git
history**, rather than as a second file in the current project tree. The
provenance audit reads its historical bytes from commit
`433f972676a3a0a39c05b34612f201bb27366d36`, verifies the exact SHA256
against the unchanged approved Stage 25 manifest, and compares the active
Notebook 23's scientific code and recorded results. This
requires a full Git checkout, rather than a source ZIP without history.

Stage 24's **original** executed notebook is retained at Git commit
`c4548596656acaffad4b024203e554c2c3aa3da0` and verified against its
unchanged approved SHA256. The historical file includes an initial Colab
setup cell; the active notebook contains twelve matching scientific code cells.
A subsequent replay reused cached outputs for 18 of the 36 Development jobs.
The audit matches those restored jobs to previously completed, scored
historical runs and checks scientific code, tables and figures against the
approved evidence, allowing only known non-scientific runtime or display
differences. Cached jobs are never described as newly trained runs.

Stage 24's original executed bytes are verified through a separate pinned historical revision. The approved protocol records and six frozen model modules remain unchanged. Stages 26–27 preserve their scientific evidence, and the committed Stage 27 report has a successful final Quality Gate.
This review does not rerun model training, qualified Test scoring or independent
replication.

The historical evidence register and three Stage 25 protocol records are
available under
[`evidence/`](../studies/wearable_activity_recognition/evidence/).
These files retain their original bytes. Historical paths embedded in approved
JSON records describe past provenance and are not updated to current locations.
The primary notebooks contain saved visible results, while large generated
model and report artefacts remain in Git-ignored `outputs/`.

## Frozen model selection and qualified evaluation

The final models and evaluation settings were selected using Development evidence alone. The historical Stage 25 proposal records remain at their original `PENDING_USER_REVIEW` status; a separate `APPROVED` manifest documents the specific one-time qualified held-out evaluation reported in Notebook 26. All three immutable files remain under [study evidence](../studies/wearable_activity_recognition/evidence/). Their source and approval integrity is checked without altering model parameters or re-evaluating the protected participants. For scientific methods and results, see the [consolidated study report](../studies/wearable_activity_recognition/HARTH_RESEARCH_METHODS_AND_EVIDENCE.md).

## Reproduction levels and what is available

| Level | Status | Conditions and limitations |
| --- | --- | --- |
| Review executed results | Available in GitHub | Saved notebooks and rounded displayed tables; not the raw prediction artifacts |
| Recreate Stage 27 report | Available without GPU or raw CSVs | Runs against committed executed notebooks and manifests |
| Check code/manifest provenance | Available without GPU or raw CSVs | Audit verifies six frozen module hashes and Stages 23–27 saved Quality Gates |
| Inspect saved model files | **Not verified in GitHub** | Three ignored files must be preserved separately and optionally supplied to the local audit |
| Re-run Stage 26 Development training | Requires raw HARTH, suitable dependencies and compute | No exact environment lockfile; strict byte-for-byte reproduction across machines is **not** established |
| Re-run qualified Test | **Do not repeat for model optimisation** | Originally approved one-time report exists; rerunning to adjust models would contaminate interpretation |
| Independent external validity | **Not established** | Only two historically inspected same-study final participants |

**Dependencies:** The repository uses `requirements.txt` with minimum versions and optional `requirements-dl.txt` for PyTorch. These are **not fully pinned lockfiles**. Executed Notebook 26 records PyTorch `2.11.0+cu128` and Tesla T4; other exact package versions and CUDA/system details have not been comprehensively archived. Do not claim guaranteed byte-for-byte recreation.

## Pre-release checks on a fresh checkout

From repository root using Python 3.11 and the standard dependencies:

```powershell
git pull origin main
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/check_harth_release_readiness.py
```

The last command is a **read-only** source/provenance check. It does not open source HARTH CSVs, deserialize models or download any files. It checks frozen-module SHA256, aligned approved/proposed manifests, source fingerprint evidence and the actual saved final Quality Gates in Notebooks 23–27.

### Windows checkout and source SHA256

The six frozen Python modules in the **GitHub main tree** have been independently checked against the approved Stage 25 SHA256 entries and match **exactly**. On Windows, `git status` can be clean while `core.autocrlf` has converted Git's original LF text to CRLF in the working tree, which changes the raw-byte SHA256.

The pre-release audit therefore performs a **strict CRLF-to-LF-only** equivalence test if and only if the exact raw-byte SHA256 does not match.

**Stage 26 frozen-protocol loading uses the same strict source-file rule.** On Windows, it accepts the original approved SHA256 or the exact same SHA256 after CRLF-to-LF conversion of a Git checkout. It does not change the approved JSON manifest, model source, settings, or any model artefact. The separate `sha256_file()` helper continues to check saved artefact bytes without line-ending normalisation. A mismatch after the permitted conversion remains a genuine stop condition: inspect local changes rather than altering the frozen record.
 It accepts only a byte-for-byte match to the historical approved hash after this single transformation. It reports affected paths as `Windows CRLF-to-LF-only verified paths`. Any other file modification is rejected; it neither rewrites the file nor relaxes or updates an approved scientific fingerprint.

To diagnose on Windows PowerShell:

```powershell
git status --short
git pull --ff-only origin main
git config --show-origin --get core.autocrlf
python .\scripts\check_harth_release_readiness.py
```

Never reset, overwrite or regenerate the frozen source/manifest merely to make the SHA256 test pass. If strict CRLF equivalence still fails, preserve the file and send the new error output for investigation.

### Optional local artefact inspection

Three trusted, originally saved Colab outputs were generated in `outputs/harth_stage26_final/`:

- `resnet_state_dict.pt` (PyTorch weights)
- `random_forest.joblib` (classical model)
- `inference_config.json` (channel order, Development scaling, labels, windows and metadata)

If you have securely retained them, copy to a **local ignored directory** and run:

```powershell
python scripts/check_harth_release_readiness.py --artifact-dir outputs/harth_stage26_final --require-artifacts
```

The audit checks presence, hashes and configuration metadata **without deserialising** PyTorch or joblib. It cannot prove original file-byte identity, because the executed Notebook 26 printed truncated SHA256 values and the original full artifact hashes were not independently archived. Preserve the original trusted model files and their newly calculated hashes externally. Do **not** download an arbitrary joblib/pickle model and execute it; deserialization may run untrusted code.

If the directory no longer exists, report the artefacts as **unverified**, not as broken research or as proven reproducible. Do not recreate them using Test outcomes.

## Source data, rights and repository safety

- Original HARTH CSVs, any derivative participant windows and outputs remain excluded under `data/raw/`, `data/processed/` and `outputs/`. Model files, binary data archives and ordinary credential files are also Git-ignored.
- MIT applies only to original project code and accompanying repository documentation. This does **not** automatically relicense the official HARTH dataset, BoneNet derivative or any third-party research content.
- The public Git tree was checked for accidentally tracked model files and data archives; a heuristic scan of the six largest/most important executed HARTH notebooks found no familiar token signatures. **This is not a complete security audit or a guarantee about historical commits, other repositories or unrecognised secrets.**
- A repository-history secrets scan and manual checking of source licences remain sensible before broad redistribution. The current audit tool does not require network access and does not scan Git history.
- A GitHub Actions green run checks its configured CPU/offline tests and earlier small notebooks. It does **not** re-run 36 GPU fits or complete blind Test validation.

## Release checklist

- [x] All central Stage 23–27 executed notebooks available; frozen source/approval lineage documented.
- [x] Main and HARTH README updated to report observed results and limitations accurately.
- [x] GitHub Actions on reviewed scientific baseline `33d553e33705839b17f9f7190c8ec747d53956c1` succeeded on 21 October 2026 ([run 39601607708](https://github.com/RezaSaadatyar/human-movement-intelligence/actions/runs/39601607708)): 144 offline tests reported `OK` (30 skipped), and configured notebook checks succeeded. Verify CI again after any further commits; this is not an independent rerun of Stage 26 Test scoring.
- [x] Pre-release, read-only source provenance and optional local artifact audit tool implemented.
- [ ] After the final documentation/QA commit, confirm a successful GitHub Actions run on that exact new head; the preceding reviewed baseline was green. Optional local model-artifact verification is separate.
- [ ] User confirms whether the three original trusted Stage 26 model files are retained and where (local secure storage only).
- [ ] Verify source/licence attribution and review history for secrets before promoting an archive.
- [x] Obsolete [draft PR #1](https://github.com/RezaSaadatyar/human-movement-intelligence/pull/1) was closed without merge, preserving history.
- [ ] **Only after user authorisation:** decide on a named GitHub release. Publishing a release is optional and separate from research execution.

A GitHub Release, if authorised later, means an immutable versioned reference (tag) plus brief, scientifically bounded release notes. It **does not** mean publishing a paper or distributing private model/data files.

## Interpretation

Do not claim clinical readiness, prospective evaluation, statistically significant superiority, independent population validation, or perfect class recall based on low sample support. Record rare classes honestly: Shuffling 3, ascending stairs 10, descending stairs 6 selected Test windows. Overlapping Test windows are correlated and are not independent subjects.

## Stage 27 execution and research-evidence provenance

The Stage 27 report contains **ten scientific code sections** plus one
Google Colab setup cell. Its setup safely clones a missing checkout or
fast-forward updates the existing `main` checkout, then checks committed
Stage 26 evidence before report synthesis.

**Current status:** the committed Stage 27 notebook has saved output in
its saved results, no recorded runtime errors, and the final
`STAGE 27 | SOURCE-GROUNDED RESEARCH SYNTHESIS QUALITY GATE: PASSED`
marker. This is a completed **recorded report synthesis**, not a new
training run, independent replication or Test evaluation.

Stage 26 and its approved qualified held-out results remain unchanged,
read-only inputs to Stage 27. Original research and approval records
remain accessible through Git history.
