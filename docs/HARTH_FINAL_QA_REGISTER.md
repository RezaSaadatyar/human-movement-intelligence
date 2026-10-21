# HARTH v2.0 | Consolidated Final QA Register

**Reviewed on:** 21 October 2026. **Scope:** original Notebooks 01–27 in `studies/wearable_activity_recognition/notebooks/`, source-controlled protocol evidence, project Markdown navigation, offline provenance checks and GitHub Actions. This is a source/output audit; it is **not** an independent rerun of GPU experiments or qualified Test evaluation.

## Reviewed scientific baseline

- **GitHub scientific baseline:** `33d553e33705839b17f9f7190c8ec747d53956c1`. [GitHub Actions run 39601607708](https://github.com/RezaSaadatyar/human-movement-intelligence/actions/runs/39601607708): **success**; 144 offline unit tests `OK`, 30 skipped, configured small notebook checks successful. Future documentation changes require a new head-specific CI check.
- **Stages 01–27:** all 27 versioned notebooks are present with saved outputs and **zero saved error outputs**. All 27 final cells contain printed descriptive success/quality-gate evidence where applicable. The recorded executed status does not independently prove bit-identical reruns across machines.
- **Stage 23:** seven models in participant-disjoint three-fold Development evaluation, 21 jobs, original historical notebook preserved by approved Git SHA256. Do not substitute exploratory Stages 01–19 scores into its matched ranking.
- **Stage 24:** 36 matched Development jobs across 12 model/loss/seed conditions. Eighteen jobs in the latest saved replay were **RESTORED from an existing cache** and 18 ran with their saved training logs. The provenance verifier checks cache keys against the approved historical executed jobs and checks scientific result tables; cached jobs must not be called fresh retraining.
- **Stage 25:** Development-based model selection and historical proposal/reconciliation/approval manifests remain distinct. No replay created a new Test authorisation.
- **Stage 26:** 13/13 code cells with saved output and a final PASSED gate. Qualified same-study held-out evaluation: S023 and S032, 500 overlapping windows each. Primary 1D-ResNet Weighted CE eight-class Macro F1 `0.7182`; predeclared Random Forest reference `0.6817`. Both participants had previously been inspected descriptively: **not blind/external validation**.
- **Stage 27:** setup plus ten scientific code sections, **11/11 executed code cells** with saved outputs, no saved errors, final source-grounded synthesis gate **PASSED**. Tables reflect rounded displays from saved evidence; Stage 27 does not refit models, reload raw Test recordings or rescore Test.

## Deduplicated findings and disposition

| Group | Finding | Action / current status |
| --- | --- | --- |
| **01–08** | Historical execution metadata gap: Stage 01 cells 6–7; 02 cell 2; 03 cells 1–3; 04 cells 1–4; 05 cells 6–7; 06 cells 5–6; 07 cell 6. **15 cells total** have `execution_count: null` while retaining real saved outputs; Stage 08 is fully counted. | **Documented, not fabricated.** Only a genuine maintainer rerun and saved updated notebook can close this metadata gap. Do not change those counts by hand. |
| **01–08** | Original exploratory work encountered/inspected participants later used in the qualified held-out assessment. Cohorts and class coverage differ from the final 29-person matched Development setup. | **Scientific qualification retained.** Do not claim a blind Test set or compare all early-stage model metrics as though from an identical benchmark. |
| **09–16** | Saved code-cell outputs and counts complete; no recorded error outputs; stage-specific quality messages present. | **No computational repairs required on examined saved evidence.** Nested Train/Validation, sampling and ablation analyses are distinct experimental settings. |
| **17–24** | Saved outputs/counts complete; Stage 23/24 research evidence differs from historical original bytes due to presentation/replay and cached runtime messages. | **Existing historical SHA256 reconciliation and evidence-aware checks retained.** Do not rewrite original approved manifests or count RESTORED jobs as newly trained fits. |
| **25–27** | Older path names, Windows line endings, strict HTML rowspan parsing and source/approval lineage previously caused notebook import/evidence-check failures. | **Already corrected in source; current saved gates PASSED.** Approved scientific settings, historical manifests and Test constraints remain unchanged. |
| **Offline CI** | Outdated ten-versus-eleven Stage 27 cell assertion and a Stage 25 expected-error text mismatch caused prior failures. | **Resolved.** Baseline GitHub Actions run 39601607708 green; new commit must be rechecked. |
| **Markdown navigation** | Stale links to moved Stage 27 protocol and HARTH images, and obsolete Stage 24/27 execution-count and CI status prose. | **Documentation corrections in final QA pass.** Maintain the old Roadmap checkpoint log as explicitly historical instead of rewriting it as current execution status. |
| **Scientific limitations** | Two historically inspected participants, correlated overlapping windows, rare class supports (3 Shuffling, 10 stairs ascending, 6 stairs descending), and no independent external/prospective validation. | **Must remain prominent** in README and reports. No clinical deployment or population generalisation claim. |

## Subsequent actions deliberately not performed here

1. **Optional historical metadata completion:** if the maintainer wishes to replace missing Stage 01–07 execution counters, actually rerun affected cells and save the genuine outputs; preserve the earlier commits for provenance.
2. **Local-only Stage 26 model artefacts:** verify secure copies of original `resnet_state_dict.pt`, `random_forest.joblib` and `inference_config.json` before making any full-artefact reproducibility claim. These files are not committed and should not be uploaded merely for repository QA.
3. **Release-adjacent safety:** complete source/licence attribution and appropriate repository-history secret review before broad redistribution. An existing CI success does not establish either of these.
4. **GitHub Release:** requires a separate, explicit authorisation from the maintainer. Do not tag a release, rescore Test or train new models as a housekeeping step.

For the current scientific summary, see the [HARTH study overview](../studies/wearable_activity_recognition/README.md). For detailed pinned-history and source checks, see [HARTH reproducibility and release readiness](HARTH_REPRODUCIBILITY.md). For old stage-by-stage planned work, see the [historical Roadmap](ROADMAP.md).
