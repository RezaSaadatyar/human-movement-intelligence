# HARTH stage numbering (29 October 2026)

The original Stage 20 research-evidence consolidation notebook was first moved to
`code/notebooks/final.ipynb`. Its historical science and saved output remain
recoverable from Git history. The outdated draft is now being superseded by the
newly written `code/notebooks/final_harth_research_audit_and_closure.ipynb`, a separate FINAL
cross-stage audit (not an additional numbered training or Test experiment). The seven subsequent active research stages
were each renumbered down by one. Their saved scientific notebook output
objects and execution counts were preserved; stage labels visible inside
previously saved output streams may still use the original number until the
notebooks are genuinely rerun and saved.

| Previous stage | Active stage | Active notebook |
| --- | --- | --- |
| 21 | 20 | `20_minority_class_and_data_quality.ipynb` |
| 22 | 21 | `21_advanced_temporal_architectures.ipynb` |
| 23 | 22 | `22_unified_matched_model_benchmark.ipynb` |
| 24 | 23 | `23_multiseed_and_imbalance_robustness.ipynb` |
| 25 | 24 | `24_model_selection_and_protocol_freeze.ipynb` |
| 26 | 25 | `25_final_development_training_and_inference.ipynb` |
| 27 | 26 | `26_final_research_evidence_and_report.ipynb` |

Stage-specific output filenames and SVG filenames in **active notebook
source** use the active stage number. Existing legacy output snapshots,
frozen Git commits, source hashes, approval-record keys and files named
`stage_25_protocol_*` retain their original historical identifiers and
must not be rewritten to make the numbering appear newer. In particular,
display-precision CSV snapshots with original 23/24/26 prefixes remain
historically attributed data sources until they are separately migrated and
verified, not relabelled measurements.

The earlier `final.ipynb` draft still reflected historical Stage 20 and had
`20_historical_*` output names and an archived execution error. The new FINAL
notebook uses `final_*` output names and intentionally begins without saved
execution outputs. Its first execution is pending; no new Test re-scoring is
allowed. `images/final_workflow.svg` depicts the historical consolidation
and is no longer used by the replacement notebook.

The earlier release-readiness/provenance tests intentionally refer to frozen
Stage 23/24/25/26/27 identities. Those tests require a separate, traceable
migration to distinguish historical verified evidence from current display
numbering. Do not treat updated stage labels as updated approved SHA256s.
