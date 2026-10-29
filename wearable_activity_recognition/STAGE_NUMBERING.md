# HARTH stage numbering (29 October 2026)

The original Stage 20 research-evidence consolidation notebook was moved to
`code/notebooks/final.ipynb`. The seven subsequent active research stages
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

The `final.ipynb` review draft has had its Git merge markers resolved and
is valid notebook JSON. However, it still reflects the former Stage 20
consolidation scope, including Stage 20 output names and an archived saved
execution error. It is not yet the completed final synthesis for active
Stages 01–26; its stage-specific output names must be reconciled before
running it alongside active Stage 20. Its workflow image is retained under
`images/final_workflow.svg`.

The earlier release-readiness/provenance tests intentionally refer to frozen
Stage 23/24/25/26/27 identities. Those tests require a separate, traceable
migration to distinguish historical verified evidence from current display
numbering. Do not treat updated stage labels as updated approved SHA256s.
