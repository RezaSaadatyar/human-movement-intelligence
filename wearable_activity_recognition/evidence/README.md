# HARTH Research Evidence and Approval Records

This directory contains small, version-controlled research evidence and approval records produced
during the HARTH study. It is not a cache of fitted models, raw signals,
full-precision predictions or newly calculated performance results.

| File | Purpose |
| --- | --- |
| `stage_01_19_evidence_register.csv` | Historical stage-by-stage evidence inventory. |
| `stage_25_protocol_proposal.json` | Original pending selection proposal. |
| `stage_25_protocol_proposal_reconciled.json` | Reconciled pending proposal with source fingerprints. |
| `stage_25_protocol_approved_qualified_test.json` | Separate approved record for one qualified same-study held-out evaluation. |

The four research records retain their original file bytes, including
historical pointers to their former `experiments/` directory. Those pointers
are historical provenance, not current links. Do not rewrite an approved
record merely to rename a directory.

The actual executable notebooks and saved visible outputs remain in
[`../code/notebooks/`](../code/notebooks/). Downloaded datasets and locally generated
model artefacts, prediction tables and report files are stored under the
repository's ignored `data/` or `outputs/` directories; they are **not**
committed here.

The original executed Stage 23 notebook no longer has a duplicate file.
Its exact approved historical bytes are retrieved from pinned repository Git
history and verified against the unchanged SHA256 in the approved manifest.
A full Git checkout with history is required for that provenance audit.
