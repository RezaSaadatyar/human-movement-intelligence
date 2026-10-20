# Stage 26 preflight blocker: original Stage 25 source hash mismatch

The committed Stage 25 `stage_25_protocol_proposal.json` records **PENDING_USER_REVIEW** and a SHA256 fingerprint for six source modules. A Stage 26 CI run (38772427888) determined that the committed `advanced_temporal_models.py` bytes do **not** match the fingerprint embedded in that proposal. Other files may also differ; review all six.

**Do not delete this guard, modify source checks to always pass, mark the proposal APPROVED automatically, train final models on unverified code, or load S023/S032.**

## Controlled reconciliation procedure (no Test use)

1. Run `git pull origin main` and use the *same repository checkout* when reviewing code and regenerating SHA256 fingerprints.
2. Compare the six recorded `module_sha256` entries with SHA256 values calculated from the current files, without silently overwriting the original Stage 25 proposed protocol.
3. Investigate the cause of the difference (different checkout, post-Stage 25 code update, earlier Colab environment, or other source difference).
4. Only after an explicit review, create **a new pending** Stage 25 JSON revision with current module hashes while leaving every frozen **methodological setting** unchanged. Record the original and new proposal versions and the rationale for change. Do not substitute a new model/hyperparameter or use any reserved Test information.
5. Confirm that `load_final_protocol(new_path, root, require_approval=False)` succeeds and that its status remains `PENDING_USER_REVIEW`.
6. Update the Stage 26 notebook's `PROPOSAL` path only to the specifically reviewed pending revision. Then execute Development fitting. A separate explicit Test evaluation approval is still required.

The Stage 26 notebook currently points at the **original** protocol and should correctly block final training until the mismatch is resolved. The source hash mismatch is an integrity checkpoint, not evidence that the model is faulty.

A synthetic-test assertion for constant-variance channel standardisation was also corrected: the true behaviour of the Train-only scaler replaces zero SD by 1e-6, so a constant training channel with changed inference magnitude scales to a large, finite value. The revised test now uses a nonconstant Development source signal instead.

**Status:** Stage 26 implementation committed; **not yet cleared for final execution**; qualified Test evaluation **not authorised**.

## Reconciliation completed after user's clean-Colab audit (2026-10-15)

The user's Colab checkout at `e2d49b1` reported `git status --short` clean and all six SHA256 mismatches. Comparing its hashes with current GitHub source content confirmed all six bytes matched the committed source files. No uncommitted local editing explains the difference. **The underlying historical cause of the original mismatched recorded hashes has not been conclusively established.**

The original `research_outputs/stage_25_protocol_proposal.json` remains unchanged for provenance. A new `research_outputs/stage_25_protocol_proposal_reconciled.json` records all six verified GitHub source SHA256 values, the original hashes, the clean-audit context, and the unchanged source cohort, model, seed, batch size, optimiser, loss, metrics and Test constraints. Its status remains **PENDING_USER_REVIEW**. The Stage 26 Development-only notebook now points to this pending reconciled proposal. Test evaluation remains disabled.

After `git pull origin main`, run `python -m unittest discover -s tests -p "test_harth_final_inference.py" -v` and the first notebook cell. If the manifest validation succeeds, continue with Development-only training. **Do not mark the proposal APPROVED automatically or load reserved Test subjects**.
