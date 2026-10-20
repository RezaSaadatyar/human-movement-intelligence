# HARTH Notebook Authoring and Output Standards

**Scope:** Applies to every new notebook and applicable reusable Python function in Stages 21–27, as well as any future notebook clean-up. Do not silently rewrite previously executed notebooks or delete original research evidence.

## 1. Detailed Markdown without hash headings

Use **bold Markdown titles** rather than Markdown heading markers for notebook section names, e.g. `**04. Per-Participant Activity Coverage**`. Provide substantive context: research objective, data assumptions, precise methods, why the next cell matters, expected outputs, scientific caveats and interpretation boundaries. When sections belong together, consolidate them **without dropping** documentation, figures, metrics or tests. Never omit a relevant model just to minimise notebook count.

## 2. Numbered, meaningful code comments

Every meaningful code cell should begin with a visually consistent single-line code comment, e.g.:

```python
# ==================== 04. Participant and Class Coverage ====================

# 1. Verify all source participants belong to the development cohort.
...

# 2. Count genuine source activity labels and include absent classes.
...

# 3. Compare participant-level prevalence without fitting any model.
...

# 4. Display the counts and confirm expected data alignment.
...
```

Use `#` for **Python comments inside code**, not Markdown section headings. Number code steps in execution order, with each comment explaining **why** the block is needed and what it accomplishes, not simply repeating the Python statement. Break up long cells logically; keep related operations together. Provide complete `Args`, `Returns` and limitations in reusable `.py` functions. Source code and notebook explanations remain in professional English.

## 3. Legible, clearly delimited outputs

Print concise, readable summaries with consistent separators and section titles for significant result blocks:

```python
print("\n" + "=" * 60)
print("STAGE 21: MINORITY-CLASS DATA AUDIT")
print("=" * 60)
print("Development participants verified:", len(development_ids))
print("Reserved Test loaded: NO")
print("Data-quality checks: PASSED")
print("=" * 60)
```

Display tabular results with descriptive column names, explicit units and informative ordering. Keep table and figure titles precise; use one figure for related panels and label legend, units, source group and colour categories clearly. Avoid misleading accuracy-only summaries, unsupported 'best model' badges, uncited numbers or stale saved outputs.

## 4. Scientific quality gates and reproducibility

Each notebook concludes with numbered assertions covering data shape and provenance, actual classification target, independence of subjects, completeness of expected outputs, no use of protected Test data where applicable, and reproducibility conditions. Distinguish:
- code successfully ran,
- an audit of actual source data passed,
- a model was evaluated on development data,
- an independent final evaluation was performed.

Assertions must match the actual scope and dynamically handle project growth. **Stage 20 originally failed when total inventory reached 20 while a check expected 19; inspect historical stages 01–19 separately from subsequent notebooks**. Do not delete, mute or bypass errors; fix incorrect assumptions. Avoid declaring 'ALL TESTS PASSED' before actually observing test results.

## 5. Stage-by-stage collaboration

Prepare just one complete stage; commit Python modules, tests, protocol and notebook to GitHub. The user pulls, runs the notebook locally in Cursor, inspects and pushes saved real outputs. The assistant then inspects the latest pushed artefacts and their Quality Gate **before** implementing the next stage. If a prior stage has not yet been pushed or is missing execution counts, explicitly report the status and seek completion rather than claiming verified success. Never pretend to have run computationally heavy notebooks locally.

## 6. Approved 2026 notebook visual-format standard

**Applies to all 27 HARTH study stages.** Sections use bold Markdown labels (for example `**05. Model Evaluation**`) rather than hash headings. Every notebook begins with a concise introduction and an **accessible, landscape left-to-right SVG workflow** from source input to scientific output. The diagrams are stored as separate editable vector files in [study images](images/) and linked by relative paths from the first Markdown cell. Never imply a skipped experimental step or clinical validation in the diagram.

Python cells use a visibly delimited research-stage comment header, meaningful numbered purpose comments and, where useful, import blocks organised by **Python standard library**, **data and machine learning**, **figures and notebook display**, **deep learning**, and **local research utilities**. Do not introduce unused imports or ML packages from another project simply for visual symmetry. Preserve runtime import order, especially local `sys.path` configuration.

For significant results and final quality gates, use concise printed report dividers and plainly labelled `print` lines, with tables and plot titles conveying metrics, cohorts and units. Do not apply ornamental separators to every minor output. Any future print-format change must be executed and its actual notebook outputs saved before declaring it verified.

**Protection of scientific provenance:** Stages 01–22 received Markdown/code-comment-only formatting updates in the existing notebooks (apart from Stage 01's purely cosmetic final print delimiters). Following the maintainer's explicit authorisation, **Stage 23's original notebook path was also formatted directly**. Its original executed bytes are preserved verbatim in [frozen_notebook_evidence](experiments/frozen_notebook_evidence/23_unified_matched_model_benchmark.ipynb) for the approved Stage 25 SHA256 record. The primary Stage 23 notebook keeps identical executable statements, saved outputs and execution counts; evidence-loading and read-only audit tools verify this equivalence against the archive. Stages **24–25** are also formatted directly in `notebooks/` with unchanged executable code and saved outputs; the historical Stage 24 version is available through Git history rather than a new archive. Stages **26–27** are likewise formatted directly in the primary research notebooks without modifying code execution, saved outputs or execution-count fields; duplicate presentation notebooks are no longer maintained. Stage 27 retains five original missing execution counts, which are reported rather than invented. Do not modify the approved historical manifest, fabricate execution counts or re-evaluate protected Test subjects.

Before treating a formatted notebook as a new executed scientific run, open it in Cursor/Jupyter and rerun only appropriate non-frozen cells. Review saved results, figure rendering and CI. **Do not rerun the one-time qualified Test to redesign a model.**
