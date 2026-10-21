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

## 5. Stage-by-stage collaboration (historical development workflow)

Prepare just one complete stage; commit Python modules, tests, protocol and notebook to GitHub. The user pulls, runs the notebook locally in Cursor, inspects and pushes saved real outputs. The assistant then inspects the latest pushed artefacts and their Quality Gate **before** implementing the next stage. If a prior stage has not yet been pushed or is missing execution counts, explicitly report the status and seek completion rather than claiming verified success. Never pretend to have run computationally heavy notebooks locally.

The staged implementation workflow above describes how the original experiments were reviewed. All 27 numbered stages have now been executed and committed. Future work should preserve the same review discipline without treating historical stage-completion instructions as unfinished tasks.

## 6. Approved 2026 notebook visual-format standard

**Applies to all 27 HARTH study stages.** Sections use bold Markdown labels (for example `**05. Model Evaluation**`) rather than hash headings. Every notebook begins with a concise introduction and an **accessible, landscape left-to-right SVG workflow** from source input to scientific output. The diagrams are stored as separate editable vector files in [study images](../images/) and linked by relative paths from the first Markdown cell. Never imply a skipped experimental step or clinical validation in the diagram.

Python cells use a visibly delimited research-stage comment header, meaningful numbered purpose comments and, where useful, import blocks organised by **Python standard library**, **data and machine learning**, **figures and notebook display**, **deep learning**, and **local research utilities**. Do not introduce unused imports or ML packages from another project simply for visual symmetry. Preserve runtime import order, especially local `sys.path` configuration.

For significant results and final quality gates, use concise printed report dividers and plainly labelled `print` lines, with tables and plot titles conveying metrics, cohorts and units. Do not apply ornamental separators to every minor output. Any future print-format change must be executed and its actual notebook outputs saved before declaring it verified.

**Protection of scientific provenance:** Stages 01–22 were visually standardised directly in the primary notebooks. Their source and research results were preserved during formatting; saved outputs are present, but **15 historical code-cell `execution_count` values in Stages 01–07 remain unrecorded**, despite saved output evidence. Do not invent those values or describe their execution metadata as complete.

Stages 23 and 24 have immutable, separately hash-verified historical executed versions in pinned Git history. Their current primary notebooks were formatted and later genuinely replayed; non-scientific differences such as runtime measurements, cached-job messages, HTML identifiers and regenerated chart presentation can occur. Provenance validation must compare executable code and scientific values with the approved records, not simply assume the current notebook bytes match the old frozen SHA256. The original Stage 25 proposal and reconciled proposal both retain their historical `PENDING_USER_REVIEW` status; a **separate approved manifest** documents the specific one-time qualified same-study held-out Test evaluation.

Stages **25–27** are recorded as executed in the current study. Stage 26 has 13 executed code cells and its approved qualified held-out Quality Gate is saved. Stage 27 has **11** executed code cells (one Colab setup plus ten research sections), no saved errors and its read-only report Quality Gate is saved as `PASSED`. The earlier statement that Stage 27 had five missing execution counts applied to an older historical version, **not** to the current committed notebook. Do not rewrite approved manifests, fabricate execution metadata, re-score the one-time qualified Test to tune a model, or treat a scientific report as external validation.

Before treating a formatted notebook as a new executed scientific run, open it in Cursor/Jupyter and rerun only appropriate non-frozen cells. Review saved results, figure rendering and CI. **Do not rerun the one-time qualified Test to redesign a model.**
