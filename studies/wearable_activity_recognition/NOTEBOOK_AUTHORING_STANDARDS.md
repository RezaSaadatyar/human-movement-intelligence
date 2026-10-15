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
