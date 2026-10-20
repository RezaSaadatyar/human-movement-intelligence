# Stage 20 | Research evidence consolidation

## Purpose

Provide a complete, traceable scientific record of **Stages 01–19** of the HARTH v2.0 wearable activity recognition study before new temporal architectures are developed. This stage **does not retrain any model** and must not manufacture outputs or imply that saved notebooks have just been rerun.

## Deliverables

- `notebooks/20_research_evidence_consolidation.ipynb`: ten full analysis sections, including source cohort protocol, notebook execution-state audit, 19-stage evidence register, scientific comparison groups, class-retention plots, probabilistic diagnostics and research limitations. All user-facing Markdown section titles use **bold Markdown** instead of heading hashes.
- `evidence/stage_01_19_evidence_register.csv`: one row for each previous stage with exact notebook filename, question, protocol, reported evidence and major limitations. Figures are manually transcribed from saved source outputs and remain subject to cross-check.
- `code/python/src/core/notebook_evidence.py`: reusable read-only inventory of saved .ipynb JSON; never executes a notebook or downloads data.
- `tests/test_notebook_evidence.py`: offline checks for saved execution counts, output-bearing cells, error objects and registry linkage.

## Interpretation

The analysis explicitly distinguishes Stage 09's full 132,397-window five-fold classical baseline from capped-window CNN experiments; Stages 13, 16 and 17 use different matched comparison groups. Stage 18 out-of-fold uncalibrated confidence metrics cannot be compared as if it were the identical model training design used for Stage 19 temperature fitting with separate Calibration participants. The 8-class task excludes unofficially mapped labels only **after** time-safe segmentation, and early exploratory inspection of the original `S023`/`S032` Test IDs is disclosed.

The notebook's evidence checks verify the presence of saved results and stage files, not whether every previously reported numeric finding has been scientifically replicated. It flags incomplete execution counts, even if a cell contains a stored display output.

## Notebook style rule for Stage 20 onwards

Each Markdown section starts with **bold text using two asterisks**, e.g. `**03. Evaluation Design**`, rather than a `#` heading. Explanatory Markdown remains comprehensive, technical terms retain their accepted spelling, and reusable functions contain documented Args/Returns plus English comments.

Related analyses can share one complete notebook, but never at the expense of method, results, visualisations or limitations. Four future architectures (TCN, Multi-Scale CNN, 1D-ResNet, Transformer) remain compulsory parts of Stages 22–23.

## Execution in Cursor

```powershell
cd D:\human-movement-intelligence
git pull origin main
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_notebook_evidence.py" -v
```

Then open `studies/wearable_activity_recognition/notebooks/20_research_evidence_consolidation.ipynb` and execute every cell. **No raw HARTH dataset download or PyTorch training is expected.** Review the evidence table and any missing execution-count rows, preserve real outputs, commit/push the executed notebook and request review before Stage 21.

## Completion checkpoint

- All 19 source notebook files and registered stages align.
- All saved error outputs and missing execution counts are truthfully reported.
- Stage comparison groups and provisional outcomes are clearly distinguished.
- Final Test is reserved from subsequent development but is not characterised as historically unseen.
- Research conclusions remain at exploratory development level.
