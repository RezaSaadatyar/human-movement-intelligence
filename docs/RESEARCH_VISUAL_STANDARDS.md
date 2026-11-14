# Research Visual Documentation Standards

**Scope:** Visual diagrams for the `human-movement-intelligence` root README, implemented study READMEs and individual Jupyter Notebooks. Diagrams communicate research structure and verified evidence; they do not replace code, experiments or source citations.

## 1. The figure hierarchy

| Location | Required illustration | Scope and purpose | Timing |
| --- | --- | --- | --- |
| Root `README.md` | **Optional repository-wide research map** | Identify independently implemented, exploratory and planned studies; show reusable research engineering practice without suggesting all studies completed every step. | Update when the research catalogue or study status changes. |
| Each implemented `<subproject_root>/README.md` | **Study-specific overview** (`<subproject_root>/images/study_overview.svg`) | Show the actual source, research question, key protocol stages, model or audit scope, evaluation boundaries, observed evidence and limitations. | Create from verified study design and revise only when its evidence changes. |
| Each study Notebook | **Stage-specific workflow** (named to match the actual notebook, e.g. `../../images/13_harth_final_development_training_and_research_synthesis.svg`) | Show only operations performed in that notebook, with scientific context and a clear source-to-output workflow. | Update when the implemented operations or relevant findings change. |
| Executed research charts | **Evidence figures** | Show observations, metrics, cohorts, units and limitations from code output; do not conflate with explanatory SVGs. | Generate from executed analyses and preserve their outputs. |

Planned directions do not require fictitious completed-experiment figures. Their planned status must remain explicit in overview diagrams.

## 2. Required scientific distinction

If included, the optional root image is a **portfolio map**, not a single experiment. A study README image summarises its **end-to-end scientific protocol**. A Notebook image summarises **one implementation stage**. They must not be the same image renamed or reproduced at different levels.

Study scope and verified status take precedence over visual symmetry. For example:

- HARTH is organised into 13 notebooks covering Development modelling and qualified same-study assessment of two previously inspected participants; figures must not label that assessment as blind or external validation.
- The Essex biomechanical load study has eight research notebooks, participant-level model evaluation and model-derived finite-element strain targets. Figures must distinguish those model-derived targets from direct physical strain measurement.
- Gait, markerless capture, neurorehabilitation and multimodal projects remain planned unless supported by their own executed artefacts.

## 3. Visual design and geometry

Use the coherent scientific landscape style established by the HARTH stage diagrams:

- **Reference SVG:** `1700 × 816 px`, `viewBox="0 0 1700 816"`; editable vector, meaningful `<title>` and `<desc>` for accessibility.
- **Main frame:** rounded corners (`20 px`), background `#F4F9FD`.
- **Title and subtitle:** top of image, left aligned; approximately `23 px` bold and `15 px`.
- **Upper scientific-context layer:** reference `x=24, y=84, width=1652, height=160`, fill `#EAF3FA`. Arrange coherent areas for data/modality, population, protocol, results and evaluation boundaries.
- **Middle illustrated-evidence layer:** reference `x=24, y=258, width=1652, height=314`, fill `#EDF6FB`. Use labelled drawings, signals, representations or appropriately sourced numerical evidence to explain the actual work. Distinguish illustrative traces from measured data.
- **Lower research-process layer:** reference `x=24, y=586, width=1652, height=195`. Use source/input, scientific operations and current outputs with aligned rounded blocks and left-to-right arrows.
- **Seven-block example:** approximately 208 × 132 px per block, initial x=38 and y=610, with 236 px horizontal increments. Arrows connect adjacent blocks, using `#245E9D`.
- **Colours:** dark input/output `#173453`; white intermediate blocks with `#B8CFDE` borders; main text `#173453`; explanatory text `#506C7E`.
- **Caption:** concise source-to-evidence summary near the lower edge (approximately y=767).
- **Typography:** scientific, legible, concise; standard font stack `DejaVu Sans, Arial, sans-serif`. Avoid clipped labels, crowded columns and tiny text.

The **five intermediate steps in the seven-block example are not mandatory**. Use the number required to faithfully represent the scientific operations and adjust width, spacing or canvas as needed. **SVG step numbers are independent from Notebook Code Cell numbers**; they represent workflow operations, not a one-to-one notebook index. Avoid artificial stage splitting or merging to force a fixed count.

Do not embed large SVG source strings into Markdown. Save figures separately and link them with repository-relative paths.

## 4. Content integrity and language

- Use clear, concise, natural British English. Avoid promotional or formulaic text, redundant descriptions, invented findings and internal implementation notes.
- Identify the actual scientific inputs and outputs, source version, evaluation unit and data/protocol boundaries.
- Label an analysis **exploratory**, **Development**, **qualified same-study** or **planned** when this distinction affects scientific interpretation.
- Report numerical results only when supported by the corresponding output table or an identified original source, using the correct evaluation denominator.
- Do not represent retrieval of data, data checks, model training and independent validation as equivalent milestones.
- Supporting graphics must not imply patient diagnosis, clinical deployment, sensor placement or biological measurement unless the source confirms it.
- SVG is a **workflow illustration**, not a faithful explanation of internal model reasoning or independent proof of scientific performance.

## 5. Review and update sequence

1. Inspect the actual README, relevant code, saved outputs and scientific protocol.
2. Identify the intended level: **repository**, **study** or **notebook**.
3. Define scientific stages and confirmed outputs for that level.
4. Create the editable SVG, then render and inspect it for text overlap, arrows, contrasts, status labels and accessibility.
5. Insert the image **after the README title/subtitle and before substantive prose**, with informative alt text and a concise scope caption.
6. Confirm relative links and data claims. Preserve unrelated source code, outputs, execution counts and evidence.
7. Keep workflow diagrams aligned with the actual implemented notebook sections and update their links when filenames change.

Keep repository-level, study-level and notebook figures distinct. Scientific results must not be changed merely to fit a diagram.
