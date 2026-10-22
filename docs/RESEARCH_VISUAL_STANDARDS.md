# Research Visual Documentation Standards

**Scope:** Visual diagrams for the `human-movement-intelligence` root README, implemented study READMEs and individual Jupyter Notebooks. Diagrams communicate research structure and verified evidence; they do not replace code, experiments or source citations.

## 1. The figure hierarchy

| Location | Required illustration | Scope and purpose | Timing |
| --- | --- | --- | --- |
| Root `README.md` | **Repository-wide research map** (`images/repository_overview.svg`) | Identify independently implemented, exploratory and planned studies; show reusable research engineering practice without suggesting all studies completed every step. | Update when the research catalogue or study status changes. |
| Each implemented `<subproject_root>/README.md` | **Study-specific overview** (`studies/<study>/images/study_overview.svg`) | Show the actual source, research question, key protocol stages, model or audit scope, evaluation boundaries, observed evidence and limitations. | Create from verified study design and revise only when its evidence changes. |
| Each study Notebook | **Stage-specific workflow** (`../images/NN_workflow.svg`) | Show only the operations performed in that Notebook, with a clear source-to-output workflow and stage-relevant scientific context. | Add as the **last content-development step**, after code execution, results review and scientific approval. |
| Executed research charts | **Evidence figures** | Show observations, metrics, cohorts, units and limitations from code output; do not conflate with explanatory SVGs. | Generate from executed analyses and preserve their outputs. |

Planned directions do not require fictitious completed-experiment figures. Their planned status must remain explicit in overview diagrams.

## 2. Required scientific distinction

The root image is a **portfolio map**, not a single experiment. A study README image summarises its **end-to-end scientific protocol**. A Notebook image summarises **one implementation stage**. They must not be the same image renamed or reproduced at different levels.

Study scope and verified status take precedence over visual symmetry. For example:

- HARTH has executed 27 stages, including Development modelling and a qualified two-person, same-study held-out assessment. Those people were inspected descriptively earlier; the figure must not describe the assessment as blind, external or clinical validation.
- BoneNet has executed derivative source inspection and a preliminary regression pilot, but the source-record-to-participant mapping is not independently verified. Its study figure must say **exploratory**, not validated prediction.
- Gait, markerless capture, neurorehabilitation and multimodal projects remain planned unless supported by their own executed artefacts.

## 3. Approved visual design and geometry

Use the coherent scientific landscape style established by the HARTH stage diagrams:

- **Reference SVG:** `1700 × 480 px`, `viewBox="0 0 1700 480"`; editable vector, meaningful `<title>` and `<desc>` for accessibility.
- **Main frame:** rounded corners (`20 px`), background `#F4F9FD`.
- **Title and subtitle:** top of image, left aligned; approximately `23 px` bold and `15 px`.
- **Upper scientific-context layer:** reference `x=24, y=84, width=1652, height=160`, fill `#EAF3FA`. Arrange four coherent context areas or adapt for the scientific question: data/modality, cohort/schema, protocol, outputs/evidence and boundaries.
- **Lower process layer:** reference `x=24, y=262, width=1652, height=174`. Use input, numbered scientific operations and output with aligned rounded boxes, concise titles and left-to-right arrows.
- **Standard seven-block geometry:** 208 × 132 px blocks, initial x=38 and y=287, with 236 px horizontal increments. Six short arrows are centred at y=353, 2.2 px thick, using `#245E9D`.
- **Colours:** dark input/output `#173453`; white intermediate blocks with `#B8CFDE` borders; main text `#173453`; explanatory text `#506C7E`.
- **Caption:** concise source-to-evidence summary at approximately y=465.
- **Typography:** scientific, legible, concise; standard font stack `DejaVu Sans, Arial, sans-serif`. Avoid clipped labels, crowded columns and tiny text.

The **five intermediate steps in the seven-block example are not mandatory**. Use the number required to faithfully represent the scientific operations and adjust width, spacing or canvas as needed. **SVG step numbers are independent from Notebook Code Cell numbers**; they represent workflow operations, not a one-to-one notebook index. Avoid artificial stage splitting or merging to force a fixed count.

Do not embed large SVG source strings into Markdown. Save figures separately and link them with repository-relative paths.

## 4. Content integrity and language

- Use clear, concise, natural British English. Avoid promotional or formulaic text, redundant descriptions, invented findings and internal implementation notes.
- Identify the actual scientific inputs and outputs, source version, evaluation unit and data/protocol boundaries.
- Make execution status visible: **implemented**, **exploratory**, **planned**, **approved**, **qualified** or **not verified**, as appropriate.
- Report numbers only from verifiable saved output or approved documentation, and use the correct evaluation denominator.
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
7. For Notebook workflows, create the SVG **after** the notebook execution-review gate, then seek final image approval before continuing to the next notebook.

Keep the root visual, individual study diagrams and Notebook illustrations separate. Do not silently revise previously approved numerical results to fit a diagram.
