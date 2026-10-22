"""Auditable selection evidence from executed HARTH Stage 23/24 notebook outputs.

Never fits models, reads original Test data, or invents missing scores.
"""
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
import pandas as pd

from core.harth_provenance import (
    read_historical_stage23_bytes,
    read_historical_stage24_bytes,
    verify_stage24_visual_format,
    verify_stage23_visual_format,
)


class _SavedTable(HTMLParser):
    """Read single-header Pandas HTML tables without optional HTML dependencies."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables=[]
        self.area=None
        self.active=False
        self.in_row=False
        self.in_cell=False
        self.head=[]
        self.body=[]
    def handle_starttag(self,tag,attrs):
        if tag=="table":
            self.active=True
            self.head,self.body=[],[]
        elif self.active and tag in ("thead","tbody"):
            self.area=tag
        elif self.active and tag=="tr":
            self.in_row=True
            self.row=[]
        elif self.in_row and tag in ("td","th"):
            self.in_cell=True
            self.value=[]
    def handle_data(self,data):
        if self.in_cell:
            self.value.append(data)
    def handle_endtag(self,tag):
        if self.in_cell and tag in ("td","th"):
            self.row.append("".join(self.value).strip())
            self.in_cell=False
        elif self.in_row and tag=="tr":
            if self.area=="thead":
                self.head.append(self.row)
            elif self.area=="tbody":
                self.body.append(self.row)
            self.in_row=False
        elif tag in ("thead","tbody"):
            self.area=None
        elif tag=="table" and self.active:
            self.tables.append((self.head,self.body))
            self.active=False


def read_html_table(html):
    """Return complete flat saved table; reject grouped/pivot headings."""
    parser=_SavedTable()
    parser.feed(html)
    if len(parser.tables)!=1:
        raise ValueError("Expected exactly one displayed research table.")
    heads,rows=parser.tables[0]
    if len(heads)!=1 or not rows or any(len(r)!=len(heads[0]) for r in rows):
        raise ValueError("Incomplete or multi-level displayed table.")
    frame=pd.DataFrame(rows,columns=heads[0])
    if "" in frame:
        frame=frame.drop(columns=[""])
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate source table columns.")
    for key in frame:
        try:
            frame[key]=pd.to_numeric(frame[key],errors="raise")
        except (TypeError,ValueError):
            pass
    return frame


def _joined(item):
    return "".join(item) if isinstance(item,list) else str(item)


def _validate_executed_notebook(nb, stage):
    """Require saved executed code cells and an explicit final Quality Gate."""
    cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    if not cells or any(c.get("execution_count") is None for c in cells):
        raise ValueError(f"Stage {stage} contains unexecuted code.")
    if any(o.get("output_type") == "error"
           for c in cells for o in c.get("outputs", [])):
        raise ValueError(f"Stage {stage} contains saved errors.")
    text = "".join(_joined(o.get("text", "")) for o in cells[-1].get("outputs", []))
    if f"Stage {stage} scientific quality gate: PASSED" not in text:
        raise ValueError(f"Stage {stage} final quality gate missing.")


def _load_executed(path, stage):
    path = Path(path)
    raw = path.read_bytes()
    nb = json.loads(raw.decode("utf-8"))
    _validate_executed_notebook(nb, stage)
    return nb, hashlib.sha256(raw).hexdigest()


def _find_frame(nb,section,columns):
    cells=[c for c in nb["cells"] if c.get("cell_type")=="code"
           and section in _joined(c.get("source",""))]
    if len(cells)!=1:
        raise ValueError(f"Missing/ambiguous evidence section {section}.")
    found=[]
    for output in cells[0].get("outputs",[]):
        html=_joined(output.get("data",{}).get("text/html",""))
        if "<table" not in html:
            continue
        try:
            frame=read_html_table(html)
        except ValueError:
            continue
        if set(columns).issubset(frame.columns):
            found.append(frame)
    if len(found)!=1 or found[0].isna().any().any():
        raise ValueError(f"Missing/incomplete saved metrics: {section}.")
    return found[0]


def _source_fingerprint(nb,label):
    streams="\n".join(
        _joined(o.get("text","")) for c in nb["cells"]
        for o in c.get("outputs",[]) if o.get("output_type")=="stream"
    )
    observed=set(re.findall(re.escape(label)+r":\s*([a-f0-9]{64})",streams))
    if len(observed)!=1:
        raise ValueError(f"Ambiguous or missing {label}.")
    return next(iter(observed))



def _load_stage23_with_verified_visual_format(path):
    """
    Read immutable saved Stage 23 evidence after user-approved format changes.

    The active notebook is the primary readable copy. Its executable lines,
    execution counts and saved outputs must match the original executed
    notebook recovered from pinned Git history. The historical SHA256 is
    checked against the unchanged approved Stage 25 manifest.
    """
    path = Path(path)
    root = path.resolve().parents[3]
    approved = json.loads(
        (root / "wearable_activity_recognition/evidence/"
         "stage_25_protocol_approved_qualified_test.json").read_text(
            encoding="utf-8"
        )
    )
    original_bytes = read_historical_stage23_bytes(
        root, approved["stage23_saved_notebook_sha256"]
    )
    historic = json.loads(original_bytes.decode("utf-8"))
    _validate_executed_notebook(historic, 23)
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    current, _ = _load_executed(path, 23)
    verify_stage23_visual_format(path, original_bytes)
    return historic, original_sha



# Stage 24's approved executed notebook remains in pinned Git history.
# Its current presentation omits only Colab setup and has two missing counters.
# Verify study code, outputs, and unchanged metadata before using the archive.
def _load_stage24_with_verified_visual_format(path):
    """Return hash-approved historical Stage 24 evidence after strict comparison."""
    path = Path(path)
    root = path.resolve().parents[3]
    approved = json.loads(
        (root / "wearable_activity_recognition/evidence/"
         "stage_25_protocol_approved_qualified_test.json").read_text(
            encoding="utf-8"
        )
    )
    expected_sha = approved["stage24_saved_notebook_sha256"]
    historical = read_historical_stage24_bytes(root, expected_sha)
    verify_stage24_visual_format(path, historical)
    executed = json.loads(historical.decode("utf-8"))
    _validate_executed_notebook(executed, 24)
    return executed, expected_sha


def load_selection_evidence(stage23_path,stage24_path):
    """
    Recover evidence linked to completed original notebooks.

    Args:
        stage23_path (str | Path): Executed seven-model benchmark notebook.
        stage24_path (str | Path): Executed multi-seed/loss notebook.

    Returns:
        dict: Source-linked DataFrames, notebook SHA256 digests, common
        original Development tensor fingerprint.

    Notes:
        Jupyter display tables are rounded, so they are selection evidence,
        NOT substitutes for full-precision model predictions.
    """
    # 1. Verify both notebooks were fully executed without saved errors.
    n23,h23=_load_stage23_with_verified_visual_format(stage23_path)
    n24,h24=_load_stage24_with_verified_visual_format(stage24_path)
    sha23=_source_fingerprint(n23,"Source SHA256")
    sha24=_source_fingerprint(n24,"Development source SHA256")
    if sha23!=sha24:
        raise ValueError("Different Development window fingerprints.")

    # 2. Recover the COMPLETE evidence tables, not copied summary claims.
    a=_find_frame(n23,"06. Complete Benchmark Results",
                  ["model","macro_f1_mean","macro_f1_sd"])
    b=_find_frame(n23,"08. Minority-Class Reporting",
                  ["model","class_code","support","precision","recall","f1"])
    c=_find_frame(n24,"06. Complete Held-Out Evidence",
                  ["architecture","loss_method","seed","fold","macro_f1"])
    d=_find_frame(n24,"10. Person-Wise Robustness",
                  ["architecture","loss_method","seed","mean_person_macro_f1"])
    e=_find_frame(n24,"11. Evidence-Bounded Robustness",
                  ["architecture","loss_method","mean_across_seed_fold_macro_f1",
                   "across_seed_sd"])

    # 3. Reject incomplete evidence and inconsistent training combinations.
    required={"Random Forest","Standard CNN","Weighted CNN","TCN",
              "Multi-Scale CNN","1D-ResNet","Time-Series Transformer"}
    if len(a)!=7 or set(a.model)!=required or len(b)!=56:
        raise ValueError("Stage 23 model/class evidence incomplete.")
    if len(c)!=36 or len(d)!=12 or len(e)!=4:
        raise ValueError("Stage 24 expected tables incomplete.")
    if (c[["architecture","loss_method","seed","fold"]].duplicated().any()
        or set(c.seed)!={42,123,2026} or set(c.fold)!={1,2,3}):
        raise ValueError("Stage 24 Fold/seed evidence invalid.")
    if not c.macro_f1.between(0,1).all():
        raise ValueError("Stage 24 metric outside its valid domain.")
    return {"stage23":a,"stage23_classes":b,"stage24_folds":c,
            "stage24_people":d,"stage24_summary":e,
            "development_sha256":sha23,"notebook23_sha256":h23,
            "notebook24_sha256":h24}
