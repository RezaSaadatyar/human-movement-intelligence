"""Offline visual-layout checks: never load raw HARTH Test data or run models."""
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "wearable_activity_recognition"
SVG = {"http://www.w3.org/2000/svg"}


class TestHarthNotebookPresentation(unittest.TestCase):
    def test_every_stage_has_horizontal_workflow_and_notebook_reference(self):
        for n in range(1, 28):
            with self.subTest(stage=n):
                name = f"{n:02d}"
                images = list((STUDY / "images").glob(f"{name}_workflow.svg"))
                self.assertEqual(len(images), 1)
                doc = ET.parse(images[0]).getroot()
                self.assertEqual(doc.tag, "{http://www.w3.org/2000/svg}svg")
                width = int(doc.attrib["width"])
                height = int(doc.attrib["height"])
                self.assertGreater(width, height * 3)
                self.assertIn("viewBox", doc.attrib)
                # Richer scientific diagrams can include sensor silhouettes,
                # overlapping-window illustrations and aligned context panels.
                # Validate the seven *actual process blocks* rather than
                # assuming every SVG rectangle is a process block.
                process = [
                    element for element in doc.iter()
                    if element.attrib.get("class") == "process-block"
                ]
                if process:
                    self.assertEqual(len(process), 7)
                    self.assertTrue(all(
                        len(group.findall("{http://www.w3.org/2000/svg}rect")) == 1
                        for group in process
                    ))
                    arrow_paths = [
                        element for element in doc.iter()
                        if element.tag == "{http://www.w3.org/2000/svg}path"
                        and element.attrib.get("marker-end") == "url(#small-arrow)"
                    ]
                    self.assertEqual(len(arrow_paths), 6)
                    self.assertEqual(len({
                        element.attrib.get("stroke-width")
                        for element in arrow_paths
                    }), 1)
                    panels = [
                        element for element in doc.iter()
                        if element.tag == "{http://www.w3.org/2000/svg}rect"
                        and element.attrib.get("x") == "24"
                        and element.attrib.get("width") == "1652"
                    ]
                    self.assertEqual(len(panels), 2)
                else:
                    # Earlier stage schematics: background plus 7 blocks.
                    self.assertEqual(
                        len(doc.findall(".//{http://www.w3.org/2000/svg}rect")), 8
                    )

                notebook_root = "notebooks" if n <= 25 else "presentation_notebooks"
                path = list((STUDY / notebook_root).glob(f"{name}_*.ipynb"))
                self.assertEqual(len(path), 1)
                notebook = json.loads(path[0].read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                first = notebook["cells"][0]
                self.assertEqual(first["cell_type"], "markdown")
                source = "".join(first["source"]) if isinstance(first["source"], list) else first["source"]
                self.assertIn(f"../images/{name}_workflow.svg", source)

    def test_original_evidence_archive_and_remaining_frozen_notebooks(self):
        # Stage 23 is formatted in the primary notebook with explicit
        # maintainer approval. Keep its original executed bytes archived.
        original_23 = STUDY / "notebooks/23_unified_matched_model_benchmark.ipynb"
        archived_23 = (STUDY / "experiments/frozen_notebook_evidence" /
                       original_23.name)
        self.assertTrue(original_23.is_file())
        self.assertTrue(archived_23.is_file())
        self.assertFalse(
            (STUDY / "presentation_notebooks" / original_23.name).exists()
        )

        # The original notebooks are now the sole viewing files for 24–25.
        for n in (24, 25):
            with self.subTest(stage=n):
                self.assertEqual(
                    len(list((STUDY / "notebooks").glob(f"{n:02d}_*.ipynb"))), 1
                )
                self.assertEqual(
                    len(list((STUDY / "presentation_notebooks").glob(f"{n:02d}_*.ipynb"))), 0
                )

        # Stages 26 and 27 retain their separate presentation copies.
        for n in (26, 27):
            with self.subTest(stage=n):
                self.assertEqual(
                    len(list((STUDY / "notebooks").glob(f"{n:02d}_*.ipynb"))), 1
                )
                self.assertEqual(
                    len(list((STUDY / "presentation_notebooks").glob(f"{n:02d}_*.ipynb"))), 1
                )


if __name__ == "__main__":
    unittest.main()
