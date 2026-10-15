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
                self.assertEqual(len(doc.findall(".//{http://www.w3.org/2000/svg}rect")), 8)

                notebook_root = "notebooks" if n <= 22 else "presentation_notebooks"
                path = list((STUDY / notebook_root).glob(f"{name}_*.ipynb"))
                self.assertEqual(len(path), 1)
                notebook = json.loads(path[0].read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                first = notebook["cells"][0]
                self.assertEqual(first["cell_type"], "markdown")
                source = "".join(first["source"]) if isinstance(first["source"], list) else first["source"]
                self.assertIn(f"../images/{name}_workflow.svg", source)

    def test_original_frozen_notebooks_and_approved_protocol_stay_in_place(self):
        for n in range(23, 28):
            with self.subTest(stage=n):
                self.assertEqual(
                    len(list((STUDY / "notebooks").glob(f"{n:02d}_*.ipynb"))), 1
                )
                self.assertEqual(
                    len(list((STUDY / "presentation_notebooks").glob(f"{n:02d}_*.ipynb"))), 1
                )


if __name__ == "__main__":
    unittest.main()
