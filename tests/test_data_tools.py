"""Offline tests for the Figshare inventory and safe-download contracts."""

import unittest

from scripts.figshare_inventory import extract_files, safe_filename


class TestFigshareInventory(unittest.TestCase):
    def test_safe_filename(self):
        self.assertEqual(safe_filename("subj04.mat"), "subj04.mat")
        for invalid in ("", ".", "..", "../data.mat", "x/y.mat", r"x\\y.mat"):
            with self.subTest(name=invalid):
                with self.assertRaises(ValueError):
                    safe_filename(invalid)

    def test_extract_files(self):
        record = {
            "files": [
                {"id": 10, "name": "processed.zip", "size": 128,
                 "download_url": "https://ndownloader.figshare.com/files/10",
                 "computed_md5": "abc"}
            ]
        }
        files = extract_files(record)
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["bytes"], 128)
        self.assertEqual(files[0]["md5"], "abc")

    def test_reject_bad_remote_metadata(self):
        for entry in (
            {"name": "../../bad.zip", "size": 12, "download_url": "https://example.org/ok"},
            {"name": "a.zip", "size": -1, "download_url": "https://example.org/ok"},
            {"name": "a.zip", "size": 10, "download_url": "http://example.org/unsafe"},
        ):
            with self.subTest(entry=entry):
                with self.assertRaises(ValueError):
                    extract_files({"files": [entry]})


if __name__ == "__main__":
    unittest.main()
