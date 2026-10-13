"""List Figshare article files, or download one explicitly selected file.

Metadata lookup is read-only. Dataset downloads require --download-file.
Files are stored locally; never add raw data to the public Git repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ARTICLE_ID = 30225739
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
CHUNK_SIZE = 1024 * 1024


def safe_filename(name: str) -> str:
    """Reject paths and traversal sequences supplied as remote filenames."""
    if not name or name in {".", ".."} or Path(name).name != name or "\\" in name:
        raise ValueError(f"Unsafe dataset filename: {name!r}")
    return name


def extract_files(record: dict) -> list[dict]:
    """Return validated file metadata without assuming a fixed archive layout."""
    files = record.get("files")
    if not isinstance(files, list):
        raise ValueError("Figshare metadata does not contain a files list.")
    result = []
    for entry in files:
        name = safe_filename(str(entry["name"]))
        size = int(entry["size"])
        if size < 0:
            raise ValueError(f"Negative file size for {name}")
        url = entry.get("download_url")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError(f"Missing secure download URL for {name}")
        result.append({
            "id": entry.get("id"), "name": name, "bytes": size,
            "download_url": url,
            "md5": entry.get("computed_md5") or entry.get("supplied_md5"),
        })
    return result


def read_metadata(api_url: str = API_URL) -> dict:
    """Request public Figshare metadata; authentication is not needed."""
    request = Request(api_url, headers={"User-Agent": "human-movement-intelligence/0.1"})
    with urlopen(request, timeout=45) as response:
        return json.load(response)


def download_file(file_info: dict, destination: Path, max_mb: float) -> Path:
    """Stream one chosen dataset file with size and optional MD5 verification."""
    destination.mkdir(parents=True, exist_ok=True)
    expected_size = int(file_info["bytes"])
    if expected_size > max_mb * 1024 * 1024:
        raise ValueError(
            f"Refusing {file_info['name']}: {expected_size} bytes exceeds "
            f"--max-mb {max_mb}. Increase the limit explicitly if appropriate."
        )
    target = destination / safe_filename(file_info["name"])
    if target.exists():
        raise FileExistsError(f"File already exists; refusing to overwrite: {target}")
    temporary = target.with_name(target.name + ".part")
    if temporary.exists():
        raise FileExistsError(f"Partial file already exists: {temporary}")

    md5 = hashlib.md5()  # Figshare-provided integrity digest, not authentication.
    count = 0
    try:
        request = Request(file_info["download_url"], headers={"User-Agent": "human-movement-intelligence/0.1"})
        with urlopen(request, timeout=90) as response, temporary.open("xb") as output:
            while True:
                block = response.read(CHUNK_SIZE)
                if not block:
                    break
                count += len(block)
                if count > expected_size or count > max_mb * 1024 * 1024:
                    raise ValueError("Download exceeded declared file size or safety limit.")
                output.write(block)
                md5.update(block)
        if count != expected_size:
            raise ValueError(f"Incomplete download: {count} of {expected_size} bytes.")
        expected_md5 = file_info.get("md5")
        if expected_md5 and md5.hexdigest().lower() != str(expected_md5).lower():
            raise ValueError("Downloaded file checksum differs from Figshare metadata.")
        os.replace(temporary, target)
        return target
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Optional local JSON inventory path (under outputs/).")
    parser.add_argument("--download-file", help="Exact filename from the returned Figshare listing.")
    parser.add_argument("--destination", type=Path, default=Path("data/raw/essex_femoral_neck_loading"))
    parser.add_argument("--max-mb", type=float, default=1024.0, help="Download safety cap (default 1024 MB).")
    args = parser.parse_args()
    if args.max_mb <= 0:
        parser.error("--max-mb must be positive.")

    try:
        record = read_metadata()
        files = extract_files(record)
        inventory = {
            "article_id": ARTICLE_ID, "article_doi": record.get("doi"),
            "version": record.get("version"), "published_date": record.get("published_date"),
            "modified_date": record.get("modified_date"), "licence": record.get("license"),
            "files": files,
        }
        print(f"Article {ARTICLE_ID} | version={inventory['version']} | files={len(files)}")
        for file in files:
            print(f"  {file['name']} | {file['bytes'] / 1048576:.2f} MiB | id={file['id']}")
        if args.manifest:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"Inventory saved: {args.manifest}")

        if args.download_file:
            selected = [item for item in files if item["name"] == args.download_file]
            if len(selected) != 1:
                raise ValueError("Filename not uniquely found. Copy the exact name from the inventory.")
            downloaded = download_file(selected[0], args.destination, args.max_mb)
            print(f"Downloaded and verified: {downloaded}")
    except (HTTPError, URLError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Data inventory error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
