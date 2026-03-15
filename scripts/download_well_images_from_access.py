#!/usr/bin/env python3
"""Download well images from get_observation_image_access results (JSON)."""
import json
import sys
from pathlib import Path

import requests

def main():
    if len(sys.argv) < 3:
        print("Usage: download_well_images_from_access.py <access_results.json> <output_dir>", file=sys.stderr)
        sys.exit(1)
    path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text())
    if isinstance(data, list):
        items = data
    else:
        items = data.get("items", data.get("image_access", [data]))
    saved = 0
    for item in items:
        well = item.get("well") or item.get("culture_id", "unknown")
        urls = item.get("download_urls") or item
        url = urls.get("standard_url") or urls.get("large_url")
        if not url:
            continue
        dest = out_dir / f"{well}.jpg"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"Saved {well} -> {dest}")
            saved += 1
        except Exception as e:
            print(f"Failed {well}: {e}", file=sys.stderr)
    print(f"Saved {saved} images to {out_dir}")

if __name__ == "__main__":
    main()
