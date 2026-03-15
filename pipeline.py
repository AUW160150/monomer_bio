"""
pipeline.py
───────────
Download all well images for a plate and run the well classifier.

This script handles the image-fetching and classification steps.
Claude Code (or any MCP client) handles the MCP calls to:
  1. Provide the image manifest (fetched via get_observation_image_access)
  2. Upload results after this script runs (update_culture_status + add_comment)

Usage:
    python pipeline.py --manifest data/<barcode>/manifest.json

The manifest is a JSON file produced by Claude Code (via MCP):
    {
      "plate_barcode": "CerealDelusion_Run1_A",
      "wells": {
        "A1": {
          "culture_id": "CLTR1...",
          "download_url": "https://..."
        },
        ...
      }
    }

After running, results are saved to:
    data/<barcode>/well_classification_results.json

Then ask Claude Code to upload the results to Culture Monitor.
"""

import json
import sys
import argparse
from pathlib import Path

import requests
from well_classifier import run_pipeline


def download_images(manifest: dict, out_dir: Path) -> int:
    """
    Download well images listed in the manifest to out_dir.
    Skips wells that already have a downloaded image.
    Returns the number of images downloaded.
    """
    wells = manifest.get("wells", {})
    downloaded = 0

    for well_id, info in wells.items():
        url = info.get("download_url") or info.get("standard_url")
        if not url:
            print(f"  [SKIP] {well_id} — no download URL in manifest")
            continue

        # Check for any existing image for this well
        existing = list(out_dir.glob(f"{well_id}.*"))
        if existing:
            print(f"  [SKIP] {well_id} — already downloaded ({existing[0].name})")
            continue

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            # Determine extension from Content-Type
            ct = resp.headers.get("content-type", "")
            if "jpeg" in ct or "jpg" in ct:
                ext = ".jpg"
            elif "png" in ct:
                ext = ".png"
            elif "tiff" in ct:
                ext = ".tiff"
            else:
                ext = ".jpg"   # default for Monomer sprite images

            out_path = out_dir / f"{well_id}{ext}"
            out_path.write_bytes(resp.content)
            print(f"  [OK]   {well_id} → {out_path.name} ({len(resp.content) // 1024} KB)")
            downloaded += 1

        except Exception as e:
            print(f"  [FAIL] {well_id} — {e}")

    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download well images and classify them for a plate."
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to manifest JSON file (produced by Claude Code via MCP)"
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip image download (use already-downloaded images in the output directory)"
    )
    args = parser.parse_args()

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    plate_barcode = manifest.get("plate_barcode", "unknown_plate")
    print(f"\nPlate: {plate_barcode}")
    print(f"Wells in manifest: {len(manifest.get('wells', {}))}\n")

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Download images ────────────────────────────────────────────────────────
    if not args.skip_download:
        print("── Downloading images ───────────────────────────────────────")
        n = download_images(manifest, out_dir)
        print(f"\n{n} image(s) downloaded.\n")

    # ── Classify ──────────────────────────────────────────────────────────────
    print("── Classifying wells ────────────────────────────────────────")
    results = run_pipeline(str(out_dir))

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = out_dir / "well_classification_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    # ── Attach culture IDs to results (for MCP upload) ────────────────────────
    enriched = {}
    wells_manifest = manifest.get("wells", {})
    for well_id, result in results.items():
        enriched[well_id] = {
            **result,
            "culture_id": wells_manifest.get(well_id, {}).get("culture_id"),
        }

    enriched_path = out_dir / "well_classification_enriched.json"
    with open(enriched_path, "w") as f:
        json.dump(enriched, f, indent=2)

    print(f"\nEnriched results (with culture IDs) saved to: {enriched_path}")
    print("\n── Next step ────────────────────────────────────────────────")
    print("Ask Claude Code to upload results to Culture Monitor:")
    print(f'  "Upload {results_path} to Culture Monitor for plate {plate_barcode}"')


if __name__ == "__main__":
    main()
