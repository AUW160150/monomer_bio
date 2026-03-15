"""
well_classifier.py
──────────────────
Single-cell / bead counting pipeline for 96-well plates.

Input  : Directory of well images (JPEG, PNG, TIFF, or OME-ZARR).
         Files must be named with the well ID as the stem:
           A1.jpg   B4.tif   C12.ome.zarr   E7.png

Output : JSON file at <image_dir>/well_classification_results.json
         { "A1": { well_id, count, label, confidence, flag, reason }, ... }

Labels:
  empty     – no beads detected
  single    – exactly one bead (good for cloning)
  multiple  – two or more beads (exclude from cloning)
  uncertain – ambiguous; needs human review

Dependencies:
    pip install scikit-image numpy tifffile
    (optional for OME-ZARR: pip install zarr ome-zarr)
"""

import os
import re
import json
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")

# ── image I/O ──────────────────────────────────────────────────────────────────
import tifffile
try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

from skimage.feature import blob_log
from skimage.filters import gaussian, median as skimage_median
from skimage.morphology import disk
from skimage.measure import regionprops, label as skimage_label


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  –  tune these for your specific bead / microscope setup
# ══════════════════════════════════════════════════════════════════════════════

# Blob detection sigma range (in pixels).
# sigma ≈ bead_radius / sqrt(2)
# For ~16–24 px diameter beads: radius 8–12 px → sigma 5.7–8.5
# Widen the range to be robust across magnifications.
BLOB_MIN_SIGMA = 3       # smallest detectable blob radius ≈ 4 px diameter
BLOB_MAX_SIGMA = 15      # largest detectable blob radius ≈ 42 px diameter
BLOB_NUM_SIGMA = 10      # steps between min and max sigma
BLOB_THRESHOLD = 0.29    # LoG response threshold — raise to reduce false positives,
                          # lower to catch dimmer beads (calibrated on test data)

# Secondary (sensitive) blob pass when the primary finds nothing.
# Catches dim beads missed at the normal threshold, but also plate artifacts
# like scratches. Any hit here → uncertain + flagged for human review.
BLOB_THRESHOLD_SENSITIVE = 0.10
# Minimum intensity ratio (intensity/p99) for a secondary-pass blob to be
# considered a candidate. Calibrated: real dim beads ≥ 1.5, background noise < 1.5.
MIN_INTENSITY_RATIO_SENSITIVE = 1.5

# Wells below this confidence are auto-flagged for human review.
CONFIDENCE_AUTOFLAG = 0.80

# Minimum blob intensity relative to image 99th percentile.
# Blobs dimmer than this are considered noise / debris.
MIN_INTENSITY_RATIO = 0.15


# ══════════════════════════════════════════════════════════════════════════════
# WELL ID HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_WELL_PATTERN = re.compile(r"^([A-H])(\d{1,2})$", re.IGNORECASE)


def parse_well_id(filename: str) -> Optional[str]:
    """Return canonical well ID (e.g. 'A1', 'B12') from filename, or None."""
    stem = Path(filename).stem
    stem = re.split(r"[_\-\s]", stem)[0]
    m = _WELL_PATTERN.match(stem)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2))}"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path: Path) -> Optional[np.ndarray]:
    """
    Load a well image as a 2-D float32 array normalised to [0, 1].
    Handles: .ome.zarr directories, .tif/.tiff, .jpg/.jpeg, .png
    """
    path = Path(path)
    suffix = "".join(path.suffixes).lower()

    # ── OME-ZARR ─────────────────────────────────────────────────────────────
    if suffix.endswith(".zarr") or path.is_dir():
        if not HAS_ZARR:
            raise ImportError("Install zarr and ome-zarr for .ome.zarr support.")
        try:
            store = zarr.open(str(path), mode="r")
            arr = store["0"][:]
            arr = _squeeze_to_2d(arr)
        except Exception as e:
            print(f"  [WARN] Could not read {path}: {e}")
            return None

    # ── TIFF ──────────────────────────────────────────────────────────────────
    elif suffix in (".tif", ".tiff", ".ome.tif", ".ome.tiff"):
        try:
            arr = tifffile.imread(str(path))
            arr = _squeeze_to_2d(arr)
        except Exception as e:
            print(f"  [WARN] Could not read {path}: {e}")
            return None

    # ── JPEG / PNG via PIL ─────────────────────────────────────────────────────
    else:
        try:
            from PIL import Image
            pil = Image.open(path).convert("L")   # grayscale
            arr = np.array(pil, dtype=np.float32)
        except Exception as e:
            print(f"  [WARN] Could not read {path}: {e}")
            return None

    arr = arr.astype(np.float32)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return arr


def _squeeze_to_2d(arr: np.ndarray) -> np.ndarray:
    """Collapse (T, C, Z, Y, X) → (Y, X) by picking the brightest channel."""
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        best_c = int(np.argmax([arr[c].mean() for c in range(arr.shape[0])]))
        return arr[best_c]
    if arr.ndim >= 4:
        arr = arr[0]
        while arr.ndim > 3:
            arr = arr[0]
        best_c = int(np.argmax([arr[c].mean() for c in range(arr.shape[0])]))
        return arr[best_c]
    raise ValueError(f"Cannot reduce array of shape {arr.shape} to 2D")


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_well(well_id: str, image: np.ndarray) -> dict:
    """
    Classify a single well image.

    Returns a dict with keys:
      well_id, count, label, confidence, flag, reason
    """
    # ── Detect blobs ──────────────────────────────────────────────────────────
    # Apply a 3×3 median filter before blob detection to suppress hot pixels
    # (single-pixel spikes from dead pixels or JPEG artifacts). Real beads
    # have spatial extent and survive the median filter intact.
    denoised = skimage_median(image, disk(1))
    blobs = blob_log(
        denoised,
        min_sigma=BLOB_MIN_SIGMA,
        max_sigma=BLOB_MAX_SIGMA,
        num_sigma=BLOB_NUM_SIGMA,
        threshold=BLOB_THRESHOLD,
    )

    img_99 = float(np.percentile(image, 99))

    # ── Filter: remove blobs whose centre pixel is too dim ───────────────────
    genuine = []
    for blob in blobs:
        y, x, sigma = blob
        cy, cx = int(round(y)), int(round(x))
        cy = np.clip(cy, 0, image.shape[0] - 1)
        cx = np.clip(cx, 0, image.shape[1] - 1)
        intensity = float(image[cy, cx])
        if img_99 > 0 and (intensity / img_99) >= MIN_INTENSITY_RATIO:
            genuine.append(blob)

    n = len(genuine)

    # ── 0 beads ───────────────────────────────────────────────────────────────
    if n == 0:
        # If blob_log found objects that failed the intensity filter, flag as uncertain
        if len(blobs) > 0:
            return _result(well_id, 0, "uncertain", 0.55, True,
                           f"{len(blobs)} faint object(s) detected but below "
                           f"intensity threshold — possible dim beads. Inspect image.")

        # Secondary sensitive pass: catch dim beads missed at normal threshold.
        # Filter by high intensity ratio to avoid noise; any hit still gets
        # flagged for human review (could be dim bead, scratch, or debris).
        faint_blobs = blob_log(
            denoised,
            min_sigma=BLOB_MIN_SIGMA,
            max_sigma=BLOB_MAX_SIGMA,
            num_sigma=BLOB_NUM_SIGMA,
            threshold=BLOB_THRESHOLD_SENSITIVE,
        )
        faint_candidates = []
        for blob in faint_blobs:
            fy, fx, _ = blob
            cy = int(np.clip(round(fy), 0, image.shape[0] - 1))
            cx = int(np.clip(round(fx), 0, image.shape[1] - 1))
            if img_99 > 0 and (float(image[cy, cx]) / img_99) >= MIN_INTENSITY_RATIO_SENSITIVE:
                faint_candidates.append(blob)
        if faint_candidates:
            n_faint = len(faint_candidates)
            return _result(well_id, 0, "uncertain", 0.60, True,
                           f"{n_faint} faint object(s) detected at sensitive threshold "
                           f"(ratio ≥ {MIN_INTENSITY_RATIO_SENSITIVE}×p99) — "
                           f"possible dim bead, scratch, or debris. Inspect image.")
        # Truly empty: use peak-to-background ratio on a Gaussian-blurred image.
        # Blur (sigma=5) removes JPEG compression artifacts so we measure real
        # spatial structure (beads) rather than compression noise.
        # Calibrated on known-empty wells: peak_ratio ~1.7–2.0 → conf ~0.87–0.95
        blurred = gaussian(image, sigma=5)
        p50 = float(np.percentile(blurred, 50))
        p99 = float(np.percentile(blurred, 99))
        peak_ratio = (p99 / p50) if p50 > 1e-6 else 1.0
        confidence = float(np.clip(1.20 - (peak_ratio - 1.0) * 0.35, 0.35, 0.95))
        reason = (f"No beads detected but image has local structure "
                  f"(blurred peak/bg ratio: {peak_ratio:.2f}) — "
                  f"may contain a dim bead. Inspect image."
                  if confidence < CONFIDENCE_AUTOFLAG else None)
        return _result(well_id, 0, "empty", confidence, confidence < CONFIDENCE_AUTOFLAG, reason)

    # ── 1 bead ────────────────────────────────────────────────────────────────
    if n == 1:
        blob = genuine[0]
        y, x, sigma = blob
        radius = sigma * 1.414

        # Edge proximity check
        h, w = image.shape
        at_edge = (y - radius < 5 or x - radius < 5 or
                   y + radius > h - 5 or x + radius > w - 5)

        # Intensity check
        cy, cx = int(round(y)), int(round(x))
        cy = np.clip(cy, 0, h - 1)
        cx = np.clip(cx, 0, w - 1)
        intensity_ratio = float(image[cy, cx]) / img_99 if img_99 > 0 else 0.0

        # Single-blob detections are always flagged for human review.
        # blob_log cannot distinguish a real bead from plate scratches or debris.
        # A scientist must confirm visually before accepting as a true single.
        issues = ["1 object detected — verify it is a bead and not a scratch or debris"]
        confidence = 0.70

        if at_edge:
            confidence -= 0.15
            issues.append("object touches image boundary — may be partially outside well")
        if intensity_ratio < 0.40:
            confidence -= 0.10
            issues.append(f"object intensity is relatively low ({intensity_ratio:.2f} × p99)")

        confidence = max(confidence, 0.35)
        reason = " | ".join(issues)
        return _result(well_id, 1, "uncertain", confidence, True, reason)

    # ── 2+ beads ──────────────────────────────────────────────────────────────
    # Confidence = how consistent the detections look (all bright, none at edge)
    h, w = image.shape
    intensities = []
    for blob in genuine:
        y, x, sigma = blob
        cy, cx = int(np.clip(round(y), 0, h-1)), int(np.clip(round(x), 0, w-1))
        intensities.append(float(image[cy, cx]) / img_99 if img_99 > 0 else 0.0)

    confidence = float(np.clip(np.mean(intensities), 0, 1))
    reason = (f"{n} beads detected — well must be excluded from clonal work. "
              f"Mean bead intensity: {np.mean(intensities):.2f} × p99")
    return _result(well_id, min(n, 2), "multiple", confidence, True, reason)


def _result(well_id, count, label, confidence, flag, reason) -> dict:
    return {
        "well_id":    well_id,
        "count":      count,
        "label":      label,
        "confidence": round(float(confidence), 3),
        "flag":       bool(flag),
        "reason":     reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(image_dir: str) -> Dict[str, dict]:
    """
    Classify all well images in a directory.

    Expects files named like: A1.jpg, B4.tif, C12.ome.zarr
    Returns dict keyed by well ID.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    well_files = []
    for p in sorted(image_dir.iterdir()):
        wid = parse_well_id(p.name)
        if wid is not None:
            well_files.append((wid, p))

    if not well_files:
        raise ValueError(
            f"No well images found in {image_dir}.\n"
            f"Expected files named like: A1.jpg, B4.tif, C12.ome.zarr"
        )

    print(f"Found {len(well_files)} well image(s) to classify.\n")
    results = {}

    for well_id, path in sorted(well_files):
        print(f"  {well_id} ({path.name}) ...", end=" ", flush=True)

        image = load_image(path)
        if image is None:
            print("SKIPPED (load error)")
            results[well_id] = _result(well_id, -1, "uncertain", 0.0, True,
                                       "Image could not be loaded.")
            continue

        result = classify_well(well_id, image)
        results[well_id] = result

        flag_str = "FLAGGED" if result["flag"] else "ok"
        print(f"[{flag_str}]  label={result['label']:<10}  "
              f"count={result['count']}  conf={result['confidence']:.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    counts = {}
    flagged = []
    for wid, r in results.items():
        counts[r["label"]] = counts.get(r["label"], 0) + 1
        if r["flag"]:
            flagged.append(wid)
    for lbl, n in sorted(counts.items()):
        print(f"  {lbl:<12} : {n}")
    print(f"\n  Flagged for review : {len(flagged)}")
    if flagged:
        print(f"  Wells             : {', '.join(sorted(flagged))}")
    print("═" * 60 + "\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python well_classifier.py <image_directory>")
        sys.exit(1)

    image_directory = sys.argv[1]
    results = run_pipeline(image_directory)

    output_path = Path(image_directory) / "well_classification_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")
