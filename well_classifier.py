"""
well_classifier.py
──────────────────
Single-cell / bead counting pipeline for 96-well plates.

Input  : Directory of well images in one of three formats:
           - OME-ZARR  : A1.ome.zarr, B4.ome.zarr  ...  (Squid default)
           - TIFF      : A1.tif, A1.tiff            ...
           - PNG/other : A1.png                     ...
         Well ID is parsed directly from the filename stem (e.g. "A1").

Output : Dictionary keyed by well ID, each value contains:
           well_id    : str   e.g. "A1"
           count      : int   0, 1, or 2  (2 means "2 or more")
           label      : str   "empty" | "single" | "multiple" | "uncertain"
           confidence : float 0.0 – 1.0
           flag       : bool  True → needs human review in Culture Monitor
           reason     : str | None  plain-English explanation when flagged

Dependencies (install once):
    pip install stardist tensorflow tifffile zarr ome-zarr scikit-image numpy
"""

import os
import re
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

# ── suppress noisy TF/Keras logs ──────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ── StarDist ───────────────────────────────────────────────────────────────────
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# ── image I/O ──────────────────────────────────────────────────────────────────
import tifffile
try:
    import zarr
    import ome_zarr.reader as ome_reader
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

from skimage.measure import regionprops


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  –  tune these for your specific bead / cell type
# ══════════════════════════════════════════════════════════════════════════════

# StarDist pretrained model name.
# '2D_versatile_fluo'  → fluorescence images (green beads, DAPI, etc.)
# '2D_versatile_he'    → H&E stained tissue (not relevant here)
STARDIST_MODEL = "2D_versatile_fluo"

# Expected bead diameter in PIXELS at your microscope magnification.
# Measure a few known single beads in Fiji/napari and set this range.
# Example: 28–48 µm beads at 10× with 0.65 µm/px ≈ 43–74 px diameter
# → area roughly π×(21)² to π×(37)² ≈ 1385–4300 px²
BEAD_AREA_MIN_PX = 500    # pixels² – below this → likely debris
BEAD_AREA_MAX_PX = 8000   # pixels² – above this → likely doublet

# Circularity threshold.  1.0 = perfect circle.
# Beads should be very round; real cells slightly less so.
CIRCULARITY_MIN = 0.70

# Confidence thresholds
CONFIDENCE_AUTOFLAG  = 0.80   # below this → always flag for human review
STARDIST_PROB_CUTOFF = 0.40   # StarDist detections below this prob are ignored
                               # (StarDist default is 0.5; lower = more sensitive)

# ══════════════════════════════════════════════════════════════════════════════
# WELL ID HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_WELL_PATTERN = re.compile(r"^([A-H])(\d{1,2})$", re.IGNORECASE)

def parse_well_id(filename: str) -> Optional[str]:
    """
    Extract well ID from a filename stem.
    Accepts:  'A1', 'a1', 'A01', 'B12', 'H9'
    Returns:  canonical uppercase string e.g. 'A1', 'B12'
    Returns None if the stem does not look like a well ID.
    """
    stem = Path(filename).stem
    # strip common suffixes like '_ch0', '_fluorescence', '_t0'
    stem = re.split(r"[_\-\s]", stem)[0]
    m = _WELL_PATTERN.match(stem)
    if m:
        row = m.group(1).upper()
        col = str(int(m.group(2)))   # remove leading zero: '01' → '1'
        return f"{row}{col}"
    return None


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path: Path) -> Optional[np.ndarray]:
    """
    Load a single 2-D grayscale fluorescence image.
    Handles: .ome.zarr directories, .tif/.tiff, .png, .jpg
    Returns a 2-D float32 numpy array, or None on failure.
    """
    path = Path(path)
    suffix = "".join(path.suffixes).lower()   # handles '.ome.zarr'

    # ── OME-ZARR directory ────────────────────────────────────────────────────
    if suffix.endswith(".zarr") or path.is_dir():
        if not HAS_ZARR:
            raise ImportError(
                "zarr and ome-zarr packages are required for .ome.zarr files.\n"
                "Run:  pip install zarr ome-zarr"
            )
        try:
            store = zarr.open(str(path), mode="r")
            # OME-ZARR layout: root → '0' (full resolution array)
            # Array shape is typically (T, C, Z, Y, X) or (C, Y, X) or (Y, X)
            arr = store["0"][:]
            arr = _squeeze_to_2d(arr)
            return arr.astype(np.float32)
        except Exception as e:
            print(f"  [WARN] Could not read {path}: {e}")
            return None

    # ── TIFF ──────────────────────────────────────────────────────────────────
    if suffix in (".tif", ".tiff", ".ome.tif", ".ome.tiff"):
        try:
            arr = tifffile.imread(str(path))
            arr = _squeeze_to_2d(arr)
            return arr.astype(np.float32)
        except Exception as e:
            print(f"  [WARN] Could not read {path}: {e}")
            return None

    # ── PNG / JPEG / other via scikit-image ───────────────────────────────────
    try:
        from skimage.io import imread
        from skimage.color import rgb2gray
        arr = imread(str(path))
        if arr.ndim == 3:
            arr = rgb2gray(arr)          # convert RGB → grayscale
        return arr.astype(np.float32)
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}")
        return None


def _squeeze_to_2d(arr: np.ndarray) -> np.ndarray:
    """
    Collapse a multi-dimensional microscopy array to (Y, X).
    Strategy: take index 0 along every extra dimension, then take the
    channel with the highest mean intensity (likely the fluorescence channel).
    """
    # Remove size-1 dimensions
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Assume first axis is channel; pick brightest channel
        channel_means = [arr[c].mean() for c in range(arr.shape[0])]
        return arr[int(np.argmax(channel_means))]
    if arr.ndim >= 4:
        # T, C, Z, Y, X  →  take t=0, z=0, pick brightest channel
        arr = arr[0]           # drop T
        while arr.ndim > 3:
            arr = arr[0]       # drop Z or other dims
        channel_means = [arr[c].mean() for c in range(arr.shape[0])]
        return arr[int(np.argmax(channel_means))]
    raise ValueError(f"Cannot reduce array of shape {arr.shape} to 2D")


# ══════════════════════════════════════════════════════════════════════════════
# MORPHOLOGY SCORING  (post-StarDist analysis)
# ══════════════════════════════════════════════════════════════════════════════

def _circularity(region) -> float:
    """4π × area / perimeter².  Returns 0 if perimeter is 0."""
    p = region.perimeter
    if p == 0:
        return 0.0
    return float(4 * np.pi * region.area / (p ** 2))


def _score_detection(region, image: np.ndarray, prob: float) -> Tuple[float, list]:
    """
    Score a single StarDist detection against expected bead morphology.

    Returns
    -------
    object_score : float  (0–1, higher = more bead-like)
    issues       : list of plain-English strings describing concerns
    """
    issues = []

    # ── 1. StarDist native probability ───────────────────────────────────────
    prob_score = float(np.clip(prob, 0, 1))

    # ── 2. Size score ─────────────────────────────────────────────────────────
    area = region.area
    if area < BEAD_AREA_MIN_PX:
        size_score = 0.25
        issues.append(
            f"Object area {area}px² is below minimum ({BEAD_AREA_MIN_PX}px²) — likely debris"
        )
    elif area > BEAD_AREA_MAX_PX:
        size_score = 0.30
        issues.append(
            f"Object area {area}px² exceeds maximum ({BEAD_AREA_MAX_PX}px²) — possible doublet"
        )
    else:
        # Smooth score: peaks at 1.0 in the middle of the expected range
        mid  = (BEAD_AREA_MIN_PX + BEAD_AREA_MAX_PX) / 2
        half = (BEAD_AREA_MAX_PX - BEAD_AREA_MIN_PX) / 2
        size_score = float(1.0 - abs(area - mid) / half * 0.3)   # min ~0.7

    # ── 3. Circularity score ──────────────────────────────────────────────────
    circ = _circularity(region)
    if circ < CIRCULARITY_MIN:
        circ_score = circ / CIRCULARITY_MIN   # scales 0→CIRCULARITY_MIN to 0→1
        issues.append(
            f"Circularity {circ:.2f} is below threshold {CIRCULARITY_MIN} "
            f"— irregular shape, may be two touching beads"
        )
    else:
        circ_score = 1.0

    # ── 4. Edge score ─────────────────────────────────────────────────────────
    h, w = image.shape
    bbox = region.bbox   # (min_row, min_col, max_row, max_col)
    at_edge = (bbox[0] == 0 or bbox[1] == 0 or
               bbox[2] == h or bbox[3] == w)
    if at_edge:
        edge_score = 0.50
        issues.append("Object touches image boundary — may be partially outside well")
    else:
        edge_score = 1.0

    # ── 5. Intensity score ────────────────────────────────────────────────────
    # We compare the object's mean intensity to the image's 99th percentile.
    # A genuine bead should be among the brightest objects.
    img_99 = float(np.percentile(image, 99))
    obj_mean = float(np.mean(image[region.coords[:, 0], region.coords[:, 1]]))
    if img_99 > 0:
        intensity_ratio = obj_mean / img_99
    else:
        intensity_ratio = 0.0

    if intensity_ratio < 0.15:
        intensity_score = 0.30
        issues.append(
            f"Object intensity ({obj_mean:.0f}) is very low relative to image "
            f"brightness — may be autofluorescence or debris"
        )
    else:
        intensity_score = min(1.0, intensity_ratio * 1.2)   # cap at 1.0

    # ── Combined object score (multiplicative — any bad signal pulls it down) ─
    object_score = (prob_score
                    * size_score
                    * circ_score
                    * edge_score
                    * intensity_score)

    return float(np.clip(object_score, 0, 1)), issues


# ══════════════════════════════════════════════════════════════════════════════
# WELL-LEVEL CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_well(
    well_id: str,
    image: np.ndarray,
    labels: np.ndarray,
    details: dict,
) -> dict:
    """
    Classify a single well given StarDist outputs.

    Parameters
    ----------
    well_id : str       e.g. "A1"
    image   : 2D array  normalised fluorescence image
    labels  : 2D array  StarDist label map (0=background, 1..N=objects)
    details : dict      StarDist details dict with keys 'prob', 'coord'

    Returns
    -------
    result dict with keys: well_id, count, label, confidence, flag, reason
    """
    probs = details.get("prob", np.array([]))
    n_raw = int(labels.max())   # number of StarDist detections

    # ── Zero detections ───────────────────────────────────────────────────────
    if n_raw == 0:
        # How empty does the image look?
        # Use the probability map if available, otherwise use image max
        prob_map = details.get("prob_map", None)
        if prob_map is not None:
            max_residual_prob = float(prob_map.max())
        else:
            # Heuristic: normalised image max as a proxy
            norm_max = float(image.max())
            max_residual_prob = min(norm_max * 0.5, 0.99)

        confidence = float(np.clip(1.0 - max_residual_prob, 0, 1))

        if confidence < CONFIDENCE_AUTOFLAG:
            return _result(well_id, 0, "uncertain", confidence, True,
                           f"No object detected but faint signal present "
                           f"(residual probability {max_residual_prob:.2f}). "
                           f"May be a very dim bead. Check fluorescence channel.")
        return _result(well_id, 0, "empty", confidence, False, None)

    # ── Score each detection ──────────────────────────────────────────────────
    regions = regionprops(labels, intensity_image=image)
    scored   = []   # list of (object_score, issues, region)

    for i, region in enumerate(regions):
        prob_i = float(probs[i]) if i < len(probs) else 0.5
        if prob_i < STARDIST_PROB_CUTOFF:
            continue   # ignore very weak detections
        score, issues = _score_detection(region, image, prob_i)
        scored.append((score, issues, region))

    # ── Re-count after filtering low-prob detections ──────────────────────────
    n_real = len(scored)

    if n_real == 0:
        # Everything was below prob cutoff → treat as empty but uncertain
        return _result(well_id, 0, "uncertain", 0.45, True,
                       f"StarDist found {n_raw} candidate(s) but all were below "
                       f"probability threshold {STARDIST_PROB_CUTOFF}. "
                       f"Likely debris or noise. Inspect image.")

    # ── Separate genuine beads from likely debris ─────────────────────────────
    # An object with object_score < 0.25 is almost certainly debris
    genuine = [(s, iss, r) for s, iss, r in scored if s >= 0.25]
    debris  = [(s, iss, r) for s, iss, r in scored if s <  0.25]

    n_genuine = len(genuine)

    # ── Build all_issues list for reason string ───────────────────────────────
    all_issues = []
    for s, iss, r in scored:
        all_issues.extend(iss)

    # ── 0 genuine objects (all detections were debris) ────────────────────────
    if n_genuine == 0:
        confidence = 0.40
        reason = (f"{n_raw} object(s) detected but all classified as debris "
                  f"(low circularity / size / intensity). "
                  + (" | ".join(all_issues) if all_issues else ""))
        return _result(well_id, 0, "uncertain", confidence, True, reason)

    # ── 1 genuine object ──────────────────────────────────────────────────────
    if n_genuine == 1:
        obj_score, obj_issues, obj_region = genuine[0]

        # Penalty if debris is also present
        debris_penalty = 0.10 * len(debris)
        well_conf = float(np.clip(obj_score - debris_penalty, 0, 1))

        # Check for hidden doublet: object significantly larger than expected
        area = obj_region.area
        doublet_suspicion = area > (BEAD_AREA_MAX_PX * 0.80)

        reason_parts = list(obj_issues)
        if debris:
            reason_parts.append(
                f"{len(debris)} additional low-score object(s) detected — "
                f"likely debris, ignored in count"
            )
        if doublet_suspicion:
            reason_parts.append(
                f"Object area ({area}px²) is close to doublet threshold — "
                f"verify it is a single bead"
            )
            well_conf = min(well_conf, 0.65)   # cap confidence for suspected doublets

        flag   = (well_conf < CONFIDENCE_AUTOFLAG) or doublet_suspicion or bool(debris)
        label  = "single" if well_conf >= CONFIDENCE_AUTOFLAG and not doublet_suspicion else "uncertain"
        reason = " | ".join(reason_parts) if reason_parts else None

        return _result(well_id, 1, label, well_conf, flag, reason)

    # ── 2+ genuine objects ────────────────────────────────────────────────────
    # Confidence = how sure we are that ALL detections are real beads
    scores    = [s for s, _, _ in genuine]
    well_conf = float(np.min(scores))   # conservative: weakest link

    reason_parts = [
        f"{n_genuine} bead-like objects detected — well must be excluded "
        f"from clonal work"
    ]
    if debris:
        reason_parts.append(
            f"Additional {len(debris)} low-score object(s) also detected — "
            f"likely debris"
        )
    if all_issues:
        reason_parts.extend(all_issues)

    return _result(well_id, min(n_genuine, 2), "multiple", well_conf, True,
                   " | ".join(reason_parts))


def _result(well_id, count, label, confidence, flag, reason) -> dict:
    """Construct a clean result dictionary."""
    return {
        "well_id":    well_id,
        "count":      count,
        "label":      label,                          # empty/single/multiple/uncertain
        "confidence": round(float(confidence), 3),
        "flag":       bool(flag),
        "reason":     reason,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(image_dir: str) -> Dict[str, dict]:
    """
    Run the full well-classification pipeline on a directory of well images.

    Parameters
    ----------
    image_dir : str
        Path to directory containing well images.
        Files must be named with the well ID, e.g.:
          A1.ome.zarr, B4.tif, C12.png, ...

    Returns
    -------
    results : dict
        { "A1": { well_id, count, label, confidence, flag, reason }, ... }
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # ── Load StarDist model (downloads automatically on first run) ────────────
    print(f"Loading StarDist model: {STARDIST_MODEL}")
    model = StarDist2D.from_pretrained(STARDIST_MODEL)
    print("Model ready.\n")

    # ── Discover well image files ─────────────────────────────────────────────
    # Collect both files and directories (for .ome.zarr)
    candidates = list(image_dir.iterdir())
    well_files = []
    for p in candidates:
        wid = parse_well_id(p.name)
        if wid is not None:
            well_files.append((wid, p))

    if not well_files:
        raise ValueError(
            f"No well images found in {image_dir}.\n"
            f"Expected files named like: A1.tif, B4.ome.zarr, C12.png"
        )

    well_files.sort(key=lambda x: x[0])   # sort by well ID
    print(f"Found {len(well_files)} well image(s) to process.\n")

    results = {}

    for well_id, path in well_files:
        print(f"  Processing {well_id} ({path.name}) ...", end=" ")

        # ── Load image ────────────────────────────────────────────────────────
        image = load_image(path)
        if image is None:
            print("SKIPPED (could not load image)")
            results[well_id] = _result(
                well_id, -1, "uncertain", 0.0, True,
                "Image could not be loaded. Check file format."
            )
            continue

        # ── Normalise to [0, 1] for StarDist ─────────────────────────────────
        # StarDist expects percentile normalisation
        image_norm = normalize(image, 1, 99.8)

        # ── Run StarDist ──────────────────────────────────────────────────────
        labels, details = model.predict_instances(
            image_norm,
            prob_thresh=STARDIST_PROB_CUTOFF,
            nms_thresh=0.4,        # non-maximum suppression overlap threshold
        )

        # ── Classify well ─────────────────────────────────────────────────────
        result = classify_well(well_id, image_norm, labels, details)
        results[well_id] = result

        # ── Console summary ───────────────────────────────────────────────────
        flag_str = "⚑ FLAGGED" if result["flag"] else "✓"
        print(
            f"{flag_str}  "
            f"count={result['count']}  "
            f"label={result['label']:<10}  "
            f"confidence={result['confidence']:.2f}"
        )

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("CLASSIFICATION SUMMARY")
    print("═" * 60)
    label_counts = {}
    flagged = []
    for wid, r in results.items():
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        if r["flag"]:
            flagged.append(wid)

    for label, n in sorted(label_counts.items()):
        print(f"  {label:<12} : {n} wells")
    print(f"\n  Total flagged for human review : {len(flagged)} wells")
    if flagged:
        print(f"  Flagged wells : {', '.join(flagged)}")
    print("═" * 60 + "\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CULTURE MONITOR UPLOAD HELPER
# ══════════════════════════════════════════════════════════════════════════════

def format_for_culture_monitor(results: Dict[str, dict]) -> str:
    """
    Format the results dict as a ready-to-paste prompt for the Monomer
    Culture Monitor MCP upload step.

    Paste the output of this function directly into your MCP client.
    """
    lines = [
        "I have classification results for this plate.",
        "For each well, I have a label (empty, single, multiple, uncertain)",
        "and a confidence score.",
        "",
        "First, use list_culture_statuses() to find the status IDs that",
        "correspond to my labels.",
        "Then, for each well:",
        "1. Update the culture status using update_culture_status.",
        "2. For any well flagged=True, add a comment explaining what the",
        "   algorithm detected and what the scientist should verify.",
        "",
        "Here are my results:",
    ]

    for wid, r in sorted(results.items()):
        conf_str = f"{r['confidence']:.2f}"
        line = f"  {wid}: {r['label']} ({conf_str})"
        if r["reason"]:
            line += f" — {r['reason']}"
        lines.append(line)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python well_classifier.py <path_to_image_directory>")
        print("")
        print("Example:")
        print("  python well_classifier.py ./data/plate_20240314/0_stitched/")
        sys.exit(1)

    image_directory = sys.argv[1]

    # Run the pipeline
    results = run_pipeline(image_directory)

    # Save results to JSON
    output_path = Path(image_directory) / "well_classification_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")

    # Print Culture Monitor upload prompt
    print("\n" + "─" * 60)
    print("CULTURE MONITOR UPLOAD PROMPT")
    print("─" * 60)
    print(format_for_culture_monitor(results))
