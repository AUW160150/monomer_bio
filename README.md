# Well Classifier — Single-Cell / Bead Counting Pipeline

An AI-assisted pipeline for classifying 96-well plate images as **empty**, **single**, **multiple**, or **uncertain**, integrated with the [Monomer Culture Monitor](https://cloud-staging.monomerbio.com/) via MCP.

---

## What it does

1. **Fetches images** from the Monomer Cloud MCP (presigned S3 URLs per well)
2. **Classifies each well** using a blob detection algorithm (Laplacian of Gaussian) — no GPU or deep learning required
3. **Uploads results** back to Culture Monitor: sets a status label per well and posts a comment explaining the classifier's reasoning
4. **Flags uncertain wells** so a scientist can open Culture Monitor, inspect the image, and make a final QC decision

### Labels

| Label | Meaning |
|---|---|
| `empty` | No beads detected |
| `single` | Exactly one bead — good for clonal work *(not yet assigned; all single detections go to `uncertain` for human confirmation)* |
| `multiple` | Two or more beads — exclude from clonal selection |
| `uncertain` | Ambiguous — needs human review (dim bead, scratch, debris, low confidence) |

---

## How it works

### Detection

Uses **`skimage.feature.blob_log`** (Laplacian of Gaussian) — a classical computer vision algorithm that detects bright circular spots on dark backgrounds. No TensorFlow or GPU needed.

**Two-stage detection:**
- **Primary pass** (`threshold=0.29`): catches bright, clearly-visible beads
- **Secondary sensitive pass** (`threshold=0.10`, `intensity ≥ 1.5 × p99`): catches dim beads missed by the primary pass — these are flagged as `uncertain` since they could also be plate scratches or debris

**Hot-pixel suppression:** a 3×3 median filter is applied before detection to remove single-pixel JPEG compression artifacts.

### Confidence scoring

- **Multiple beads**: confidence = mean bead intensity relative to background
- **Single blob**: always `uncertain` at 0.70 — blob_log cannot distinguish a real bead from a scratch or debris without human confirmation
- **Empty** (0 blobs, primary + secondary): confidence based on peak-to-background ratio of the Gaussian-blurred image. A uniform background (autofluorescence) scores high; local structure (possible dim bead) scores lower

Wells below `CONFIDENCE_AUTOFLAG = 0.80` are automatically flagged.

---

## Setup

### 1. Python environment

```bash
python -m venv monomer
source monomer/bin/activate      # Windows: monomer\Scripts\activate
pip install -r requirements.txt
```

Additional dependencies for the classifier (not in `requirements.txt`):

```bash
pip install scikit-image numpy tifffile pillow
# Optional (for OME-ZARR support):
pip install zarr ome-zarr
```

### 2. Connect to the Monomer Cloud MCP

The classifier uses Claude Code (with the Monomer MCP) to fetch image URLs and upload results. You do **not** need to handle OAuth in Python — Claude Code handles all MCP calls.

```bash
claude mcp add --scope user --transport http monomer-cloud https://backend-staging.monomerbio.com/mcp
```

Then start a Claude Code session:

```bash
claude
```

Type `/mcp` and authenticate with the `monomer-cloud` server.

---

## Running the pipeline

The workflow has two parts: Claude Code handles MCP calls; Python handles image processing.

### Step 1 — Build a manifest (Claude Code)

In your Claude Code session, ask:

```
Build a manifest for plate <PLATE_BARCODE> and save it to data/<PLATE_BARCODE>/manifest.json
```

Claude Code will call `get_observation_image_access` for each well and produce a manifest:

```json
{
  "plate_barcode": "ET_FL_Run_2",
  "wells": {
    "H1": { "culture_id": "CLTR1...", "download_url": "https://..." },
    "H2": { "culture_id": "CLTR1...", "download_url": "https://..." }
  }
}
```

> **Note:** Presigned URLs expire after 1 hour. Run the pipeline promptly after building the manifest.

### Step 2 — Download images and classify (Python)

```bash
python pipeline.py --manifest data/<PLATE_BARCODE>/manifest.json
```

This will:
- Download each well image from the presigned URL
- Run `well_classifier.py` on the image directory
- Save results to `data/<PLATE_BARCODE>/well_classification_results.json`
- Save an enriched file with culture IDs attached: `well_classification_enriched.json`

### Step 3 — Upload results to Culture Monitor (Claude Code)

Ask Claude Code:

```
Upload the results in data/<PLATE_BARCODE>/well_classification_enriched.json to Culture Monitor.
For each well, set the culture status to match the label and add a comment for any uncertain or flagged well.
```

Claude Code will call `list_culture_statuses`, `update_culture_status`, and `add_comment` for each well.

### Running a single well (quick test)

You can also run Claude Code directly on a single well without building a manifest:

```
Fetch the image for well G2 on plate CerealDelusion_Run1_A, classify it with well_classifier.py, and upload the result.
```

---

## Output format

`well_classification_results.json`:

```json
{
  "G2": {
    "well_id": "G2",
    "count": 0,
    "label": "uncertain",
    "confidence": 0.6,
    "flag": true,
    "reason": "1 faint object(s) detected at sensitive threshold — possible dim bead, scratch, or debris. Inspect image."
  }
}
```

| Field | Type | Description |
|---|---|---|
| `well_id` | str | e.g. `"G2"` |
| `count` | int | 0, 1, or 2 (2 means 2 or more) |
| `label` | str | `empty` / `uncertain` / `multiple` |
| `confidence` | float | 0.0 – 1.0 |
| `flag` | bool | `true` → needs human review in Culture Monitor |
| `reason` | str | Plain-English explanation when flagged |

---

## Configuration

Edit the constants at the top of `well_classifier.py`:

| Constant | Default | Description |
|---|---|---|
| `BLOB_MIN_SIGMA` | `3` | Smallest detectable blob (radius ≈ 4 px) |
| `BLOB_MAX_SIGMA` | `15` | Largest detectable blob (radius ≈ 21 px) |
| `BLOB_THRESHOLD` | `0.29` | Primary LoG threshold — raise to reduce false positives |
| `BLOB_THRESHOLD_SENSITIVE` | `0.10` | Secondary pass threshold for dim beads |
| `MIN_INTENSITY_RATIO_SENSITIVE` | `1.5` | Min intensity/p99 for secondary-pass candidates |
| `CONFIDENCE_AUTOFLAG` | `0.80` | Wells below this are auto-flagged |
| `MIN_INTENSITY_RATIO` | `0.15` | Min intensity/p99 for primary-pass blobs |

**Calibrating thresholds for your microscope:** run on a known single-bead well and check the `count` and `confidence`. If real beads are being missed, lower `BLOB_THRESHOLD`. If noise is being detected, raise it or increase `MIN_INTENSITY_RATIO`.

---

## Supported image formats

| Format | Notes |
|---|---|
| `.jpg` / `.jpeg` | Default format from Monomer sprite images |
| `.png` | Supported |
| `.tif` / `.tiff` | Supported via `tifffile` |
| `.ome.tiff` | Supported via `tifffile` |
| `.ome.zarr` | Supported if `zarr` and `ome-zarr` are installed |

Image files must be named with the well ID as the stem: `A1.jpg`, `B4.tif`, `G2.png`.

---

## Known limitations

- **Single-bead wells are always flagged** — the algorithm cannot distinguish a real bead from plate scratches or small debris particles. A human must confirm.
- **JPEG compression** can inflate spatial statistics, affecting empty-well confidence scoring. The Gaussian blur pre-processing mitigates this but does not eliminate it.
- **Presigned URLs expire** after 1 hour — do not delay between building the manifest and running the pipeline.
- **Calibration is microscope-specific** — thresholds (`BLOB_THRESHOLD`, `MIN_INTENSITY_RATIO_SENSITIVE`) were calibrated on Monomer/Squid fluorescence images and may need adjustment for different setups.
