#!/usr/bin/env python3
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)  # allow ~1 trillion pixels
import cv2
import math
import numpy as np
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    raise SystemExit("tqdm not found. Install it with: pip install tqdm")

# =========================
# USER-CHANGEABLE PARAMETERS
# =========================
ROOT_DIR = os.path.expanduser("~/Tulane Meat ML data")   # source folder (top-level scan only)
OUTPUT_DIR = os.path.join(ROOT_DIR, "tiles")             # where tiles go
TILE_SIZE = 1024                                         # tile size (pixels)
# Empty/white detection
WHITE_THRESHOLD = 230                                    # per-channel threshold (0..255)
WHITE_RATIO_LIMIT = 0.70                                 # fraction of pixels >= threshold to call it empty
# Blurriness detection (variance of Laplacian)
BLURRY_THRESHOLD = 5.0
# JPEG save params (as "lossless" as JPEG can be)
JPEG_QUALITY = 100                                       # 100 = best
# =========================

# Output subfolders
EMPTY_DIR  = os.path.join(OUTPUT_DIR, "empty")  # formerly "white"
BLURRY_DIR = os.path.join(OUTPUT_DIR, "blurry")
TISSUE_DIR = os.path.join(OUTPUT_DIR, "tissue")

# Create output dirs
for d in (OUTPUT_DIR, EMPTY_DIR, BLURRY_DIR, TISSUE_DIR):
    os.makedirs(d, exist_ok=True)

JPEG_PARAMS = [
    cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY,
    cv2.IMWRITE_JPEG_PROGRESSIVE, 0  # keep baseline JPEG (widely compatible)
]

def iter_top_level_jpegs(folder: Path):
    exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
    for p in folder.iterdir():
        if p.is_file() and p.suffix in exts:
            yield p

def classify_tile(tile_bgr: np.ndarray) -> str:
    """
    Returns one of: 'empty', 'blurry', 'tissue'
    """
    # Empty/white test
    # A pixel is "white" if ALL channels >= WHITE_THRESHOLD
    white_mask = np.all(tile_bgr >= WHITE_THRESHOLD, axis=2)
    white_ratio = float(np.sum(white_mask)) / (tile_bgr.shape[0] * tile_bgr.shape[1])
    if white_ratio > WHITE_RATIO_LIMIT:
        return "empty"

    # Blurry test (variance of Laplacian on grayscale)
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < BLURRY_THRESHOLD:
        return "blurry"

    return "tissue"

def tile_image(img_bgr: np.ndarray, tile_size: int):
    """
    Generator yielding (row_idx, col_idx, y_start, x_start, tile_bgr)
    Uses the same "snap back" strategy at edges as your original code (no padding).
    """
    h, w = img_bgr.shape[:2]
    if h < tile_size or w < tile_size:
        # Too small to produce any full tiles with this edge policy
        return

    x_steps = math.ceil(w / tile_size)
    y_steps = math.ceil(h / tile_size)

    for ri in range(y_steps):
        for ci in range(x_steps):
            x_start = ci * tile_size
            y_start = ri * tile_size

            # Snap back at right/bottom edges so we always take a full 1024x1024 crop
            if x_start + tile_size > w:
                x_start = w - tile_size
            if y_start + tile_size > h:
                y_start = h - tile_size

            # Safe guard (in case of rounding quirks)
            if x_start < 0 or y_start < 0:
                continue

            tile = img_bgr[y_start:y_start + tile_size, x_start:x_start + tile_size]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                # Shouldn't happen with the snap-back logic; skip if it does
                continue

            yield (ri, ci, y_start, x_start, tile)

def save_tile(dest_root: str, cls: str, slide_stem: str, r: int, c: int, tile_bgr: np.ndarray):
    out_dir = {
        "empty": EMPTY_DIR,
        "blurry": BLURRY_DIR,
        "tissue": TISSUE_DIR
    }[cls]
    out_name = f"{slide_stem}_r{r}_c{c}.jpg"
    out_path = os.path.join(out_dir, out_name)
    cv2.imwrite(out_path, tile_bgr, JPEG_PARAMS)

def main():
    src = Path(ROOT_DIR)
    files = list(iter_top_level_jpegs(src))
    if not files:
        print(f"No JPG files found at top level of: {ROOT_DIR}")
        return

    print(f"Found {len(files)} image(s). Tiling into {OUTPUT_DIR}")
    for img_path in tqdm(files, desc="Slides", unit="img"):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            tqdm.write(f"[WARN] Could not read image: {img_path.name} — skipping.")
            continue

        h, w = img.shape[:2]
        if h < TILE_SIZE or w < TILE_SIZE:
            tqdm.write(f"[WARN] {img_path.name} is smaller than {TILE_SIZE}x{TILE_SIZE} — skipping.")
            continue

        # Count tiles to size progress bar
        x_steps = math.ceil(w / TILE_SIZE)
        y_steps = math.ceil(h / TILE_SIZE)
        total_tiles = x_steps * y_steps

        pbar = tqdm(total=total_tiles, leave=False, desc=f"Tiling {img_path.name}", unit="tile")
        seen = set()  # avoid duplicates due to snap-back making identical (r,c) appear more than once at far edge

        for r, c, ys, xs, tile in tile_image(img, TILE_SIZE):
            key = (ys, xs)  # start coords uniquely identify a tile
            if key in seen:
                continue
            seen.add(key)

            cls = classify_tile(tile)
            save_tile(OUTPUT_DIR, cls, img_path.stem, r, c, tile)
            pbar.update(1)
        pbar.close()

    print("✅ Done. Tiles written to:")
    print(f"   {EMPTY_DIR}\n   {BLURRY_DIR}\n   {TISSUE_DIR}")

if __name__ == "__main__":
    main()
