#!/usr/bin/env python3
"""
Cut large pseudocolored images into smaller tiles along original stitch boundaries.

Preserves 10% overlap (204px) between tiles for potential re-stitching.
Target: 5×5 grid (scales down to 2×2 minimum if needed).
Limits: ≤15MB file size AND ≤13112px in either dimension.

Original tiles: 2048×2048px with 204px overlap (stride = 1844px)
"""

from pathlib import Path
from typing import Tuple, List, Optional
import io

import numpy as np
from PIL import Image

# Disable decompression bomb check for legitimately large stitched images
Image.MAX_IMAGE_PIXELS = None

# ========================= CONFIGURATION ========================= #

PSEUDOCOLORED_FOLDER = Path(r"G:\Shared drives\Instapath Shared Drive\Data for ML\Meat_Luci_Tulane\Pseudocolored")

# Size limits
MAX_FILE_SIZE_MB = 15
MAX_DIMENSION_PX = 13112

# Tile properties (derived from 2×2 example: 3892×3892px)
TILE_SIZE = 2048        # Original tile dimension
OVERLAP = 204           # Overlap between adjacent tiles
STRIDE = 1844           # Effective stride (TILE_SIZE - OVERLAP)

# Grid size preferences (try in order)
PREFERRED_GRIDS = [5, 4, 3, 2]  # 5×5 down to 2×2

# Output options
SAVE_TILES = True
DRY_RUN = False         # If True, only analyze without saving

# ======================================================================== #
def get_jpeg_save_params(img: Image.Image) -> dict:
    """
    Return JPEG save params copied from the opened image if available.
    For non-JPEG sources, provide reasonable defaults.
    Works with derived images (crops) — no 'quality=\"keep\"'.
    """
    params = {
        "format": "JPEG",
        "quality": 90,        # default fallback
        "subsampling": 0,     # 4:4:4 preserves detail better for masks/overlays
        "optimize": True,
        "progressive": False,
    }
    try:
        if getattr(img, "format", None) == "JPEG" or img.format == "JPG":
            info = img.info or {}
            if "qtables" in info:
                params["qtables"] = info["qtables"]
            if "subsampling" in info:
                params["subsampling"] = info["subsampling"]
            if "progressive" in info:
                params["progressive"] = info["progressive"]
            if "quality" in info and isinstance(info["quality"], int):
                params["quality"] = info["quality"]
    except Exception:
        pass
    return params


def calculate_grid_dimensions(image_width: int, image_height: int) -> Tuple[int, int]:
    """
    Calculate number of tiles in original stitched image.
    Returns (cols, rows) of original tiles.
    """
    # First tile starts at 0, subsequent tiles at stride intervals
    # Last tile ends at image edge
    cols = 1 + max(0, int(np.ceil((image_width - TILE_SIZE) / STRIDE)))
    rows = 1 + max(0, int(np.ceil((image_height - TILE_SIZE) / STRIDE)))
    return cols, rows


def calculate_piece_size(grid_size: int) -> Tuple[int, int]:
    """
    Calculate pixel dimensions of a piece containing grid_size × grid_size tiles.
    Returns (width, height) in pixels including overlaps.
    """
    # Each piece spans grid_size tiles with (grid_size-1) overlaps between them
    dimension = TILE_SIZE + (grid_size - 1) * STRIDE
    return dimension, dimension


def estimate_jpeg_size(img: Image.Image, sample_crop_size: int = 2048) -> float:
    """
    Estimate JPEG size by encoding a sample crop using explicit JPEG params.
    """
    width, height = img.size

    crop_size = min(sample_crop_size, width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    sample = img.crop((left, top, left + crop_size, top + crop_size))

    save_params = get_jpeg_save_params(img)

    buffer = io.BytesIO()
    try:
        sample.save(buffer, **save_params)
    except Exception:
        # Fallback if qtables/subsampling from source cause issues
        buffer = io.BytesIO()
        sample.save(buffer, format="JPEG", quality=90, subsampling=0, optimize=True)

    sample_bytes = buffer.tell()
    sample_pixels = crop_size * crop_size
    total_pixels = width * height
    estimated_bytes = (sample_bytes / sample_pixels) * total_pixels
    return estimated_bytes / (1024 * 1024)


def find_optimal_grid(total_cols: int, total_rows: int, img: Image.Image) -> Optional[int]:
    """
    Find largest grid size that fits within limits.
    Returns grid_size (e.g., 5 for 5×5), or None if even 2×2 doesn't fit.
    """
    for grid_size in PREFERRED_GRIDS:
        # Check if we can make at least one piece of this size
        if grid_size > total_cols or grid_size > total_rows:
            continue
        
        piece_w, piece_h = calculate_piece_size(grid_size)
        
        # Check dimension limit
        if piece_w > MAX_DIMENSION_PX or piece_h > MAX_DIMENSION_PX:
            continue
        
        # Check file size by creating a test crop
        test_crop = img.crop((0, 0, min(piece_w, img.width), min(piece_h, img.height)))
        test_size_mb = estimate_jpeg_size(test_crop, sample_crop_size=1024)
        
        if test_size_mb <= MAX_FILE_SIZE_MB:
            return grid_size
    
    return None


def extract_tile_piece(img: Image.Image, start_col: int, start_row: int, 
                       grid_size: int, total_cols: int, total_rows: int
                       ) -> Tuple[Image.Image, bool, int, int]:
    """
    Extract a piece containing grid_size×grid_size tiles starting at (start_col, start_row).
    Returns (cropped_image, is_partial, actual_cols, actual_rows).
    """
    # Calculate pixel coordinates
    x_start = start_col * STRIDE
    y_start = start_row * STRIDE
    
    # Calculate how many tiles actually fit
    actual_cols = min(grid_size, total_cols - start_col)
    actual_rows = min(grid_size, total_rows - start_row)
    
    # Calculate piece dimensions
    piece_w = TILE_SIZE + (actual_cols - 1) * STRIDE
    piece_h = TILE_SIZE + (actual_rows - 1) * STRIDE
    
    # Crop (ensure we don't exceed image bounds)
    x_end = min(x_start + piece_w, img.width)
    y_end = min(y_start + piece_h, img.height)
    
    cropped = img.crop((x_start, y_start, x_end, y_end))
    
    # Check if partial
    is_partial = (actual_cols < grid_size) or (actual_rows < grid_size) or \
                 (cropped.width < piece_w) or (cropped.height < piece_h)
    
    return cropped, is_partial, actual_cols, actual_rows


def process_image(image_path: Path):
    """Process a single large image."""
    print(f"\n{'='*70}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*70}")
    
    # Load image
    img = Image.open(image_path)
    width, height = img.size
    file_size_mb = image_path.stat().st_size / (1024 * 1024)
    
    print(f"  Dimensions: {width}×{height} px")
    print(f"  File size: {file_size_mb:.2f} MB")
    
    # Check if processing needed
    if width <= MAX_DIMENSION_PX and height <= MAX_DIMENSION_PX and file_size_mb <= MAX_FILE_SIZE_MB:
        print(f"  ✓ Image already within limits, skipping")
        return
    
    # Calculate original tile grid
    total_cols, total_rows = calculate_grid_dimensions(width, height)
    print(f"  Original grid: {total_cols}×{total_rows} tiles (stride={STRIDE}px, overlap={OVERLAP}px)")
    
    # Find optimal grid size
    print(f"  Finding optimal grid size...")
    grid_size = find_optimal_grid(total_cols, total_rows, img)
    
    if grid_size is None:
        print(f"  ❌ ERROR: Cannot fit image within limits even with 2×2 grid!")
        return
    
    piece_w, piece_h = calculate_piece_size(grid_size)
    print(f"  ✓ Using {grid_size}×{grid_size} tile grid per piece ({piece_w}×{piece_h} px)")
    
    # Calculate how many pieces we'll create
    pieces_cols = int(np.ceil(total_cols / grid_size))
    pieces_rows = int(np.ceil(total_rows / grid_size))
    total_pieces = pieces_cols * pieces_rows
    print(f"  Will create: {pieces_cols}×{pieces_rows} = {total_pieces} piece(s)")
    
    if DRY_RUN:
        print(f"  🏃 DRY RUN - not saving")
        return
    
    # Extract and save pieces
    saved_count = 0
    partial_count = 0
    base_name = image_path.stem
    
    for row_idx in range(pieces_rows):
        for col_idx in range(pieces_cols):
            start_col = col_idx * grid_size
            start_row = row_idx * grid_size
            
            piece_img, is_partial, actual_cols, actual_rows = extract_tile_piece(
                img, start_col, start_row, grid_size, total_cols, total_rows
            )
            
            # Generate filename
            piece_name = f"{base_name}_grid_{grid_size}x{grid_size}_r{row_idx}_c{col_idx}.jpg"
            piece_path = image_path.parent / piece_name
            
            # Save with original quality
            if SAVE_TILES:
                save_params = get_jpeg_save_params(img)  # use params from the ORIGINAL opened image
                try:
                    piece_img.save(piece_path, **save_params)
                except Exception:
                    # robust fallback
                    piece_img.save(piece_path, format="JPEG", quality=90, subsampling=0, optimize=True)
                    print("Estimated image quality of 90, not original quality")

                piece_size_mb = piece_path.stat().st_size / (1024 * 1024)
                
                status = "PARTIAL" if is_partial else "full"
                print(f"    [{row_idx},{col_idx}] {status}: {piece_img.width}×{piece_img.height} px, "
                      f"{piece_size_mb:.2f} MB", end="")
                
                if is_partial:
                    expected_w, expected_h = calculate_piece_size(grid_size)
                    partial_w = expected_w - piece_img.width
                    partial_h = expected_h - piece_img.height
                    print(f" (short by {partial_w}×{partial_h} px, "
                          f"covers {actual_cols}×{actual_rows} tiles)")
                    partial_count += 1
                else:
                    print()
                
                saved_count += 1
    
    print(f"\n  ✅ Saved {saved_count} piece(s)")
    if partial_count > 0:
        print(f"  ⚠️  {partial_count} partial piece(s) at edges")


def main():
    """Main processing loop."""
    print("=" * 70)
    print("Large Image Tile Cutter")
    print("=" * 70)
    print(f"Folder: {PSEUDOCOLORED_FOLDER}")
    print(f"Limits: ≤{MAX_FILE_SIZE_MB}MB, ≤{MAX_DIMENSION_PX}px")
    print(f"Tile config: {TILE_SIZE}px tiles, {OVERLAP}px overlap, {STRIDE}px stride")
    print(f"Target grids: {' > '.join([f'{g}×{g}' for g in PREFERRED_GRIDS])}")
    print(f"Dry run: {DRY_RUN}")
    
    if not PSEUDOCOLORED_FOLDER.exists():
        raise SystemExit(f"❌ Folder not found: {PSEUDOCOLORED_FOLDER}")
    
    # Find all JPG images
    images = list(PSEUDOCOLORED_FOLDER.glob("*.jpg"))
    if not images:
        print(f"\n⚠️  No JPG files found")
        return
    
    print(f"\n🔍 Found {len(images)} image(s)")
    
    # Filter to only large images
    large_images = []
    for img_path in images:
        # Skip already-split images
        if "_grid_" in img_path.stem:
            continue
        
        try:
            with Image.open(img_path) as img:
                w, h = img.size
            size_mb = img_path.stat().st_size / (1024 * 1024)
            
            if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX or size_mb > MAX_FILE_SIZE_MB:
                large_images.append(img_path)
        except Exception as e:
            print(f"⚠️  Could not check {img_path.name}: {e}")
    
    if not large_images:
        print(f"\n✅ No images exceed limits - nothing to process!")
        return
    
    print(f"📏 {len(large_images)} image(s) exceed limits and will be processed")
    
    # Process each large image
    for img_path in large_images:
        try:
            process_image(img_path)
        except Exception as e:
            print(f"\n❌ FAILED {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()