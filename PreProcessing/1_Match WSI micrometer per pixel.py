# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:56:26 2025

@author: flori
"""

import os
import pandas as pd
import cv2
import math

# --- CONFIG ---
folder = r"C:\Users\flori\Desktop\Instapath\Engineering\ML\Working Images\original WSI"
csv_file = os.path.join(folder, "WSI_Scale_Sizes.csv")  # replace with actual csv filename
output_folder = os.path.dirname(folder)
os.makedirs(output_folder, exist_ok=True)

# --- Load CSV ---
df = pd.read_csv(csv_file, header=None, skiprows=1)

# Get target micrometer/pixel (max of col 6 and col 11)
col6_max = df[5].max()   # 6th column (0-indexed = 5)
col11_max = df[10].max() # 11th column (0-indexed = 10)
target = max(col6_max, col11_max)

print(f"Target micrometer/pixel value: {target}")

# --- Iterate over rows ---
for idx, row in df.iterrows():
    prefix = str(row[0]).strip()
    auto_mpp = float(row[5]) if not pd.isna(row[5]) else None
    he_mpp   = float(row[10]) if not pd.isna(row[10]) else None

    # Build possible image names
    candidates = []
    if auto_mpp:
        candidates.append((f"{prefix}_Auto", auto_mpp))
    if he_mpp:
        candidates.append((f"{prefix}_HE", he_mpp))

    for base_name, mpp in candidates:
        # Find matching image file
        for fname in os.listdir(folder):
            if fname.startswith(base_name) and fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                fpath = os.path.join(folder, fname)

                # Load image
                img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"⚠️ Could not read {fname}")
                    continue

                h, w = img.shape[:2]

                # Compute scaling factor
                scale = mpp / target
                new_w = max(1, math.ceil(w * scale))
                new_h = max(1, math.ceil(h * scale))

                # Resample image with high-quality interpolation
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

                # Save as lossless PNG (no compression)
                out_name = os.path.splitext(fname)[0] + ".png"
                out_path = os.path.join(output_folder, out_name)
                cv2.imwrite(out_path, resized, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

                print(f"Resampled {fname}: {w}x{h} → {new_w}x{new_h}, saved as {out_name}")

print(f"✅ Done. All resampled images saved as PNGs in: {output_folder}")
