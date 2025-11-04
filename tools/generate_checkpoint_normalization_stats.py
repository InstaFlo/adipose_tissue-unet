#!/usr/bin/env python3
"""
Generate Normalization Statistics for Existing Checkpoints

This script scans the checkpoints directory and creates normalization_stats.json files
for existing checkpoint directories by computing statistics from their corresponding
datasets in ~/Data_for_ML/Meat_Luci_Tulane/_build_{timestamp}/

Usage:
    python generate_checkpoint_normalization_stats.py [--overwrite]
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, List
import glob

import numpy as np
import cv2


def extract_timestamp_from_checkpoint_dir(checkpoint_name: str) -> Optional[str]:
    """Extract timestamp from checkpoint directory name like '20251031_124942_adipose_sybreosin_1024_finetune'"""
    match = re.search(r'^(\d{8}_\d{6})_', checkpoint_name)
    if match:
        return match.group(1)
    return None


def compute_mean_std_from_images(image_paths: List[Path], max_n: int = None) -> Tuple[float, float]:
    """
    Compute mean and standard deviation from a list of image paths.
    Same logic as in train_adipose_unet_2.py
    
    Args:
        image_paths: List of paths to training images
        max_n: Maximum number of images to process (None = all)
        
    Returns:
        Tuple of (mean, std)
    """
    print(f"Computing normalization statistics from {len(image_paths)} images...")
    
    vals = []
    processed = 0
    
    for i, img_path in enumerate(image_paths):
        if max_n and i >= max_n:
            break
            
        try:
            # Load image as grayscale (same as training script)
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                continue
                
            vals.append(img.reshape(-1))
            processed += 1
            
            # Progress indicator
            if processed % 100 == 0:
                print(f"  Processed {processed}/{len(image_paths)} images...")
                
        except Exception as e:
            print(f"Warning: Error processing {img_path}: {e}")
            continue
    
    if not vals:
        raise ValueError("No valid images found for computing statistics")
        
    # Concatenate all pixel values and compute statistics
    vals = np.concatenate(vals)
    mean = float(vals.mean())
    std = float(vals.std() + 1e-10)  # Add small epsilon like in training script
    
    print(f"✓ Computed statistics from {processed} images: mean={mean:.4f}, std={std:.4f}")
    return mean, std


def find_dataset_for_timestamp(timestamp: str, base_data_root: Path) -> Optional[Path]:
    """
    Find the dataset directory corresponding to a timestamp.
    
    Args:
        timestamp: Build timestamp like '20251031_124942'
        base_data_root: Base data directory path
        
    Returns:
        Path to dataset directory or None if not found
    """
    # Try exact match first
    exact_build_dir = base_data_root / f"_build_{timestamp}"
    if exact_build_dir.exists() and (exact_build_dir / "dataset" / "train" / "images").exists():
        return exact_build_dir
    
    # If exact match not found, look for closest build directories
    build_pattern = str(base_data_root / "_build_*")
    build_dirs = glob.glob(build_pattern)
    
    if not build_dirs:
        return None
    
    # Parse timestamps and find the closest one before or equal to our target
    target_dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    closest_build = None
    closest_dt = None
    
    for build_path in build_dirs:
        build_dir = Path(build_path)
        build_timestamp_match = re.search(r'_build_(\d{8}_\d{6})$', build_dir.name)
        
        if build_timestamp_match:
            build_timestamp = build_timestamp_match.group(1)
            try:
                build_dt = datetime.strptime(build_timestamp, "%Y%m%d_%H%M%S")
                
                # Find the most recent build that's <= target timestamp
                if build_dt <= target_dt:
                    if closest_dt is None or build_dt > closest_dt:
                        closest_dt = build_dt
                        closest_build = build_dir
                        
            except ValueError:
                continue
    
    if closest_build and (closest_build / "dataset" / "train" / "images").exists():
        print(f"  Using closest dataset: {closest_build} for checkpoint timestamp {timestamp}")
        return closest_build
        
    return None


def create_normalization_stats(checkpoint_dir: Path, dataset_dir: Path) -> Dict:
    """
    Create normalization statistics dictionary for a checkpoint.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        dataset_dir: Path to dataset build directory
        
    Returns:
        Dictionary with normalization statistics
    """
    train_images_dir = dataset_dir / "dataset" / "train" / "images"
    
    # Get all training images
    train_image_paths = sorted(train_images_dir.glob("*.jpg"))
    
    if not train_image_paths:
        raise ValueError(f"No training images found in {train_images_dir}")
    
    # Compute statistics
    mean, std = compute_mean_std_from_images(train_image_paths)
    
    # Extract build timestamp from dataset path
    build_timestamp = extract_timestamp_from_checkpoint_dir(dataset_dir.name)
    if not build_timestamp:
        build_timestamp = extract_timestamp_from_checkpoint_dir(checkpoint_dir.name)
    
    # Create statistics dictionary (matching format from train_adipose_unet_2.py)
    stats = {
        "mean": mean,
        "std": std,
        "normalization_method": "zscore",  # Default assumption for existing models
        "dataset_path": str(dataset_dir),
        "num_training_images": len(train_image_paths),
        "build_timestamp": build_timestamp,
        "timestamp_saved": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "percentile_low": None,  # Assume zscore normalization
        "percentile_high": None,
        "batch_size": None,  # Unknown for existing checkpoints
        "image_size": [1024, 1024],  # Standard size for this project
        "augmentation": "moderate",  # Standard augmentation for this project
        "preprocessing_applied": "stain_normalization"
    }
    
    return stats


def process_checkpoint(checkpoint_dir: Path, base_data_root: Path, overwrite: bool = False) -> bool:
    """
    Process a single checkpoint directory and create normalization_stats.json if needed.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        base_data_root: Base data directory path
        overwrite: Whether to overwrite existing normalization_stats.json
        
    Returns:
        True if processed successfully, False otherwise
    """
    normalization_stats_path = checkpoint_dir / "normalization_stats.json"
    
    # Skip if file already exists and not overwriting
    if normalization_stats_path.exists() and not overwrite:
        print(f"⏭️  Skipping {checkpoint_dir.name} - normalization_stats.json already exists")
        return True
    
    # Extract timestamp from checkpoint directory name
    timestamp = extract_timestamp_from_checkpoint_dir(checkpoint_dir.name)
    if not timestamp:
        print(f"❌ Could not extract timestamp from {checkpoint_dir.name}")
        return False
    
    print(f"\n📊 Processing checkpoint: {checkpoint_dir.name}")
    print(f"   Extracted timestamp: {timestamp}")
    
    # Find corresponding dataset
    dataset_dir = find_dataset_for_timestamp(timestamp, base_data_root)
    if not dataset_dir:
        print(f"❌ Could not find corresponding dataset for timestamp {timestamp}")
        return False
    
    print(f"   Using dataset: {dataset_dir}")
    
    try:
        # Create normalization statistics
        stats = create_normalization_stats(checkpoint_dir, dataset_dir)
        
        # Save to JSON file
        with open(normalization_stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Created normalization_stats.json")
        print(f"   Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"   Training images: {stats['num_training_images']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing checkpoint {checkpoint_dir.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate normalization statistics for existing checkpoints"
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing normalization_stats.json files'
    )
    parser.add_argument(
        '--checkpoints-dir',
        type=str,
        default='checkpoints',
        help='Path to checkpoints directory (default: checkpoints)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='~/Data_for_ML/Meat_Luci_Tulane',
        help='Base data directory (default: ~/Data_for_ML/Meat_Luci_Tulane)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    checkpoints_dir = Path(args.checkpoints_dir)
    base_data_root = Path(args.data_root).expanduser()
    
    if not checkpoints_dir.exists():
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        return 1
    
    if not base_data_root.exists():
        print(f"Error: Base data directory not found: {base_data_root}")
        return 1
    
    print(f"🔍 Scanning checkpoints in: {checkpoints_dir}")
    print(f"📂 Base data directory: {base_data_root}")
    print(f"🔄 Overwrite existing files: {args.overwrite}")
    print("=" * 80)
    
    # Find all timestamped checkpoint directories
    checkpoint_dirs = []
    for item in checkpoints_dir.iterdir():
        if item.is_dir() and extract_timestamp_from_checkpoint_dir(item.name):
            checkpoint_dirs.append(item)
    
    if not checkpoint_dirs:
        print("No timestamped checkpoint directories found.")
        return 0
    
    # Sort by timestamp (most recent first)
    checkpoint_dirs.sort(key=lambda x: extract_timestamp_from_checkpoint_dir(x.name), reverse=True)
    
    print(f"Found {len(checkpoint_dirs)} timestamped checkpoint directories:")
    for cp_dir in checkpoint_dirs:
        timestamp = extract_timestamp_from_checkpoint_dir(cp_dir.name)
        status = "✅" if (cp_dir / "normalization_stats.json").exists() else "⏳"
        print(f"  {status} {cp_dir.name} (timestamp: {timestamp})")
    
    # Process each checkpoint
    print("\n" + "=" * 80)
    success_count = 0
    
    for checkpoint_dir in checkpoint_dirs:
        if process_checkpoint(checkpoint_dir, base_data_root, args.overwrite):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"✅ Successfully processed {success_count}/{len(checkpoint_dirs)} checkpoints")
    
    if success_count < len(checkpoint_dirs):
        print(f"❌ Failed to process {len(checkpoint_dirs) - success_count} checkpoints")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
