#!/usr/bin/env python3
"""
Analyze which source images were used across all dataset builds.
Identifies images that appear in all test sets.
"""

import os
import re
from pathlib import Path
from collections import Counter, defaultdict

# Build directory
BUILD_BASE = Path("/home/luci/Data_for_ML/Meat_Luci_Tulane")

# Pattern to extract source image from tile filename
# Tiles are named like: "ImageName_grid_5x5_rX_cY_rZ_cW.jpg"
# We want everything before the final _rZ_cW pattern
TILE_PATTERN = re.compile(r'(.+)_r\d+_c\d+\.jpg$')

def extract_source_image(tile_filename):
    """Extract the source image name from a tile filename."""
    match = TILE_PATTERN.match(tile_filename)
    if match:
        return match.group(1)
    return None

def get_test_images(build_dir):
    """Get all test image filenames from a build directory."""
    test_images_dir = build_dir / "dataset" / "test" / "images"
    if not test_images_dir.exists():
        return []
    
    return [f.name for f in test_images_dir.iterdir() if f.is_file() and f.suffix == '.jpg']

def main():
    # Find all build directories
    build_dirs = sorted([d for d in BUILD_BASE.iterdir() 
                        if d.is_dir() and d.name.startswith('_build_')])
    
    print(f"Found {len(build_dirs)} build directories\n")
    print("=" * 80)
    
    # Store source images for each build
    builds_data = {}
    all_source_images = set()
    
    for build_dir in build_dirs:
        build_name = build_dir.name
        print(f"\nAnalyzing {build_name}...")
        
        # Get test images
        test_tiles = get_test_images(build_dir)
        
        if not test_tiles:
            print(f"  WARNING: No test images found!")
            continue
        
        # Extract source images
        source_images = set()
        for tile in test_tiles:
            source = extract_source_image(tile)
            if source:
                source_images.add(source)
        
        builds_data[build_name] = {
            'tiles': test_tiles,
            'sources': source_images,
            'tile_count': len(test_tiles),
            'source_count': len(source_images)
        }
        
        all_source_images.update(source_images)
        
        print(f"  Test tiles: {len(test_tiles)}")
        print(f"  Source images: {len(source_images)}")
    
    print("\n" + "=" * 80)
    print("\nSUMMARY")
    print("=" * 80)
    
    # Count how many builds each source image appears in
    source_image_counts = Counter()
    for build_data in builds_data.values():
        for source in build_data['sources']:
            source_image_counts[source] += 1
    
    total_builds = len(builds_data)
    
    # Find images in ALL builds
    images_in_all = [img for img, count in source_image_counts.items() 
                     if count == total_builds]
    
    print(f"\nTotal unique source images across all builds: {len(all_source_images)}")
    print(f"Source images appearing in ALL {total_builds} builds: {len(images_in_all)}")
    
    if images_in_all:
        print("\n" + "=" * 80)
        print("SOURCE IMAGES IN ALL BUILDS:")
        print("=" * 80)
        for img in sorted(images_in_all):
            print(f"  • {img}")
    
    # Show distribution
    print("\n" + "=" * 80)
    print("DISTRIBUTION ACROSS BUILDS:")
    print("=" * 80)
    distribution = Counter(source_image_counts.values())
    for build_count in sorted(distribution.keys(), reverse=True):
        img_count = distribution[build_count]
        print(f"  {build_count}/{total_builds} builds: {img_count} source images")
    
    # Show images NOT in all builds (if any)
    images_not_in_all = [img for img, count in source_image_counts.items() 
                         if count < total_builds]
    
    if images_not_in_all:
        print("\n" + "=" * 80)
        print(f"SOURCE IMAGES NOT IN ALL BUILDS ({len(images_not_in_all)} total):")
        print("=" * 80)
        for img in sorted(images_not_in_all):
            count = source_image_counts[img]
            missing_from = total_builds - count
            print(f"  • {img} (in {count}/{total_builds} builds, missing from {missing_from})")
    
    # Detailed build-by-build breakdown
    print("\n" + "=" * 80)
    print("BUILD-BY-BUILD BREAKDOWN:")
    print("=" * 80)
    for build_name in sorted(builds_data.keys()):
        data = builds_data[build_name]
        print(f"\n{build_name}:")
        print(f"  Tiles: {data['tile_count']}")
        print(f"  Source images: {data['source_count']}")
        print(f"  Sources: {', '.join(sorted(list(data['sources'])[:5]))}{'...' if len(data['sources']) > 5 else ''}")

if __name__ == "__main__":
    main()
