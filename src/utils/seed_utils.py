"""
Seed Utilities for Reproducible Randomization

Provides centralized seed management from seed.csv file for all scripts in the pipeline.
"""

from pathlib import Path
from typing import Optional


def load_seed(seed_file: str = "seed.csv", default: int = 865) -> int:
    """
    Load seed value from seed.csv file
    
    Args:
        seed_file: Path to seed file (default: seed.csv in project root)
        default: Default seed if file not found (default: 865)
        
    Returns:
        Seed value as integer
        
    Raises:
        ValueError: If seed file contains invalid data
    """
    seed_path = Path(seed_file)
    
    if not seed_path.exists():
        print(f"⚠️  Seed file not found: {seed_path}, using default seed: {default}")
        return default
    
    try:
        with open(seed_path, 'r') as f:
            content = f.read().strip()
            seed = int(content)
            print(f"✓ Loaded seed from {seed_file}: {seed}")
            return seed
    except ValueError as e:
        raise ValueError(f"Invalid seed value in {seed_file}: {content}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading seed file {seed_file}: {e}") from e


def get_project_seed() -> int:
    """
    Convenience function to get seed from project root seed.csv
    
    Returns:
        Seed value from seed.csv or default (865)
    """
    return load_seed("seed.csv", default=865)
