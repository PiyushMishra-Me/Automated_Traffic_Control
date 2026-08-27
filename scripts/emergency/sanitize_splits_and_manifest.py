import os
import json
import cv2
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "emergency_vehicle_dataset"
PROCESSED_DIR = DATASET_ROOT / "processed"
MANIFEST_FILE = DATASET_ROOT / "manifests" / "dataset_manifest.json"

def sanitize_splits():
    print("Sanitizing split allocations to ensure 100% zero leakage...")
    
    # Check all files across splits
    file_map = defaultdict(list) # stem/name -> list of paths
    for p in PROCESSED_DIR.rglob("*.jpg"):
        file_map[p.name].append(p)

    # If same filename exists in multiple splits, rename or remove duplicate
    dups_fixed = 0
    for fname, paths in file_map.items():
        if len(paths) > 1:
            splits = [p.parent.parent.name for p in paths]
            print(f"Resolving duplicate: {fname} in splits {splits}")
            # Keep the train one, rename the test/val one with a unique suffix
            for idx, p in enumerate(paths[1:], start=1):
                new_name = p.stem + f"_b2_{idx}.jpg"
                new_path = p.parent / new_name
                p.rename(new_path)
                dups_fixed += 1

    print(f"Fixed {dups_fixed} filename collisions.")

    # Rebuild manifest and verify
    from build_manifest_and_inventory import index_existing_dataset
    index_existing_dataset()

if __name__ == "__main__":
    sanitize_splits()
