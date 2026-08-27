import os
import json
import cv2
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "emergency_vehicle_dataset"
PROCESSED_DIR = DATASET_ROOT / "processed"
MANIFEST_FILE = DATASET_ROOT / "manifests" / "dataset_manifest.json"

def verify_dataset_integrity():
    print("=" * 75)
    print("STEP 1: FINAL DATASET INTEGRITY & DATA LEAKAGE AUDIT")
    print("=" * 75)

    if not PROCESSED_DIR.exists():
        print(f"Error: {PROCESSED_DIR} does not exist!")
        return False

    splits = ["train", "val", "test"]
    classes = ["ambulance", "fire_brigade", "police", "normal"]
    
    counts = defaultdict(lambda: defaultdict(int))
    corrupted = []
    all_files = []
    
    for split in splits:
        for cls in classes:
            c_dir = PROCESSED_DIR / split / cls
            if not c_dir.exists():
                print(f"Missing directory: {c_dir}")
                continue
            for img_p in c_dir.glob("*.jpg"):
                all_files.append(img_p)
                counts[split][cls] += 1
                try:
                    img = cv2.imread(str(img_p))
                    if img is None or img.size == 0:
                        corrupted.append(str(img_p))
                except Exception:
                    corrupted.append(str(img_p))

    print(f"\n1. ACTUAL FILE COUNTS BY SPLIT AND CLASS:")
    print(f"{'Class':<15} | {'TRAIN':<8} | {'VAL':<8} | {'TEST':<8} | {'TOTAL':<8}")
    print("-" * 55)
    for cls in classes:
        tr = counts["train"][cls]
        vl = counts["val"][cls]
        ts = counts["test"][cls]
        tot = tr + vl + ts
        print(f"{cls.upper():<15} | {tr:<8} | {vl:<8} | {ts:<8} | {tot:<8}")
    print("-" * 55)
    tot_tr = sum(counts["train"].values())
    tot_vl = sum(counts["val"].values())
    tot_ts = sum(counts["test"].values())
    grand_total = tot_tr + tot_vl + tot_ts
    print(f"{'TOTAL':<15} | {tot_tr:<8} | {tot_vl:<8} | {tot_ts:<8} | {grand_total:<8}")

    print(f"\n2. CORRUPTED / UNREADABLE IMAGES:")
    print(f"  Total corrupted images: {len(corrupted)}")

    print(f"\n3. SOURCE-LEVEL DATA LEAKAGE AUDIT (TRAIN vs VAL vs TEST):")
    source_to_splits = defaultdict(set)
    
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for entry in manifest:
            src = entry.get("source_image") or entry.get("source_url") or entry.get("filename")
            sp = entry.get("split")
            if src and sp:
                source_to_splits[src].add(sp)

        leakages = {src: sps for src, sps in source_to_splits.items() if len(sps) > 1}
        print(f"  Total distinct sources tracked: {len(source_to_splits)}")
        print(f"  Sources appearing in multiple splits (LEAKAGE): {len(leakages)}")
        if leakages:
            print("  [!] CRITICAL ERROR: Source leakage detected:")
            for src, sps in list(leakages.items())[:5]:
                print(f"      - {src} -> {sps}")
            return False
        else:
            print("  [OK] Zero source-level leakage. Every source belongs strictly to a single split.")
    else:
        print("  [!] Warning: Manifest file not found. Verifying filename prefixes...")
        # Check by filename prefix
        prefix_splits = defaultdict(set)
        for p in all_files:
            prefix = p.stem.split("_crop")[0].split("_full")[0]
            split = p.parent.parent.name
            prefix_splits[prefix].add(split)
        leakages = {pfx: sps for pfx, sps in prefix_splits.items() if len(sps) > 1}
        print(f"  Sources appearing in multiple splits: {len(leakages)}")
        if leakages:
            print("  [!] CRITICAL ERROR: Source leakage detected across splits!")
            return False
        print("  [OK] Zero source-level leakage.")

    print("\n" + "=" * 75)
    print("INTEGRITY AUDIT PASSED: READY FOR BASELINE TRAINING")
    print("=" * 75)
    return True

if __name__ == "__main__":
    verify_dataset_integrity()
