import os
import cv2
from pathlib import Path
from collections import defaultdict

root = Path("data/emergency_vehicle_dataset")

print("=" * 70)
print("EMERGENCY VEHICLE DATASET CURRENT STATUS AUDIT")
print("=" * 70)

print("\n1. RAW DIRECTORY INVENTORY (data/emergency_vehicle_dataset/raw):")
raw_counts = {}
if (root / "raw").exists():
    for d in sorted((root / "raw").iterdir()):
        if d.is_dir():
            files = list(d.glob("*"))
            raw_counts[d.name] = len(files)
            valid_imgs = [f for f in files if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
            print(f"  - raw/{d.name:<15}: {len(files):>4} files ({len(valid_imgs)} valid images)")
print(f"  Total raw images downloaded: {sum(raw_counts.values())}")

print("\n2. PROCESSED CROPS INVENTORY (data/emergency_vehicle_dataset/processed):")
split_class_counts = defaultdict(lambda: defaultdict(int))
total_crops = 0
corrupt_crops = 0

if (root / "processed").exists():
    for split in ["train", "val", "test"]:
        split_dir = root / "processed" / split
        if split_dir.exists():
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    crops = list(cls_dir.glob("*.jpg"))
                    cnt = len(crops)
                    split_class_counts[split][cls_dir.name] = cnt
                    total_crops += cnt
                    for c in crops:
                        try:
                            img = cv2.imread(str(c))
                            if img is None or img.size == 0:
                                corrupt_crops += 1
                        except Exception:
                            corrupt_crops += 1

classes = ["ambulance", "fire_brigade", "police", "normal"]
print(f"  {'Class':<15} | {'TRAIN':<8} | {'VAL':<8} | {'TEST':<8} | {'TOTAL':<8}")
print("  " + "-" * 50)
for c in classes:
    tr = split_class_counts["train"][c]
    vl = split_class_counts["val"][c]
    ts = split_class_counts["test"][c]
    tot = tr + vl + ts
    print(f"  {c.upper():<15} | {tr:<8} | {vl:<8} | {ts:<8} | {tot:<8}")
print("  " + "-" * 50)
tot_tr = sum(split_class_counts["train"].values())
tot_vl = sum(split_class_counts["val"].values())
tot_ts = sum(split_class_counts["test"].values())
print(f"  {'TOTAL':<15} | {tot_tr:<8} | {tot_vl:<8} | {tot_ts:<8} | {total_crops:<8}")

print(f"\n3. INTEGRITY AUDIT:")
print(f"  - Corrupt / Unreadable Crops: {corrupt_crops}")
print(f"  - All image files verified readable: {'YES' if corrupt_crops == 0 else 'NO'}")
print("=" * 70)
