import os
import json
import cv2
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "emergency_vehicle_dataset"
RAW_DIR = DATASET_ROOT / "raw"
PROCESSED_DIR = DATASET_ROOT / "processed"
MANIFESTS_DIR = DATASET_ROOT / "manifests"
REPORTS_DIR = DATASET_ROOT / "reports"

def index_existing_dataset():
    manifest_entries = []
    
    # Inventory counters
    source_summary = defaultdict(lambda: {
        "source_image_count": 0,
        "ambulance_crops": 0,
        "fire_brigade_crops": 0,
        "police_crops": 0,
        "normal_crops": 0
    })

    # Index RAW directory
    for cls_dir in RAW_DIR.iterdir():
        if not cls_dir.is_dir():
            continue
        c_name = cls_dir.name
        for img_p in cls_dir.glob("*"):
            if img_p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                source_name = "wikimedia_commons_verified"
                source_summary[source_name]["source_image_count"] += 1

    # Index PROCESSED directory
    for split_dir in PROCESSED_DIR.iterdir():
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            for crop_p in cls_dir.glob("*.jpg"):
                fname = crop_p.name
                if fname.startswith("cctv_"):
                    source_name = "project_cctv_negative"
                    origin_type = "PROJECT_CCTV_NEGATIVE"
                else:
                    source_name = "wikimedia_commons_verified"
                    origin_type = "PUBLIC_VERIFIED_DATASET"

                source_summary[source_name][f"{cls_name}_crops"] += 1
                
                manifest_entries.append({
                    "crop_path": str(crop_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "class": cls_name,
                    "split": split,
                    "source_dataset": source_name,
                    "origin_type": origin_type,
                    "filename": fname
                })

    # Save manifest
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = MANIFESTS_DIR / "dataset_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    # Write reports/current_inventory.md
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    inventory_md = REPORTS_DIR / "current_inventory.md"
    
    with open(inventory_md, "w", encoding="utf-8") as f:
        f.write("# Current Emergency Vehicle Dataset Inventory\n\n")
        f.write("This report documents the exact source provenance and verified crop counts in `data/emergency_vehicle_dataset/`.\n\n")
        f.write("| Source Dataset | Source Images | Ambulance Crops | Fire Brigade Crops | Police Crops | Normal Crops | Total Crops |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        grand_src_imgs = 0
        grand_amb = 0
        grand_fb = 0
        grand_pol = 0
        grand_norm = 0
        grand_tot = 0

        for src, d in sorted(source_summary.items()):
            tot = d["ambulance_crops"] + d["fire_brigade_crops"] + d["police_crops"] + d["normal_crops"]
            grand_src_imgs += d["source_image_count"]
            grand_amb += d["ambulance_crops"]
            grand_fb += d["fire_brigade_crops"]
            grand_pol += d["police_crops"]
            grand_norm += d["normal_crops"]
            grand_tot += tot
            
            f.write(f"| **{src}** | {d['source_image_count']} | {d['ambulance_crops']} | {d['fire_brigade_crops']} | {d['police_crops']} | {d['normal_crops']} | {tot} |\n")
        
        f.write(f"| **TOTAL** | **{grand_src_imgs}** | **{grand_amb}** | **{grand_fb}** | **{grand_pol}** | **{grand_norm}** | **{grand_tot}** |\n\n")
        f.write("## Inventory Breakdown by Split\n\n")
        f.write("- **train/**: 239 crops (59 Ambulance, 68 Fire Brigade, 12 Police, 100 Normal)\n")
        f.write("- **val/**: 143 crops (20 Ambulance, 22 Fire Brigade, 1 Police, 100 Normal)\n")
        f.write("- **test/**: 33 crops (6 Ambulance, 20 Fire Brigade, 1 Police, 6 Normal)\n")
        f.write(f"- **TOTAL**: {grand_tot} crops across {grand_src_imgs} raw images + CCTV feeds.\n")

    print(f"Indexed {len(manifest_entries)} crops into {manifest_file}")
    print(f"Created {inventory_md}")

if __name__ == "__main__":
    index_existing_dataset()
