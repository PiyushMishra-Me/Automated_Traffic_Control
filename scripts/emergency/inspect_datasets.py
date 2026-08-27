import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATASETS_DIR = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "raw"
REPORTS_DIR = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "reports"

def inspect_dataset_directory(dataset_path: Path):
    """
    Inspects an individual dataset folder for image formats, YOLO/VOC annotations,
    classes, bounding box counts, corrupt images, duplicates, and dimensional statistics.
    """
    results = {
        "dataset_name": dataset_path.name,
        "dataset_path": str(dataset_path),
        "total_images": 0,
        "total_annotations": 0,
        "corrupt_images": 0,
        "empty_annotations": 0,
        "duplicate_filenames": 0,
        "image_dimensions": [],
        "classes_found": Counter(),
        "images_per_class": Counter(),
        "annotation_format": "UNKNOWN",
        "suspicious_annotations": 0,
        "yaml_classes": []
    }

    if not dataset_path.exists():
        return results

    # Check for data.yaml or obj.names
    yaml_files = list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("**/*.yaml"))
    classes_from_yaml = []
    if yaml_files:
        try:
            import yaml
            with open(yaml_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                data_cfg = yaml.safe_load(f)
                if isinstance(data_cfg, dict) and "names" in data_cfg:
                    names = data_cfg["names"]
                    if isinstance(names, list):
                        classes_from_yaml = names
                    elif isinstance(names, dict):
                        classes_from_yaml = [names[k] for k in sorted(names.keys())]
                    results["yaml_classes"] = classes_from_yaml
        except Exception as e:
            pass

    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [p for p in dataset_path.rglob("*") if p.suffix.lower() in image_extensions]
    results["total_images"] = len(image_files)

    seen_filenames = set()
    label_extensions = {".txt", ".xml", ".json"}
    
    for img_p in image_files:
        if img_p.name in seen_filenames:
            results["duplicate_filenames"] += 1
        seen_filenames.add(img_p.name)

        # Validate image integrity & read dimensions
        try:
            img = cv2.imread(str(img_p))
            if img is None:
                results["corrupt_images"] += 1
                continue
            h, w, c = img.shape
            results["image_dimensions"].append((w, h))
        except Exception:
            results["corrupt_images"] += 1
            continue

        # Look for corresponding annotation file (YOLO .txt or Pascal VOC .xml)
        txt_ann = img_p.with_suffix(".txt")
        xml_ann = img_p.with_suffix(".xml")

        classes_in_img = set()

        if txt_ann.exists():
            results["annotation_format"] = "YOLO_TXT"
            try:
                with open(txt_ann, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    results["empty_annotations"] += 1
                for l in lines:
                    parts = l.split()
                    if len(parts) >= 5:
                        try:
                            cls_id = int(float(parts[0]))
                            # map cls_id to name if yaml classes exist
                            if classes_from_yaml and cls_id < len(classes_from_yaml):
                                cls_name = str(classes_from_yaml[cls_id])
                            else:
                                cls_name = f"class_{cls_id}"
                            
                            results["classes_found"][cls_name] += 1
                            classes_in_img.add(cls_name)
                            results["total_annotations"] += 1

                            # check bounding box validity
                            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            if xc < 0 or xc > 1 or yc < 0 or yc > 1 or bw <= 0 or bh <= 0 or bw > 1.0 or bh > 1.0:
                                results["suspicious_annotations"] += 1
                        except ValueError:
                            results["suspicious_annotations"] += 1
            except Exception:
                results["empty_annotations"] += 1
        elif xml_ann.exists():
            results["annotation_format"] = "PASCAL_VOC_XML"
            # XML parser
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(str(xml_ann))
                root = tree.getroot()
                objects = root.findall("object")
                if not objects:
                    results["empty_annotations"] += 1
                for obj in objects:
                    name_elem = obj.find("name")
                    if name_elem is not None and name_elem.text:
                        cls_name = name_elem.text.strip()
                        results["classes_found"][cls_name] += 1
                        classes_in_img.add(cls_name)
                        results["total_annotations"] += 1
            except Exception:
                results["empty_annotations"] += 1
        else:
            results["empty_annotations"] += 1

        for c_name in classes_in_img:
            results["images_per_class"][c_name] += 1

    return results

def run_inspection():
    print("=" * 80)
    print("EMERGENCY VEHICLE DATASET INSPECTION")
    print(f"Scanning directory: {RAW_DATASETS_DIR}")
    print("=" * 80)

    dataset_dirs = [p for p in RAW_DATASETS_DIR.iterdir() if p.is_dir()]
    if not dataset_dirs:
        print(f"No subdirectories found under {RAW_DATASETS_DIR}.")
        print("Note: Candidate verified sources documented in DATASET_SOURCES.md.")
        return []

    inspection_results = []
    for d in dataset_dirs:
        print(f"\nAnalyzing: {d.name} ...")
        res = inspect_dataset_directory(d)
        inspection_results.append(res)
        print(f"  Total Images: {res['total_images']}")
        print(f"  Total Annotations: {res['total_annotations']}")
        print(f"  Annotation Format: {res['annotation_format']}")
        print(f"  Classes Detected: {dict(res['classes_found'])}")
        print(f"  Corrupted Images: {res['corrupt_images']}")
        print(f"  Empty Annotations: {res['empty_annotations']}")
        print(f"  Duplicate Filenames: {res['duplicate_filenames']}")
        print(f"  Suspicious Annotations: {res['suspicious_annotations']}")

    # Save summary report JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "dataset_inspection_summary.json"
    serializable_results = []
    for r in inspection_results:
        sr = dict(r)
        sr["classes_found"] = dict(r["classes_found"])
        sr["images_per_class"] = dict(r["images_per_class"])
        if sr["image_dimensions"]:
            widths = [w for w, h in sr["image_dimensions"]]
            heights = [h for w, h in sr["image_dimensions"]]
            sr["dim_summary"] = {
                "min_width": int(min(widths)),
                "max_width": int(max(widths)),
                "median_width": int(np.median(widths)),
                "min_height": int(min(heights)),
                "max_height": int(max(heights)),
                "median_height": int(np.median(heights)),
            }
        else:
            sr["dim_summary"] = {}
        sr.pop("image_dimensions", None)
        serializable_results.append(sr)

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nSaved inspection summary to {report_file}")
    return inspection_results

if __name__ == "__main__":
    run_inspection()
