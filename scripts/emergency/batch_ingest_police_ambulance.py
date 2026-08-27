import os
import sys
import json
import time
import random
import urllib.request
import urllib.parse
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "emergency_vehicle_dataset"
RAW_DIR = DATASET_ROOT / "raw"
PROCESSED_DIR = DATASET_ROOT / "processed"
MANIFESTS_DIR = DATASET_ROOT / "manifests"
REPORTS_DIR = DATASET_ROOT / "reports"

USER_AGENT = "EmergencyTrafficResearch/1.0 (contact: research@trafficmanagement.org)"

BATCH_QUERIES = {
    "police": [
        "Delhi Police vehicle",
        "Mumbai Police vehicle",
        "Kerala Police vehicle",
        "Police car in India",
        "Police Bolero India",
        "Police PCR van India",
        "Police motorcycle India",
        "Indian police vehicle"
    ],
    "ambulance": [
        "108 ambulance India",
        "Force Traveller ambulance",
        "Tata Winger ambulance",
        "Ambulance in Karnataka",
        "Ambulance in Tamil Nadu",
        "Ambulance in Maharashtra"
    ]
}

TARGET_MAX_IMAGES = {
    "police": 25,
    "ambulance": 15
}

def search_wikimedia_files(query: str, limit: int = 8):
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json"
        f"&list=search&srsearch={urllib.parse.quote(query)}&srnamespace=6&srlimit={limit}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            search_items = data.get("query", {}).get("search", [])
            titles = [item["title"] for item in search_items if item["title"].lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not titles:
                return []
            
            info_url = (
                "https://commons.wikimedia.org/w/api.php?action=query&format=json"
                f"&titles={urllib.parse.quote('|'.join(titles))}&prop=imageinfo&iiprop=url|size"
            )
            req_info = urllib.request.Request(info_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req_info, timeout=10) as info_resp:
                info_data = json.loads(info_resp.read().decode('utf-8'))
                pages = info_data.get("query", {}).get("pages", {})
                results = []
                for p_id, p_info in pages.items():
                    imageinfo = p_info.get("imageinfo", [])
                    if imageinfo and "url" in imageinfo[0]:
                        results.append((p_info.get("title", f"img_{p_id}"), imageinfo[0]["url"]))
                return results
    except Exception as e:
        print(f"  Warning searching '{query}': {e}")
        return []

def download_image(url: str, save_path: Path):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            content = resp.read()
            with open(save_path, 'wb') as f:
                f.write(content)
        return True
    except Exception:
        return False

def run_bounded_batch():
    print("=" * 75)
    print("STARTING CONTROLLED BATCH INGESTION (PRIORITY: POLICE & AMBULANCE)")
    print("=" * 75)

    start_total_time = time.time()
    
    # Load existing manifest to avoid duplicate downloads
    manifest_file = MANIFESTS_DIR / "dataset_manifest.json"
    manifest_entries = []
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_entries = json.load(f)
        except Exception:
            manifest_entries = []

    seen_source_files = {entry.get("source_image") for entry in manifest_entries if "source_image" in entry}

    # Count existing raw files to assign sequential IDs
    existing_raw_counts = {
        c: len(list((RAW_DIR / c).glob("*"))) if (RAW_DIR / c).exists() else 0
        for c in ["ambulance", "fire_brigade", "police", "normal"]
    }

    print(f"Initial raw image counts: {existing_raw_counts}")

    download_stats = {"police": 0, "ambulance": 0}
    crops_added = defaultdict(int)
    corrupted_count = 0

    download_start_time = time.time()
    downloaded_new_files = [] # list of (cls_name, raw_path, img_url)

    for cls_name, queries in BATCH_QUERIES.items():
        max_target = TARGET_MAX_IMAGES[cls_name]
        print(f"\n--- Ingesting Batch for Class: {cls_name.upper()} (Target: max {max_target} images) ---")
        (RAW_DIR / cls_name).mkdir(parents=True, exist_ok=True)
        
        cls_seen_urls = set()
        cls_img_idx = existing_raw_counts[cls_name]

        for q in queries:
            if download_stats[cls_name] >= max_target:
                break
            
            print(f"  Searching: '{q}' ...")
            results = search_wikimedia_files(q, limit=6)
            time.sleep(0.8) # Polite delay

            for title, img_url in results:
                if download_stats[cls_name] >= max_target:
                    break
                if img_url in cls_seen_urls:
                    continue
                cls_seen_urls.add(img_url)

                raw_filename = f"{cls_name}_{cls_img_idx:04d}.jpg"
                raw_path = RAW_DIR / cls_name / raw_filename

                if raw_path.exists():
                    cls_img_idx += 1
                    continue

                time.sleep(0.4)
                ok = download_image(img_url, raw_path)
                if not ok:
                    continue

                # Validate image readability
                img = cv2.imread(str(raw_path))
                if img is None or img.shape[0] < 50 or img.shape[1] < 50:
                    corrupted_count += 1
                    if raw_path.exists():
                        raw_path.unlink()
                    continue

                download_stats[cls_name] += 1
                cls_img_idx += 1
                downloaded_new_files.append((cls_name, raw_path, img_url))
                print(f"    [+] Downloaded: {raw_filename} ({img.shape[1]}x{img.shape[0]})")

    download_duration = time.time() - download_start_time
    print(f"\nDownload phase completed in {download_duration:.1f}s. Total newly downloaded: {len(downloaded_new_files)}")

    # Extraction phase with YOLOv8s
    extraction_start_time = time.time()
    print("\n--- Running Vehicle Crop Extraction ---")
    model = YOLO("yolov8s.pt")

    random.seed(101)

    for cls_name, raw_path, img_url in downloaded_new_files:
        img = cv2.imread(str(raw_path))
        if img is None:
            continue

        h_img, w_img = img.shape[:2]
        
        # Source-level deterministic split
        rand_val = random.random()
        if rand_val < 0.70:
            split = "train"
        elif rand_val < 0.85:
            split = "val"
        else:
            split = "test"

        (PROCESSED_DIR / split / cls_name).mkdir(parents=True, exist_ok=True)

        results = model.predict(source=img, conf=0.15, classes=[2, 3, 5, 7], device='cpu', verbose=False)
        crop_idx = 0
        has_valid_crop = False

        if results and len(results[0].boxes) > 0:
            for b in results[0].boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                bw, bh = x2 - x1, y2 - y1

                if bw < 25 or bh < 25:
                    continue

                pad_x = int(bw * 0.08)
                pad_y = int(bh * 0.08)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w_img, x2 + pad_x)
                cy2 = min(h_img, y2 + pad_y)

                crop = img[cy1:cy2, cx1:cx2]
                if crop.shape[0] < 20 or crop.shape[1] < 20:
                    continue

                crop_resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
                crop_filename = f"{raw_path.stem}_crop{crop_idx:02d}.jpg"
                crop_save_path = PROCESSED_DIR / split / cls_name / crop_filename
                cv2.imwrite(str(crop_save_path), crop_resized)

                crops_added[cls_name] += 1
                has_valid_crop = True

                manifest_entries.append({
                    "crop_path": str(crop_save_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "source_image": str(raw_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "source_url": img_url,
                    "class": cls_name,
                    "split": split,
                    "origin_type": "PUBLIC_VERIFIED_DATASET",
                    "original_crop_w": bw,
                    "original_crop_h": bh
                })
                crop_idx += 1

        if not has_valid_crop:
            crop_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
            crop_filename = f"{raw_path.stem}_full00.jpg"
            crop_save_path = PROCESSED_DIR / split / cls_name / crop_filename
            cv2.imwrite(str(crop_save_path), crop_resized)

            crops_added[cls_name] += 1
            manifest_entries.append({
                "crop_path": str(crop_save_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_image": str(raw_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_url": img_url,
                "class": cls_name,
                "split": split,
                "origin_type": "PUBLIC_VERIFIED_DATASET",
                "original_crop_w": w_img,
                "original_crop_h": h_img
            })

    extraction_duration = time.time() - extraction_start_time
    print(f"Extraction phase completed in {extraction_duration:.1f}s.")

    # Write updated manifest
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    # Re-run build_manifest_and_inventory to update Markdown report
    from build_manifest_and_inventory import index_existing_dataset
    index_existing_dataset()

    print("\n" + "=" * 75)
    print("CONTROLLED BATCH SUMMARY")
    print("=" * 75)
    print(f"Download duration: {download_duration:.1f}s")
    print(f"Extraction duration: {extraction_duration:.1f}s")
    print(f"Total new images downloaded: {len(downloaded_new_files)}")
    print(f"Corrupted / invalid images rejected: {corrupted_count}")
    print(f"New crops generated:")
    for k, v in crops_added.items():
        print(f"  - {k.upper()}: +{v} crops")
    print("=" * 75)

if __name__ == "__main__":
    run_bounded_batch()
