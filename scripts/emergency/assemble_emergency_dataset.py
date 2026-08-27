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
RAW_DIR = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "processed"
MANIFESTS_DIR = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "manifests"

TARGET_CLASSES = ["ambulance", "fire_brigade", "police", "normal"]

SEARCH_QUERIES = {
    "ambulance": [
        "Ambulance in India",
        "Force Traveller ambulance",
        "108 Ambulance India",
        "Tata Winger ambulance",
        "Hospital ambulance vehicle India",
        "Ambulance van India",
        "Emergency ambulance India"
    ],
    "fire_brigade": [
        "Fire engine in India",
        "Fire brigade India",
        "Fire tender India",
        "Fire fighting truck India",
        "Fire service vehicle India",
        "Fire truck in India"
    ],
    "police": [
        "Police vehicle in India",
        "Police car India",
        "Police Bolero India",
        "Police Scorpio India",
        "Police Gypsy India",
        "Traffic police India vehicle",
        "Police van India",
        "Police motorcycle India"
    ],
    "normal": [
        "Force Traveller India",
        "Tata Winger India",
        "Mahindra Bolero India",
        "Mahindra Scorpio India",
        "Toyota Innova India",
        "Tata truck India",
        "Ashok Leyland truck India",
        "Maruti Omni India",
        "Maruti Eeco India"
    ]
}

USER_AGENT = "EmergencyTrafficResearch/1.0 (contact: research@trafficmanagement.org)"

def search_wikimedia_files(query: str, limit: int = 20):
    url = (
        "https://commons.wikimedia.org/w/api.php?action=query&format=json"
        f"&list=search&srsearch={urllib.parse.quote(query)}&srnamespace=6&srlimit={limit}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            search_items = data.get("query", {}).get("search", [])
            titles = [item["title"] for item in search_items if item["title"].lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not titles:
                return []
            
            # Fetch image URLs in a batch
            info_url = (
                "https://commons.wikimedia.org/w/api.php?action=query&format=json"
                f"&titles={urllib.parse.quote('|'.join(titles))}&prop=imageinfo&iiprop=url|size"
            )
            req_info = urllib.request.Request(info_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req_info, timeout=12) as info_resp:
                info_data = json.loads(info_resp.read().decode('utf-8'))
                pages = info_data.get("query", {}).get("pages", {})
                results = []
                for p_id, p_info in pages.items():
                    imageinfo = p_info.get("imageinfo", [])
                    if imageinfo and "url" in imageinfo[0]:
                        results.append((p_info.get("title", f"img_{p_id}"), imageinfo[0]["url"]))
                return results
    except Exception as e:
        print(f"  Warning searching for '{query}': {e}")
        return []

def download_image(url: str, save_path: Path):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            with open(save_path, 'wb') as f:
                f.write(content)
        return True
    except Exception:
        return False

def extract_crops_from_project_videos(model, target_crops_count=120):
    print("\nExtracting authentic CCTV normal/hard-negative crops from project video feeds...")
    video_splits = {
        "data/uploads/my_traffic.mp4": "train",
        "data/uploads/bidirectional.mp4": "val",
        "data/uploads/sample_north.mp4": "test"
    }

    crops_added = {"train": 0, "val": 0, "test": 0}
    
    for rel_vpath, split in video_splits.items():
        vpath = PROJECT_ROOT / rel_vpath
        if not vpath.exists():
            continue
        
        cap = cv2.VideoCapture(str(vpath))
        frame_idx = 0
        extracted_for_video = 0
        
        while cap.isOpened() and extracted_for_video < target_crops_count:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 6 != 0:
                continue
                
            results = model.predict(source=frame, conf=0.25, classes=[2, 3, 5, 7], device='cpu', verbose=False)
            if results and len(results[0].boxes) > 0:
                h_img, w_img = frame.shape[:2]
                for b in results[0].boxes:
                    xyxy = b.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    bw, bh = x2 - x1, y2 - y1
                    if bw < 32 or bh < 32:
                        continue
                    
                    pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
                    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    cx2, cy2 = min(w_img, x2 + pad_x), min(h_img, y2 + pad_y)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.shape[0] < 24 or crop.shape[1] < 24:
                        continue
                    
                    crop_resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
                    c_name = "normal"
                    c_fname = f"cctv_{Path(rel_vpath).stem}_f{frame_idx:04d}_box{extracted_for_video:03d}.jpg"
                    save_path = PROCESSED_DIR / split / c_name / c_fname
                    cv2.imwrite(str(save_path), crop_resized)
                    crops_added[split] += 1
                    extracted_for_video += 1
                    if extracted_for_video >= target_crops_count:
                        break
        cap.release()
    print(f"Extracted CCTV crops: {crops_added}")
    return crops_added

def assemble_and_extract_crops():
    print("=" * 80)
    print("ACQUIRING AUTHENTIC INDIAN EMERGENCY & HARD-NEGATIVE DATASET")
    print("=" * 80)

    for c in TARGET_CLASSES:
        (RAW_DIR / c).mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            (PROCESSED_DIR / split / c).mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8s.pt")

    random.seed(42)
    total_downloaded = 0
    total_crops_generated = defaultdict(lambda: defaultdict(int))
    dataset_manifest = []

    for cls_name, queries in SEARCH_QUERIES.items():
        print(f"\nProcessing Category: {cls_name.upper()}")
        seen_urls = set()
        img_idx = 0

        for q in queries:
            print(f"  Searching query: '{q}' ...")
            results = search_wikimedia_files(q, limit=12)
            time.sleep(0.8)

            for title, img_url in results:
                if img_url in seen_urls:
                    continue
                seen_urls.add(img_url)

                img_filename = f"{cls_name}_{img_idx:04d}.jpg"
                raw_path = RAW_DIR / cls_name / img_filename

                if not raw_path.exists():
                    time.sleep(0.4)
                    ok = download_image(img_url, raw_path)
                    if not ok:
                        continue

                img = cv2.imread(str(raw_path))
                if img is None or img.shape[0] < 50 or img.shape[1] < 50:
                    if raw_path.exists():
                        raw_path.unlink()
                    continue

                total_downloaded += 1
                img_idx += 1

                # Deterministic source-level split (70% train, 15% val, 15% test)
                rand_val = random.random()
                if rand_val < 0.70:
                    split = "train"
                elif rand_val < 0.85:
                    split = "val"
                else:
                    split = "test"

                results = model.predict(source=img, conf=0.15, classes=[2, 3, 5, 7], device='cpu', verbose=False)
                
                h_img, w_img = img.shape[:2]
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
                        crop_filename = f"{cls_name}_{img_idx:04d}_crop{crop_idx:02d}.jpg"
                        crop_save_path = PROCESSED_DIR / split / cls_name / crop_filename
                        cv2.imwrite(str(crop_save_path), crop_resized)

                        total_crops_generated[split][cls_name] += 1
                        has_valid_crop = True

                        dataset_manifest.append({
                            "crop_path": str(crop_save_path.relative_to(PROJECT_ROOT)),
                            "source_image": str(raw_path.relative_to(PROJECT_ROOT)),
                            "source_url": img_url,
                            "class": cls_name,
                            "split": split,
                            "original_crop_w": bw,
                            "original_crop_h": bh
                        })
                        crop_idx += 1

                if not has_valid_crop:
                    crop_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                    crop_filename = f"{cls_name}_{img_idx:04d}_full00.jpg"
                    crop_save_path = PROCESSED_DIR / split / cls_name / crop_filename
                    cv2.imwrite(str(crop_save_path), crop_resized)

                    total_crops_generated[split][cls_name] += 1
                    dataset_manifest.append({
                        "crop_path": str(crop_save_path.relative_to(PROJECT_ROOT)),
                        "source_image": str(raw_path.relative_to(PROJECT_ROOT)),
                        "source_url": img_url,
                        "class": cls_name,
                        "split": split,
                        "original_crop_w": w_img,
                        "original_crop_h": h_img
                    })

    # Also extract authentic CCTV normal crops with source-level split
    cctv_crops = extract_crops_from_project_videos(model, target_crops_count=100)
    for sp, cnt in cctv_crops.items():
        total_crops_generated[sp]["normal"] += cnt

    # Save manifest
    manifest_file = MANIFESTS_DIR / "dataset_manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("DATASET ASSEMBLY & CROPPING COMPLETE")
    print("=" * 80)
    print(f"Total Source Images Downloaded: {total_downloaded}")
    print(f"Manifest written to: {manifest_file}")
    print("\nFINAL CROP SPLIT DISTRIBUTION:")
    print(f"{'Class':<15} | {'TRAIN':<8} | {'VAL':<8} | {'TEST':<8} | {'TOTAL':<8}")
    print("-" * 55)
    
    grand_total = 0
    for c in TARGET_CLASSES:
        tr = total_crops_generated["train"][c]
        vl = total_crops_generated["val"][c]
        ts = total_crops_generated["test"][c]
        tot = tr + vl + ts
        grand_total += tot
        print(f"{c.upper():<15} | {tr:<8} | {vl:<8} | {ts:<8} | {tot:<8}")
    print("-" * 55)
    print(f"{'TOTAL':<15} | {sum(total_crops_generated['train'].values()):<8} | {sum(total_crops_generated['val'].values()):<8} | {sum(total_crops_generated['test'].values()):<8} | {grand_total:<8}")
    print("=" * 80)

if __name__ == "__main__":
    assemble_and_extract_crops()
