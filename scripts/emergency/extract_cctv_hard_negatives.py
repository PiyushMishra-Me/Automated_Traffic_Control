import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "emergency_vehicle_dataset"
CCTV_HARD_NEG_DIR = DATASET_ROOT / "cctv_hard_negatives"
PROCESSED_DIR = DATASET_ROOT / "processed"
MANIFESTS_DIR = DATASET_ROOT / "manifests"
REPORTS_DIR = DATASET_ROOT / "reports"
BASELINE_MODEL_PATH = PROJECT_ROOT / "runs" / "emergency_classifier" / "baseline" / "weights" / "best.pt"

SCALE_BINS = [
    ("<32 px", lambda h: h < 32),
    ("32–40 px", lambda h: 32 <= h < 40),
    ("40–48 px", lambda h: 40 <= h < 48),
    ("48–56 px", lambda h: 48 <= h < 56),
    ("56–64 px", lambda h: 56 <= h < 64),
    ("64–80 px", lambda h: 64 <= h < 80),
    ("80–100 px", lambda h: 80 <= h < 100),
    (">100 px", lambda h: h >= 100)
]

def get_scale_bin(h):
    for name, func in SCALE_BINS:
        if func(h):
            return name
    return ">100 px"

def extract_cctv_hard_negatives():
    print("=" * 80)
    print("STEP 1 to 5: EXTRACTING REAL CCTV HARD-NEGATIVE CROPS WITH PROVENANCE")
    print("=" * 80)

    CCTV_HARD_NEG_DIR.mkdir(parents=True, exist_ok=True)
    for name, _ in SCALE_BINS:
        clean_bin_name = name.replace("–", "-").replace("<", "lt_").replace(">", "gt_").replace(" ", "_")
        (CCTV_HARD_NEG_DIR / clean_bin_name).mkdir(parents=True, exist_ok=True)

    detector = YOLO("yolov8s.pt")
    
    videos = [
        ("my_traffic.mp4", PROJECT_ROOT / "data" / "uploads" / "my_traffic.mp4"),
        ("bidirectional.mp4", PROJECT_ROOT / "data" / "uploads" / "bidirectional.mp4")
    ]

    cctv_manifest = []
    crops_per_scale = defaultdict(int)
    crops_per_class = defaultdict(int)
    tracks_seen = set()
    crops_per_track = defaultdict(list)

    total_extracted = 0

    for vname, vpath in videos:
        if not vpath.exists():
            print(f"Warning: {vpath} not found!")
            continue
        print(f"\nProcessing {vname} for CCTV hard negative extraction...")
        
        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0

        # Sample every 3rd frame to capture realistic motion/blur/scale transitions
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 3 != 0:
                continue

            results = detector.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.15,
                classes=[2, 3, 5, 7], # car, motorcycle, bus, truck
                imgsz=960,
                device='cpu',
                verbose=False
            )

            if results and len(results[0].boxes) > 0 and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                classes_detected = results[0].boxes.cls.int().cpu().numpy()
                class_names_map = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

                h_f, w_f = frame.shape[:2]

                for xyxy, tid, c_id in zip(boxes, track_ids, classes_detected):
                    x1, y1, x2, y2 = map(int, xyxy)
                    bw, bh = x2 - x1, y2 - y1
                    if bw < 12 or bh < 12: # capture even tiny distant vehicles
                        continue

                    track_key = f"{vname}_track_{tid}"
                    tracks_seen.add(track_key)

                    # Context padding
                    pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
                    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    cx2, cy2 = min(w_f, x2 + pad_x), min(h_f, y2 + pad_y)
                    crop = frame[cy1:cy2, cx1:cx2]

                    if crop.shape[0] < 10 or crop.shape[1] < 10:
                        continue

                    crop_resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)

                    scale_bin = get_scale_bin(bh)
                    clean_bin_name = scale_bin.replace("–", "-").replace("<", "lt_").replace(">", "gt_").replace(" ", "_")

                    v_cls_name = class_names_map.get(int(c_id), "vehicle")
                    crop_filename = f"{vname}_f{frame_idx:04d}_tid{tid:03d}_{v_cls_name}_h{bh:03d}.jpg"
                    crop_path = CCTV_HARD_NEG_DIR / clean_bin_name / crop_filename

                    cv2.imwrite(str(crop_path), crop_resized)

                    total_extracted += 1
                    crops_per_scale[scale_bin] += 1
                    crops_per_class[v_cls_name] += 1

                    meta = {
                        "crop_path": str(crop_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "source_video": vname,
                        "frame_number": frame_idx,
                        "track_id": int(tid),
                        "track_key": track_key,
                        "vehicle_class": v_cls_name,
                        "bbox": [x1, y1, x2, y2],
                        "bbox_width": int(bw),
                        "bbox_height": int(bh),
                        "scale_bin": scale_bin
                    }
                    cctv_manifest.append(meta)
                    crops_per_track[track_key].append(meta)

        cap.release()

    # Save CCTV manifest
    cctv_manifest_file = MANIFESTS_DIR / "cctv_hard_negatives_manifest.json"
    with open(cctv_manifest_file, "w", encoding="utf-8") as f:
        json.dump(cctv_manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"Total CCTV Hard-Negative Crops Extracted: {total_extracted}")
    print(f"Total Unique CCTV Vehicle Tracks:         {len(tracks_seen)}")
    print(f"CCTV Manifest Saved:                     {cctv_manifest_file}")

    print("\n1. SCALE DISTRIBUTION OF CCTV CROPS:")
    print(f"{'Scale Group':<15} | {'Crop Count':<12} | {'Share (%)':<10}")
    print("-" * 45)
    for bin_name, _ in SCALE_BINS:
        cnt = crops_per_scale[bin_name]
        pct = cnt / total_extracted * 100.0 if total_extracted > 0 else 0.0
        print(f"{bin_name:<15} | {cnt:<12} | {pct:<9.1f}%")
    print("-" * 45)

    print("\n2. VEHICLE CLASS DISTRIBUTION OF CCTV CROPS:")
    for v_cls, cnt in sorted(crops_per_class.items()):
        print(f"  - {v_cls:<12}: {cnt:>5} crops ({(cnt/total_extracted*100):.1f}%)")

    return cctv_manifest, crops_per_scale, crops_per_class, len(tracks_seen)

def analyze_existing_emergency_scale_distribution():
    print("\n" + "=" * 80)
    print("STEP 6: SCALE DISTRIBUTION OF EXISTING EMERGENCY TRAINING DATASET")
    print("=" * 80)

    manifest_file = MANIFESTS_DIR / "dataset_manifest.json"
    if not manifest_file.exists():
        print("Manifest not found.")
        return {}

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Class -> Scale Bin -> Count
    emergency_scales = defaultdict(lambda: defaultdict(int))
    class_totals = defaultdict(int)

    for entry in manifest:
        cls_name = entry.get("class", "unknown")
        h = entry.get("original_crop_h", 128)
        bin_name = get_scale_bin(h)
        emergency_scales[cls_name][bin_name] += 1
        class_totals[cls_name] += 1

    classes = ["ambulance", "fire_brigade", "police", "normal"]
    print(f"{'Scale Group':<12} | {'AMBULANCE':<10} | {'FIRE_BRIGADE':<12} | {'POLICE':<10} | {'NORMAL':<10} | {'TOTAL':<8}")
    print("-" * 75)
    
    total_small_emergency = 0
    for bin_name, _ in SCALE_BINS:
        amb = emergency_scales["ambulance"][bin_name]
        fb = emergency_scales["fire_brigade"][bin_name]
        pol = emergency_scales["police"][bin_name]
        norm = emergency_scales["normal"][bin_name]
        tot = amb + fb + pol + norm
        if bin_name in ["<32 px", "32–40 px", "40–48 px"]:
            total_small_emergency += (amb + fb + pol)
        print(f"{bin_name:<12} | {amb:<10} | {fb:<12} | {pol:<10} | {norm:<10} | {tot:<8}")
    print("-" * 75)
    print(f"{'TOTAL':<12} | {class_totals['ambulance']:<10} | {class_totals['fire_brigade']:<12} | {class_totals['police']:<10} | {class_totals['normal']:<10} | {len(manifest):<8}")

    print("\nFINDING ON SMALL EMERGENCY DATA (<48 px):")
    if total_small_emergency == 0:
        print("  [!] INSUFFICIENT SMALL EMERGENCY DATA: 0 samples < 48 px exist in the emergency classes.")
        print("      All positive emergency training samples are high-resolution web photos (>64 px).")
    else:
        print(f"  Small emergency samples (<48 px): {total_small_emergency}")

    return emergency_scales

def evaluate_resolution_gating_and_temporal_combos():
    print("\n" + "=" * 80)
    print("STEP 7 & 8: RESOLUTION GATE & TEMPORAL COMBINATION EXPERIMENT")
    print("=" * 80)

    if not BASELINE_MODEL_PATH.exists():
        print("Baseline model not found!")
        return

    classifier = YOLO(str(BASELINE_MODEL_PATH))
    detector = YOLO("yolov8s.pt")

    videos = [
        ("my_traffic.mp4", PROJECT_ROOT / "data" / "uploads" / "my_traffic.mp4"),
        ("bidirectional.mp4", PROJECT_ROOT / "data" / "uploads" / "bidirectional.mp4")
    ]

    gate_thresholds = [32, 40, 48, 56, 64]
    temporal_combos = [
        (40, 5), (48, 5), (56, 5),
        (48, 10), (56, 10), (64, 10)
    ]

    experiment_results = {}

    for vname, vpath in videos:
        if not vpath.exists():
            continue
        print(f"\nEvaluating Resolution Gating on {vname} ...")
        cap = cv2.VideoCapture(str(vpath))
        frame_idx = 0

        # Store all detections per frame and track
        all_crops = [] # list of (track_id, crop_resized, bh, frame_idx)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = detector.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.20,
                classes=[2, 3, 5, 7],
                imgsz=960,
                device='cpu',
                verbose=False
            )

            if results and len(results[0].boxes) > 0 and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                h_f, w_f = frame.shape[:2]

                for xyxy, tid in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, xyxy)
                    bw, bh = x2 - x1, y2 - y1
                    if bw < 15 or bh < 15:
                        continue

                    pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
                    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    cx2, cy2 = min(w_f, x2 + pad_x), min(h_f, y2 + pad_y)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.shape[0] < 10 or crop.shape[1] < 10:
                        continue

                    crop_resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
                    all_crops.append((int(tid), crop_resized, int(bh), frame_idx))

        cap.release()

        total_crops = len(all_crops)
        unique_tracks = len(set(tid for tid, _, _, _ in all_crops))
        print(f"  Total crops collected: {total_crops} across {unique_tracks} tracks.")

        # Run classification once for all crops
        print("  Running classifier predictions...")
        crop_predictions = [] # (tid, bh, frame_idx, pred_cls, conf, is_emergency)
        for tid, crop_img, bh, f_idx in all_crops:
            cls_res = classifier.predict(source=crop_img, imgsz=128, device='cpu', verbose=False)
            top1_idx = int(cls_res[0].probs.top1)
            pred_cls = cls_res[0].names[top1_idx]
            conf = float(cls_res[0].probs.top1conf.cpu().numpy())
            is_emergency = pred_cls in ["ambulance", "fire_brigade", "police"]
            crop_predictions.append((tid, bh, f_idx, pred_cls, conf, is_emergency))

        # STEP 7: Resolution Gate Evaluation
        print(f"\n--- Resolution Gate Sweep on {vname} (Single-Frame False Positives) ---")
        print(f"{'Gate Height':<12} | {'Evaluated':<10} | {'Rejected (Pending)':<20} | {'False Alarms':<13} | {'Raw FP Rate':<12} | {'Traffic Rejected':<16}")
        print("-" * 95)
        
        gate_data = {}
        for gate_h in [0] + gate_thresholds:
            evaluated = [p for p in crop_predictions if p[1] >= gate_h]
            rejected = [p for p in crop_predictions if p[1] < gate_h]
            
            n_eval = len(evaluated)
            n_rej = len(rejected)
            n_fp = sum(1 for p in evaluated if p[5])
            fp_rate = (n_fp / n_eval * 100.0) if n_eval > 0 else 0.0
            rej_pct = (n_rej / total_crops * 100.0) if total_crops > 0 else 0.0
            
            gate_name = f">= {gate_h} px" if gate_h > 0 else "None (0 px)"
            print(f"{gate_name:<12} | {n_eval:<10} | {n_rej:<20} | {n_fp:<13} | {fp_rate:<11.2f}% | {rej_pct:<15.1f}%")
            gate_data[gate_h] = {
                "evaluated": n_eval,
                "rejected": n_rej,
                "false_alarms": n_fp,
                "fp_rate": fp_rate,
                "rejected_pct": rej_pct
            }
        print("-" * 95)

        # STEP 8: Temporal + Resolution Combination Sweep
        print(f"\n--- Step 8: Resolution Gate + Temporal Combination Sweep on {vname} ---")
        print(f"{'Combination':<22} | {'Total Tracks':<13} | {'False Confirmed Tracks':<24} | {'Track FP Rate':<14}")
        print("-" * 80)
        
        track_preds_map = defaultdict(list)
        for tid, bh, f_idx, pred_cls, conf, is_emergency in crop_predictions:
            track_preds_map[tid].append({"bh": bh, "conf": conf, "is_emergency": is_emergency, "frame": f_idx})

        combo_data = {}
        for gate_h, consec_k in [(0, 1), (0, 5)] + temporal_combos:
            confirmed_tracks = 0
            for tid, preds in track_preds_map.items():
                max_consec = 0
                curr_consec = 0
                for p in preds:
                    # Gated by resolution AND confidence
                    if p["bh"] >= gate_h and p["is_emergency"] and p["conf"] >= 0.60:
                        curr_consec += 1
                        if curr_consec > max_consec:
                            max_consec = curr_consec
                    else:
                        curr_consec = 0
                if max_consec >= consec_k:
                    confirmed_tracks += 1

            combo_name = f"{gate_h}px + {consec_k} frames" if gate_h > 0 else f"No Gate + {consec_k} frame(s)"
            track_fp_rate = (confirmed_tracks / unique_tracks * 100.0) if unique_tracks > 0 else 0.0
            print(f"{combo_name:<22} | {unique_tracks:<13} | {confirmed_tracks:<24} | {track_fp_rate:<13.1f}%")
            combo_data[f"{gate_h}px_{consec_k}f"] = {
                "gate_h": gate_h,
                "consec_k": consec_k,
                "confirmed_tracks": confirmed_tracks,
                "track_fp_rate": track_fp_rate
            }
        print("-" * 80)

        experiment_results[vname] = {
            "total_crops": total_crops,
            "unique_tracks": unique_tracks,
            "gate_data": gate_data,
            "combo_data": combo_data
        }

    # Save detailed report
    report_file = REPORTS_DIR / "cctv_hard_negative_and_gate_study.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\nSaved detailed analysis to {report_file}")
    return experiment_results

if __name__ == "__main__":
    cctv_manifest, crops_per_scale, crops_per_class, num_tracks = extract_cctv_hard_negatives()
    emergency_scales = analyze_existing_emergency_scale_distribution()
    gate_results = evaluate_resolution_gating_and_temporal_combos()
