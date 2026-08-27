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
TEST_DIR = DATASET_ROOT / "processed" / "test"
MANIFEST_FILE = DATASET_ROOT / "manifests" / "dataset_manifest.json"
MODEL_PATH = PROJECT_ROOT / "runs" / "emergency_classifier" / "baseline" / "weights" / "best.pt"
REPORTS_DIR = DATASET_ROOT / "reports"

CLASSES = ["ambulance", "fire_brigade", "police", "normal"]

def evaluate_held_out_test_set(model):
    print("=" * 75)
    print("STEP 3 & 4: EVALUATION ON HELD-OUT TEST SET & CONFIDENCE ANALYSIS")
    print("=" * 75)

    y_true = []
    y_pred = []
    confidences = []
    crop_info = [] # list of (true_cls, pred_cls, conf, crop_w, crop_h, crop_path)

    # Load manifest to look up original crop dimensions
    crop_dims_map = {}
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for entry in manifest:
            cp = entry.get("crop_path", "").replace("\\", "/")
            crop_dims_map[Path(cp).name] = (entry.get("original_crop_w", 128), entry.get("original_crop_h", 128))

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = TEST_DIR / cls_name
        if not cls_dir.exists():
            continue
        for img_p in cls_dir.glob("*.jpg"):
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            results = model.predict(source=img, imgsz=128, device='cpu', verbose=False)
            r = results[0]
            top1_idx = int(r.probs.top1)
            pred_cls = r.names[top1_idx]
            conf = float(r.probs.top1conf.cpu().numpy())

            y_true.append(cls_name)
            y_pred.append(pred_cls)
            confidences.append(conf)

            orig_w, orig_h = crop_dims_map.get(img_p.name, (128, 128))
            crop_info.append({
                "true_cls": cls_name,
                "pred_cls": pred_cls,
                "confidence": conf,
                "orig_w": orig_w,
                "orig_h": orig_h,
                "path": str(img_p.name)
            })

    total_test = len(y_true)
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    overall_accuracy = correct / total_test if total_test > 0 else 0.0

    # Build Confusion Matrix
    cm = {c_true: {c_pred: 0 for c_pred in CLASSES} for c_true in CLASSES}
    for yt, yp in zip(y_true, y_pred):
        cm[yt][yp] += 1

    # Per-class precision, recall, F1
    per_class_metrics = {}
    precision_list = []
    recall_list = []
    f1_list = []

    for c in CLASSES:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in CLASSES if other != c)
        fn = sum(cm[c][other] for other in CLASSES if other != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_metrics[c] = {
            "test_samples": sum(cm[c].values()),
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

    macro_precision = np.mean(precision_list)
    macro_recall = np.mean(recall_list)
    macro_f1 = np.mean(f1_list)

    print(f"\n1. OVERALL TEST METRICS (N = {total_test}):")
    print(f"  Accuracy:        {overall_accuracy * 100:.2f}%")
    print(f"  Macro Precision: {macro_precision * 100:.2f}%")
    print(f"  Macro Recall:    {macro_recall * 100:.2f}%")
    print(f"  Macro F1-Score:  {macro_f1 * 100:.2f}%")

    print("\n2. CONFUSION MATRIX (Rows: Actual, Cols: Predicted):")
    print(f"{'Actual / Pred':<15} | {'AMBULANCE':<10} | {'FIRE_BRIGADE':<12} | {'POLICE':<8} | {'NORMAL':<8} | {'TOTAL':<6}")
    print("-" * 75)
    for c_true in CLASSES:
        row = cm[c_true]
        tot = sum(row.values())
        print(f"{c_true.upper():<15} | {row['ambulance']:<10} | {row['fire_brigade']:<12} | {row['police']:<8} | {row['normal']:<8} | {tot:<6}")
    print("-" * 75)

    print("\n3. PER-CLASS PERFORMANCE BREAKDOWN:")
    print(f"{'Class':<15} | {'Samples':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 60)
    for c in CLASSES:
        m = per_class_metrics[c]
        print(f"{c.upper():<15} | {m['test_samples']:<8} | {m['precision']*100:<9.1f}% | {m['recall']*100:<9.1f}% | {m['f1']*100:<9.1f}%")
    print("-" * 60)

    # 4. Critical False Positive / Negative Analysis
    normal_fp_amb = cm["normal"]["ambulance"]
    normal_fp_fb = cm["normal"]["fire_brigade"]
    normal_fp_pol = cm["normal"]["police"]
    print(f"\n4. CRITICAL FALSE EMERGENCY CONFUSION FROM 'NORMAL':")
    print(f"  NORMAL -> AMBULANCE:    {normal_fp_amb}")
    print(f"  NORMAL -> FIRE_BRIGADE: {normal_fp_fb}")
    print(f"  NORMAL -> POLICE:       {normal_fp_pol}")
    print(f"  Emergency -> NORMAL (Missed Emergency):")
    for em_c in ["ambulance", "fire_brigade", "police"]:
        print(f"    {em_c.upper()} -> NORMAL: {cm[em_c]['normal']}")

    # 5. Confidence Analysis
    correct_confs = [item["confidence"] for item in crop_info if item["true_cls"] == item["pred_cls"] and item["true_cls"] != "normal"]
    incorrect_confs = [item["confidence"] for item in crop_info if item["true_cls"] != item["pred_cls"] and item["pred_cls"] != "normal"]
    normal_fp_confs = [item["confidence"] for item in crop_info if item["true_cls"] == "normal" and item["pred_cls"] != "normal"]

    print("\n5. PREDICTION CONFIDENCE DISTRIBUTION:")
    if correct_confs:
        print(f"  Correct Emergency Predictions: Mean={np.mean(correct_confs):.3f} | Min={min(correct_confs):.3f} | Max={max(correct_confs):.3f}")
    else:
        print("  Correct Emergency Predictions: N/A")
    if incorrect_confs:
        print(f"  Incorrect Emergency Predictions: Mean={np.mean(incorrect_confs):.3f} | Min={min(incorrect_confs):.3f} | Max={max(incorrect_confs):.3f}")
    if normal_fp_confs:
        print(f"  NORMAL False Positive Confidence: Mean={np.mean(normal_fp_confs):.3f} | Min={min(normal_fp_confs):.3f} | Max={max(normal_fp_confs):.3f}")
    else:
        print("  NORMAL False Positive Confidence: None (0 false emergency alarms from NORMAL test crops)")

    # 6. Small Vehicle Analysis by Scale Groups
    print("\n" + "=" * 75)
    print("STEP 5: SMALL VEHICLE ACCURACY EVALUATION BY SCALE GROUP")
    print("=" * 75)
    scale_bins = [
        ("<32 px", lambda w, h: h < 32),
        ("32–48 px", lambda w, h: 32 <= h < 48),
        ("48–64 px", lambda w, h: 48 <= h < 64),
        ("64–100 px", lambda w, h: 64 <= h < 100),
        (">100 px", lambda w, h: h >= 100)
    ]

    print(f"{'Scale Group':<12} | {'Samples':<8} | {'Correct':<8} | {'Accuracy':<10} | {'Status':<15}")
    print("-" * 65)
    for bin_name, bin_func in scale_bins:
        items = [item for item in crop_info if bin_func(item["orig_w"], item["orig_h"])]
        n_items = len(items)
        if n_items >= 3:
            n_corr = sum(1 for it in items if it["true_cls"] == it["pred_cls"])
            acc = n_corr / n_items * 100.0
            print(f"{bin_name:<12} | {n_items:<8} | {n_corr:<8} | {acc:<9.1f}% | VALIDATED")
        else:
            print(f"{bin_name:<12} | {n_items:<8} | N/A      | N/A        | INSUFFICIENT DATA ({n_items} samples)")
    print("-" * 65)

    return {
        "overall_accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_metrics": per_class_metrics,
        "crop_info": crop_info
    }

def evaluate_real_cctv_false_positives(model):
    print("\n" + "=" * 75)
    print("STEP 6 & 7: REAL CCTV FALSE-POSITIVE EVALUATION & TEMPORAL SIMULATION")
    print("Videos: data/uploads/my_traffic.mp4 & data/uploads/bidirectional.mp4")
    print("=" * 75)

    # Use existing YOLOv8s detector to get vehicle crops
    detector = YOLO("yolov8s.pt")
    
    videos = [
        PROJECT_ROOT / "data" / "uploads" / "my_traffic.mp4",
        PROJECT_ROOT / "data" / "uploads" / "bidirectional.mp4"
    ]

    overall_cctv_results = {}

    for vpath in videos:
        if not vpath.exists():
            continue
        vname = vpath.name
        print(f"\nProcessing Real CCTV Feed: {vname} ...")

        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        
        # Track predictions per track_id across frames
        # track_id -> list of predictions {"frame": idx, "cls": pred_cls, "conf": conf, "is_emergency": bool}
        track_predictions = defaultdict(list)
        total_crops_evaluated = 0
        raw_predictions_count = defaultdict(int)
        emergency_confidences = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Run tracking to simulate realistic video pipeline
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
                    if bw < 25 or bh < 25:
                        continue

                    # Extract crop with 8% padding
                    pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
                    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    cx2, cy2 = min(w_f, x2 + pad_x), min(h_f, y2 + pad_y)
                    crop = frame[cy1:cy2, cx1:cx2]

                    if crop.shape[0] < 20 or crop.shape[1] < 20:
                        continue

                    crop_resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)

                    # Classify crop
                    cls_res = model.predict(source=crop_resized, imgsz=128, device='cpu', verbose=False)
                    top1_idx = int(cls_res[0].probs.top1)
                    pred_cls = cls_res[0].names[top1_idx]
                    conf = float(cls_res[0].probs.top1conf.cpu().numpy())

                    is_emergency = pred_cls in ["ambulance", "fire_brigade", "police"]
                    raw_predictions_count[pred_cls] += 1
                    total_crops_evaluated += 1

                    if is_emergency:
                        emergency_confidences.append(conf)

                    track_predictions[tid].append({
                        "frame": frame_idx,
                        "pred_cls": pred_cls,
                        "conf": conf,
                        "is_emergency": is_emergency,
                        "crop_h": bh
                    })

        cap.release()

        # Report Raw Single-Frame Results
        total_raw_emergency_fp = raw_predictions_count["ambulance"] + raw_predictions_count["fire_brigade"] + raw_predictions_count["police"]
        raw_fp_rate = (total_raw_emergency_fp / total_crops_evaluated * 100.0) if total_crops_evaluated > 0 else 0.0

        print(f"\n--- Raw Single-Frame Predictions on {vname} ---")
        print(f"  Total Vehicle Crops Evaluated: {total_crops_evaluated}")
        print(f"  NORMAL Predictions:            {raw_predictions_count['normal']} ({(raw_predictions_count['normal']/total_crops_evaluated*100):.1f}%)")
        print(f"  AMBULANCE False Alarms:        {raw_predictions_count['ambulance']}")
        print(f"  FIRE_BRIGADE False Alarms:     {raw_predictions_count['fire_brigade']}")
        print(f"  POLICE False Alarms:           {raw_predictions_count['police']}")
        print(f"  Raw Single-Frame FP Rate:      {raw_fp_rate:.2f}%")
        if emergency_confidences:
            print(f"  False Alarm Confidence: Mean={np.mean(emergency_confidences):.3f} | Max={max(emergency_confidences):.3f}")

        # Step 7: Temporal Confirmation Simulation (Offline)
        print(f"\n--- Step 7: Temporal Confirmation Simulation on {vname} ---")
        print(f"  Total Tracked Vehicles: {len(track_predictions)}")
        
        # Test temporal threshold rules: N consecutive frames of emergency prediction
        for consecutive_k in [1, 3, 5, 10]:
            confirmed_tracks = 0
            for tid, preds in track_predictions.items():
                # Check for run of consecutive emergency predictions
                max_consec = 0
                curr_consec = 0
                for p in preds:
                    if p["is_emergency"] and p["conf"] >= 0.60:
                        curr_consec += 1
                        if curr_consec > max_consec:
                            max_consec = curr_consec
                    else:
                        curr_consec = 0
                if max_consec >= consecutive_k:
                    confirmed_tracks += 1

            track_fp_rate = (confirmed_tracks / len(track_predictions) * 100.0) if track_predictions else 0.0
            print(f"  Requirement: {consecutive_k:>2} Consecutive Frames (Conf >= 0.60) -> False Confirmed Tracks: {confirmed_tracks:>3} / {len(track_predictions)} ({track_fp_rate:.1f}%)")

        overall_cctv_results[vname] = {
            "total_crops": total_crops_evaluated,
            "raw_counts": dict(raw_predictions_count),
            "raw_fp_rate": raw_fp_rate,
            "total_tracks": len(track_predictions)
        }

    return overall_cctv_results

def run_full_evaluation():
    if not MODEL_PATH.exists():
        print(f"Model path {MODEL_PATH} not found!")
        return

    print(f"Loading trained classifier from {MODEL_PATH} ...")
    model = YOLO(str(MODEL_PATH))

    test_results = evaluate_held_out_test_set(model)
    cctv_results = evaluate_real_cctv_false_positives(model)

    # Save summary report JSON
    final_report_data = {
        "test_results": {
            "overall_accuracy": test_results["overall_accuracy"],
            "macro_precision": test_results["macro_precision"],
            "macro_recall": test_results["macro_recall"],
            "macro_f1": test_results["macro_f1"],
            "confusion_matrix": test_results["confusion_matrix"],
            "per_class_metrics": test_results["per_class_metrics"]
        },
        "cctv_evaluation": cctv_results
    }

    report_path = REPORTS_DIR / "baseline_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report_data, f, indent=2)

    print(f"\nFull evaluation results saved to {report_path}")

if __name__ == "__main__":
    run_full_evaluation()
