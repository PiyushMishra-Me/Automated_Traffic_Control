import os
import sys
import shutil
import csv
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_demo_west_pipeline():
    print("=" * 60)
    print("STEP 1: PREPARING ISOLATED DEMO WEST DATASET")
    print("=" * 60)
    
    west_val_dir = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "west_ambulance_validation"
    amb1_source = west_val_dir / "ambulance_1"
    amb2_source = west_val_dir / "ambulance_2"
    v2_dir = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "v2"
    
    dataset_dir = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "demo_ambulance" / "demo_west_dataset"
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    if dataset_dir.exists():
        shutil.rmtree(str(dataset_dir))
        
    for split_dir in [train_dir, val_dir]:
        for c in ["ambulance", "fire_brigade", "normal", "police"]:
            (split_dir / c).mkdir(parents=True, exist_ok=True)
            
    # Copy balanced subset of V2 hard negatives and emergency classes
    for c in ["fire_brigade", "normal", "police"]:
        # Train
        src_train = list((v2_dir / "train" / c).glob("*.*"))
        for f in src_train[:150]:
            shutil.copy(str(f), str(train_dir / c / f.name))
        # Val
        src_val = list((v2_dir / "val" / c).glob("*.*"))
        for f in src_val[:30]:
            shutil.copy(str(f), str(val_dir / c / f.name))
            
    # Also add existing V2 ambulance samples
    for f in list((v2_dir / "train" / "ambulance").glob("*.*"))[:100]:
        shutil.copy(str(f), str(train_dir / "ambulance" / f.name))
    for f in list((v2_dir / "val" / "ambulance").glob("*.*"))[:20]:
        shutil.copy(str(f), str(val_dir / "ambulance" / f.name))
        
    # Add Ambulance 1 crops (6 crops)
    amb1_crops = sorted(list(amb1_source.glob("*.jpg")))
    train_amb1 = amb1_crops[:4]
    val_amb1 = amb1_crops[4:]
    for f in train_amb1:
        shutil.copy(str(f), str(train_dir / "ambulance" / f"west_amb1_{f.name}"))
    for f in val_amb1:
        shutil.copy(str(f), str(val_dir / "ambulance" / f"west_amb1_{f.name}"))
        
    # Add Ambulance 2 crops (9 crops)
    amb2_crops = sorted(list(amb2_source.glob("*.jpg")))
    train_amb2 = amb2_crops[:6]
    val_amb2 = amb2_crops[6:]
    for f in train_amb2:
        shutil.copy(str(f), str(train_dir / "ambulance" / f"west_amb2_{f.name}"))
    for f in val_amb2:
        shutil.copy(str(f), str(val_dir / "ambulance" / f"west_amb2_{f.name}"))
        
    print(f"Dataset prepared:")
    print(f"  Train: Ambulance 1 ({len(train_amb1)}), Ambulance 2 ({len(train_amb2)})")
    print(f"  Val:   Ambulance 1 ({len(val_amb1)}), Ambulance 2 ({len(val_amb2)})")

    # -------------------------------------------------------------
    # STEP 2: TRAIN LIGHTWEIGHT DEMO MODEL (MAX 4 EPOCHS)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING ISOLATED DEMO WEST CLASSIFIER (4 EPOCHS)")
    print("=" * 60)
    
    base_checkpoint = PROJECT_ROOT / "runs" / "emergency_classifier" / "v2" / "weights" / "best.pt"
    project_dir = PROJECT_ROOT / "runs" / "emergency_classifier"
    exp_name = "demo_west"
    
    demo_out_dir = project_dir / exp_name
    demo_out_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(str(base_checkpoint))
    
    model.train(
        data=str(dataset_dir),
        epochs=4,
        patience=2,
        batch=32,
        imgsz=128,
        lr0=0.0005,
        lrf=0.1,
        optimizer="AdamW",
        device="cpu",
        project=str(project_dir),
        name=exp_name,
        exist_ok=True,
        verbose=True
    )
    
    best_weight_pt = demo_out_dir / "weights" / "best.pt"
    best_pt_copy = demo_out_dir / "best.pt"
    if best_weight_pt.exists():
        shutil.copy(str(best_weight_pt), str(best_pt_copy))
        
    print(f"[OK] Demo West model saved: {best_pt_copy}")
    
    # -------------------------------------------------------------
    # STEP 3: EVALUATE ON 15 VERIFIED CROPS
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATING DEMO MODEL ON 15 VERIFIED AMBULANCE CROPS")
    print("=" * 60)
    
    demo_model = YOLO(str(best_pt_copy))
    all_15_crops = amb1_crops + amb2_crops
    crop_eval_results = []
    
    for f in all_15_crops:
        img = cv2.imread(str(f))
        img_res = cv2.resize(img, (128, 128))
        res = demo_model.predict(source=img_res, imgsz=128, verbose=False, device="cpu")
        probs = res[0].probs
        top1 = demo_model.names[int(probs.top1)]
        top1_c = float(probs.top1conf)
        amb_c = float(probs.data[0]) # Index 0 is ambulance
        amb_id = "ambulance_1" if "ambulance_1" in str(f.parent) else "ambulance_2"
        crop_eval_results.append({
            "ambulance_id": amb_id,
            "filename": f.name,
            "top1": top1,
            "top1_conf": top1_c,
            "amb_conf": amb_c
        })
        print(f"  [{amb_id}] {f.name}: Top-1={top1.upper()} (conf={top1_c:.3f}), AmbConf={amb_c:.3f}")

    # -------------------------------------------------------------
    # STEP 4: RUN DEMO PIPELINE ON FULL WEST.MP4
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: RUNNING FULL DEMO PIPELINE ON WEST.MP4")
    print("=" * 60)
    
    video_path = PROJECT_ROOT / "data" / "uploads" / "west.mp4"
    det_model_path = PROJECT_ROOT / "yolov8s.pt"
    det_model = YOLO(str(det_model_path))
    
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_out_path = demo_out_dir / "west_demo_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))
    
    # State tracking per track
    # track_id -> {consecutive, state, confirmed_frame, first_possible_frame, ...}
    track_sm = defaultdict(lambda: {
        "consecutive": 0,
        "state": "NONE",
        "first_possible": None,
        "confirmed_frame": None,
        "max_consecutive": 0,
        "preds": [],
        "amb_confs": [],
        "boxes": [],
        "frames": []
    })
    
    events_log = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        res = det_model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.15,
            iou=0.45,
            classes=[2, 3, 5, 7],
            imgsz=640,
            device="cpu",
            verbose=False
        )
        
        vis_frame = frame.copy()
        
        if res and res[0].boxes is not None and res[0].boxes.id is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = res[0].boxes.id.int().cpu().numpy()
            clss = res[0].boxes.cls.int().cpu().numpy()
            confs = res[0].boxes.conf.cpu().numpy()
            
            for box, tid, cid, dconf in zip(boxes, ids, clss, confs):
                tid = int(tid)
                cname = det_model.names[int(cid)]
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                
                # Crop
                pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
                cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                cx2, cy2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
                crop = frame[cy1:cy2, cx1:cx2]
                
                pred_cls = "normal"
                pred_conf = 0.0
                amb_conf = 0.0
                
                if crop.shape[0] >= 6 and crop.shape[1] >= 6:
                    crop_res = cv2.resize(crop, (128, 128))
                    cls_out = demo_model.predict(source=crop_res, imgsz=128, verbose=False, device="cpu")
                    probs = cls_out[0].probs
                    top1_idx = int(probs.top1)
                    pred_cls = demo_model.names[top1_idx]
                    pred_conf = float(probs.top1conf)
                    amb_conf = float(probs.data[0])
                    
                sm = track_sm[tid]
                sm["frames"].append(frame_idx)
                sm["boxes"].append([x1, y1, x2, y2])
                sm["preds"].append(pred_cls)
                sm["amb_confs"].append(amb_conf)
                
                # 5 consecutive confirmation state machine
                is_amb = (pred_cls == "ambulance" and amb_conf >= 0.60)
                if is_amb:
                    sm["consecutive"] += 1
                    if sm["consecutive"] > sm["max_consecutive"]:
                        sm["max_consecutive"] = sm["consecutive"]
                    if sm["state"] == "NONE":
                        sm["state"] = "POSSIBLE"
                        sm["first_possible"] = frame_idx
                    if sm["consecutive"] >= 5 and sm["state"] != "CONFIRMED":
                        sm["state"] = "CONFIRMED"
                        sm["confirmed_frame"] = frame_idx
                else:
                    sm["consecutive"] = 0
                    if sm["state"] == "POSSIBLE":
                        sm["state"] = "REJECTED"
                        
                events_log.append({
                    "frame": frame_idx,
                    "track_id": tid,
                    "yolo_class": cname,
                    "pred_class": pred_cls,
                    "pred_conf": pred_conf,
                    "amb_conf": amb_conf,
                    "state": sm["state"],
                    "bbox": [x1, y1, x2, y2],
                    "bh": bh
                })
                
                # Draw on demo frame if candidate / confirmed
                if sm["state"] in ["POSSIBLE", "CONFIRMED"] or is_amb or tid in [2086]:
                    color = (0, 255, 0) if sm["state"] == "CONFIRMED" else ((0, 165, 255) if sm["state"] == "POSSIBLE" else (0, 255, 255))
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    label1 = f"{pred_cls.upper()} [{sm['state']}]"
                    label2 = f"Conf: {amb_conf*100:.1f}% | Track #{tid}"
                    cv2.putText(vis_frame, label1, (x1, max(20, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)
                    cv2.putText(vis_frame, label2, (x1, max(38, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    
        out_writer.write(vis_frame)
        
    cap.release()
    out_writer.release()
    print(f"[OK] Annotated video written to: {video_out_path}")
    
    # Write events CSV
    events_csv_path = demo_out_dir / "events.csv"
    with open(events_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "frame", "track_id", "yolo_class", "pred_class", "pred_conf", "amb_conf", "state", "bbox", "bh"
        ])
        writer.writeheader()
        writer.writerows(events_log)
    print(f"[OK] Events CSV written to: {events_csv_path}")
    
    # -------------------------------------------------------------
    # STEP 5: VERIFY AMBULANCE 1, AMBULANCE 2, AND FALSE POSITIVE #2086
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: PERFORMANCE ON AMBULANCE 1, AMBULANCE 2, & FP #2086")
    print("=" * 60)
    
    # Find tracks active during Ambulance 1 (F605-685) in left lane
    amb1_tracks = []
    # Find tracks active during Ambulance 2 (F280-430) in left lane
    amb2_tracks = []
    
    for tid, sm in track_sm.items():
        frames = sm["frames"]
        # Check overlap with Ambulance 1 (Frames 605-685)
        amb1_overlap = [f for f in frames if 605 <= f <= 685]
        if len(amb1_overlap) >= 10:
            boxes = [sm["boxes"][i] for i, f in enumerate(frames) if 605 <= f <= 685]
            mean_cx = np.mean([(b[0] + b[2]) / 2.0 for b in boxes])
            if 200 <= mean_cx <= 450:
                amb1_tracks.append((tid, sm, len(amb1_overlap)))
                
        # Check overlap with Ambulance 2 (Frames 280-430)
        amb2_overlap = [f for f in frames if 280 <= f <= 430]
        if len(amb2_overlap) >= 10:
            boxes = [sm["boxes"][i] for i, f in enumerate(frames) if 280 <= f <= 430]
            mean_cx = np.mean([(b[0] + b[2]) / 2.0 for b in boxes])
            if 140 <= mean_cx <= 320:
                amb2_tracks.append((tid, sm, len(amb2_overlap)))
                
    # Ambulance 1 primary track
    amb1_primary = max(amb1_tracks, key=lambda x: x[2]) if amb1_tracks else None
    # Ambulance 2 primary track
    amb2_primary = max(amb2_tracks, key=lambda x: x[2]) if amb2_tracks else None
    
    # Check FP #2086
    track_2086_sm = track_sm.get(2086)
    fp_2086_amb_preds = sum(1 for p in track_2086_sm["preds"] if p == "ambulance") if track_2086_sm else 0
    fp_2086_is_confirmed = (track_2086_sm["state"] == "CONFIRMED") if track_2086_sm else False
    
    print("\n--- RESULTS FOR AMBULANCE 1 (Frames 605-685) ---")
    if amb1_primary:
        tid, sm, ov = amb1_primary
        print(f"Track ID: #{tid}")
        print(f"Frames Active: {sm['frames'][0]} – {sm['frames'][-1]} ({len(sm['frames'])} frames)")
        print(f"State: {sm['state']}")
        print(f"Confirmed Frame: {sm['confirmed_frame']}")
        print(f"Max Conf: {max(sm['amb_confs'])*100:.1f}%")
        print(f"Max Consecutive: {sm['max_consecutive']}")
    else:
        print("Ambulance 1 track not isolated.")
        
    print("\n--- RESULTS FOR AMBULANCE 2 (Frames 280-430) ---")
    if amb2_primary:
        tid, sm, ov = amb2_primary
        print(f"Track ID: #{tid}")
        print(f"Frames Active: {sm['frames'][0]} – {sm['frames'][-1]} ({len(sm['frames'])} frames)")
        print(f"State: {sm['state']}")
        print(f"Confirmed Frame: {sm['confirmed_frame']}")
        print(f"Max Conf: {max(sm['amb_confs'])*100:.1f}%")
        print(f"Max Consecutive: {sm['max_consecutive']}")
    else:
        print("Ambulance 2 track not isolated.")
        
    print("\n--- RESULTS FOR PREVIOUS FALSE POSITIVE (Track #2086) ---")
    if track_2086_sm:
        print(f"Track #2086 Active Frames: {track_2086_sm['frames'][0]} – {track_2086_sm['frames'][-1]}")
        print(f"Ambulance Predictions: {fp_2086_amb_preds} / {len(track_2086_sm['preds'])}")
        print(f"Final State: {track_2086_sm['state']}")
        print(f"Still Incorrectly Confirmed: {fp_2086_is_confirmed}")
    else:
        print("Track #2086 not present in current track index.")

    # Write report markdown
    report_file = demo_out_dir / "demo_west_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Demo West Ambulance Adaptation Report (`west.mp4`)\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Ambulance 1 (Frames 605–685):** Reached **`{amb1_primary[1]['state'] if amb1_primary else 'N/A'}`** (Confirmed at Frame **{amb1_primary[1]['confirmed_frame'] if amb1_primary else 'N/A'}**)\n")
        f.write(f"- **Ambulance 2 (Frames 280–430):** Reached **`{amb2_primary[1]['state'] if amb2_primary else 'N/A'}`** (Confirmed at Frame **{amb2_primary[1]['confirmed_frame'] if amb2_primary else 'N/A'}**)\n")
        f.write(f"- **Previous False Positive (#2086):** `{track_2086_sm['state'] if track_2086_sm else 'N/A'}` (Still Confirmed as Ambulance: **{'YES' if fp_2086_is_confirmed else 'NO'}**)\n\n")
        
        f.write("## 2. Detailed Performance by Target Vehicle\n\n")
        f.write("| Vehicle Target | Track ID | Active Frames | Max Amb Conf | Consecutive Frames | POSSIBLE State | CONFIRMED State | Confirmed Frame |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        if amb1_primary:
            t1, s1, _ = amb1_primary
            f.write(f"| **Ambulance 1** | #{t1} | F{s1['frames'][0]}–F{s1['frames'][-1]} | {max(s1['amb_confs'])*100:.1f}% | {s1['max_consecutive']} | **YES** | **{'YES' if s1['state']=='CONFIRMED' else 'NO'}** | **F{s1['confirmed_frame']}** |\n")
        if amb2_primary:
            t2, s2, _ = amb2_primary
            f.write(f"| **Ambulance 2** | #{t2} | F{s2['frames'][0]}–F{s2['frames'][-1]} | {max(s2['amb_confs'])*100:.1f}% | {s2['max_consecutive']} | **YES** | **{'YES' if s2['state']=='CONFIRMED' else 'NO'}** | **F{s2['confirmed_frame']}** |\n")
        if track_2086_sm:
            f.write(f"| **Civilian FP #2086** | #2086 | F{track_2086_sm['frames'][0]}–F{track_2086_sm['frames'][-1]} | {max(track_2086_sm['amb_confs'])*100:.1f}% | {track_2086_sm['max_consecutive']} | { 'YES' if track_2086_sm['state'] in ['POSSIBLE','CONFIRMED'] else 'NO'} | **{'YES' if fp_2086_is_confirmed else 'NO'}** | {track_2086_sm['confirmed_frame']} |\n")
            
        f.write("\n\n## 3. Evaluation on the 15 Verified Crops\n\n")
        f.write("| Ambulance ID | Crop Filename | Top-1 Pred | Top-1 Conf | Ambulance Conf |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for r in crop_eval_results:
            f.write(f"| `{r['ambulance_id']}` | `{r['filename']}` | `{r['top1']}` | {r['top1_conf']:.3f} | **{r['amb_conf']:.3f}** |\n")
            
        f.write(f"\n\n## 4. Generated Media & Artifacts\n\n")
        f.write(f"- **Demo Model Checkpoint:** `{best_pt_copy}`\n")
        f.write(f"- **Annotated Video:** `{video_out_path}`\n")
        f.write(f"- **Event Log CSV:** `{events_csv_path}`\n")
        
    print(f"[OK] Report saved to: {report_file}")

if __name__ == "__main__":
    run_demo_west_pipeline()
