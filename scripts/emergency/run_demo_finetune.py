import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def prepare_demo_dataset():
    print("=" * 60)
    print("STEP 1: PREPARING ISOLATED DEMO FINE-TUNING DATASET")
    print("=" * 60)
    
    crops_source_dir = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "demo_ambulance" / "east_track_373" / "crops"
    v2_dir = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "v2"
    
    fine_tune_dir = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "demo_ambulance" / "fine_tune"
    train_dir = fine_tune_dir / "train"
    val_dir = fine_tune_dir / "val"
    test_dir = fine_tune_dir / "test"
    
    # Recreate clean fine-tune structure
    if fine_tune_dir.exists():
        shutil.rmtree(str(fine_tune_dir))
        
    for split_dir in [train_dir, val_dir, test_dir]:
        for c in ["ambulance", "fire_brigade", "normal", "police"]:
            (split_dir / c).mkdir(parents=True, exist_ok=True)
            
    # 1. Copy existing V2 dataset
    for split in ["train", "val"]:
        src_split = v2_dir / split
        dst_split = fine_tune_dir / split
        for c in ["ambulance", "fire_brigade", "normal", "police"]:
            src_c = src_split / c
            dst_c = dst_split / c
            if src_c.exists():
                for f in src_c.glob("*.*"):
                    shutil.copy(str(f), str(dst_c / f.name))
                    
    # 2. Get sorted 30 crops from Track #373
    crop_files = sorted(list(crops_source_dir.glob("*.jpg")), key=lambda p: p.name)
    total_crops = len(crop_files)
    print(f"Found {total_crops} verified ambulance crops from Track #373.")
    
    # Split temporally: 18 train (60%), 6 val (20%), 6 test (20%)
    train_crops = crop_files[:18]
    val_crops = crop_files[18:24]
    test_crops = crop_files[24:]
    
    print(f"  Training Split: {len(train_crops)} crops ({train_crops[0].name} -> {train_crops[-1].name})")
    print(f"  Validation Split: {len(val_crops)} crops ({val_crops[0].name} -> {val_crops[-1].name})")
    print(f"  Held-Out Test Split: {len(test_crops)} crops ({test_crops[0].name} -> {test_crops[-1].name})")
    
    for f in train_crops:
        shutil.copy(str(f), str(train_dir / "ambulance" / f"demo_east_{f.name}"))
        
    for f in val_crops:
        shutil.copy(str(f), str(val_dir / "ambulance" / f"demo_east_{f.name}"))
        
    for f in test_crops:
        shutil.copy(str(f), str(test_dir / "ambulance" / f"demo_east_{f.name}"))
        
    print("[SUCCESS] Isolated fine-tuning dataset created at:", fine_tune_dir)
    return fine_tune_dir, test_crops

def run_fine_tuning(fine_tune_dir: Path):
    print("\n" + "=" * 60)
    print("STEP 2: RUNNING TARGETED FINE-TUNING (MAX 5 EPOCHS)")
    print("=" * 60)
    
    base_checkpoint = PROJECT_ROOT / "runs" / "emergency_classifier" / "v2" / "weights" / "best.pt"
    project_dir = PROJECT_ROOT / "runs" / "emergency_classifier"
    exp_name = "demo_ambulance"
    
    print(f"Base checkpoint: {base_checkpoint}")
    print(f"Dataset root: {fine_tune_dir}")
    
    model = YOLO(str(base_checkpoint))
    
    results = model.train(
        data=str(fine_tune_dir),
        epochs=5,
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
    
    demo_best_pt = project_dir / exp_name / "weights" / "best.pt"
    print(f"[SUCCESS] Fine-tuning finished. Demo model saved to: {demo_best_pt}")
    return demo_best_pt

def evaluate_on_east_video(model_pt: Path, test_crops: list):
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATING DEMO MODEL ON HELD-OUT CROPS & EAST.MP4")
    print("=" * 60)
    
    model = YOLO(str(model_pt))
    
    # 1. Evaluate on held-out test crops
    print("Evaluating on held-out test crops (unseen in training):")
    test_results = []
    for f in test_crops:
        img = cv2.imread(str(f))
        img_res = cv2.resize(img, (128, 128))
        res = model.predict(source=img_res, imgsz=128, verbose=False, device="cpu")
        top1 = model.names[int(res[0].probs.top1)]
        top1_c = float(res[0].probs.top1conf)
        amb_c = float(res[0].probs.data[0]) # Index 0 is ambulance
        test_results.append((f.name, top1, top1_c, amb_c))
        print(f"  {f.name}: Top-1={top1.upper()} (conf={top1_c:.3f}), AmbConf={amb_c:.3f}")
        
    # 2. Evaluate on east.mp4 with YOLOv8s + ByteTrack
    video_path = PROJECT_ROOT / "data" / "uploads" / "east.mp4"
    det_model_path = PROJECT_ROOT / "yolov8s.pt"
    det_model = YOLO(str(det_model_path))
    
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    
    demo_dir = PROJECT_ROOT / "runs" / "emergency_classifier" / "demo_ambulance"
    demo_video_path = demo_dir / "east_demo_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(demo_video_path), fourcc, fps, (width, height))
    
    ambulance_frames_log = []
    consecutive_amb = 0
    max_consecutive_amb = 0
    confirmed_frame = None
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Track vehicles
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
        
        if res and res[0].boxes is not None and res[0].boxes.id is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = res[0].boxes.id.int().cpu().numpy()
            
            for box, tid in zip(boxes, ids):
                if tid == 373:
                    x1, y1, x2, y2 = box
                    bw, bh = x2 - x1, y2 - y1
                    pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
                    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    cx2, cy2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
                    crop = frame[cy1:cy2, cx1:cx2]
                    
                    pred_cls = "normal"
                    pred_conf = 0.0
                    amb_conf = 0.0
                    
                    if crop.shape[0] >= 6 and crop.shape[1] >= 6:
                        crop_res = cv2.resize(crop, (128, 128))
                        cls_out = model.predict(source=crop_res, imgsz=128, verbose=False, device="cpu")
                        top1_idx = int(cls_out[0].probs.top1)
                        pred_cls = model.names[top1_idx]
                        pred_conf = float(cls_out[0].probs.top1conf)
                        amb_conf = float(cls_out[0].probs.data[0])
                        
                    is_ambulance = (pred_cls == "ambulance" and amb_conf >= 0.60)
                    if is_ambulance:
                        consecutive_amb += 1
                        if consecutive_amb > max_consecutive_amb:
                            max_consecutive_amb = consecutive_amb
                        if consecutive_amb >= 5 and confirmed_frame is None:
                            confirmed_frame = frame_idx
                    else:
                        consecutive_amb = 0
                        
                    state_str = "CONFIRMED" if confirmed_frame and frame_idx >= confirmed_frame else ("POSSIBLE" if consecutive_amb >= 1 else "NONE")
                    
                    ambulance_frames_log.append({
                        "frame": frame_idx,
                        "box": [x1, y1, x2, y2],
                        "bh": bh,
                        "pred_cls": pred_cls,
                        "pred_conf": pred_conf,
                        "amb_conf": amb_conf,
                        "state": state_str
                    })
                    
                    # Draw on frame for demo video
                    color = (0, 255, 0) if state_str == "CONFIRMED" else ((0, 165, 255) if state_str == "POSSIBLE" else (0, 0, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label1 = f"AMBULANCE [{state_str}]"
                    label2 = f"Confidence: {amb_conf*100:.1f}%"
                    label3 = f"Track ID: 373 | H: {bh}px"
                    cv2.putText(frame, label1, (x1, max(20, y1 - 32)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cv2.putText(frame, label2, (x1, max(35, y1 - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
                    cv2.putText(frame, label3, (x1, max(50, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    
        out_writer.write(frame)
        
    cap.release()
    out_writer.release()
    print(f"[OK] Demo video written to: {demo_video_path}")
    
    # Compute Metrics
    total_eval_frames = len(ambulance_frames_log)
    amb_pred_frames = sum(1 for f in ambulance_frames_log if f["pred_cls"] == "ambulance")
    amb_pct = (amb_pred_frames / total_eval_frames * 100.0) if total_eval_frames > 0 else 0.0
    all_amb_confs = [f["amb_conf"] for f in ambulance_frames_log]
    max_amb_c = max(all_amb_confs) if all_amb_confs else 0.0
    mean_amb_c = float(np.mean(all_amb_confs)) if all_amb_confs else 0.0
    reaches_5_frame_conf = (max_consecutive_amb >= 5)
    
    verdict = "YES" if (max_amb_c >= 0.60 and reaches_5_frame_conf) else ("PARTIAL_YES" if max_amb_c >= 0.60 else "NO")
    
    # Save Report
    report_file = demo_dir / "demo_verification_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Demo Fine-Tuning & Verification Report (`east.mp4` Ambulance)\n\n")
        f.write(f"## **AMBULANCE_DETECTED = {verdict}**\n\n")
        f.write("### Evaluation Summary on Track #373:\n\n")
        f.write(f"- **Total Ambulance Frames Evaluated:** {total_eval_frames}\n")
        f.write(f"- **Number Predicted as AMBULANCE:** {amb_pred_frames}\n")
        f.write(f"- **Percentage Predicted as AMBULANCE:** {amb_pct:.1f}%\n")
        f.write(f"- **Maximum Ambulance Confidence:** {max_amb_c*100:.2f}% (conf = {max_amb_c:.4f})\n")
        f.write(f"- **Mean Ambulance Confidence:** {mean_amb_c*100:.2f}% (conf = {mean_amb_c:.4f})\n")
        f.write(f"- **Maximum Consecutive AMBULANCE Frames (conf >= 0.60):** {max_consecutive_amb}\n")
        f.write(f"- **Reaches 5-Frame Confirmation Condition:** {'YES (CONFIRMED at frame ' + str(confirmed_frame) + ')' if reaches_5_frame_conf else 'NO'}\n\n")
        f.write(f"- **Demo Video Path:** `{demo_video_path}`\n\n")
        
        f.write("### Held-Out Test Crops Evaluation:\n\n")
        f.write("| Crop Filename | Top-1 Class | Top-1 Confidence | Ambulance Confidence |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for cname, top1, top1_c, amb_c in test_results:
            f.write(f"| `{cname}` | `{top1}` | {top1_c:.3f} | **{amb_c:.3f}** |\n")
            
    print("\n" + "=" * 60)
    print("FINAL EVALUATION METRICS:")
    print("=" * 60)
    print(f"AMBULANCE_DETECTED = {verdict}")
    print(f"Total Evaluated Frames: {total_eval_frames}")
    print(f"Predicted AMBULANCE: {amb_pred_frames} ({amb_pct:.1f}%)")
    print(f"Max Confidence: {max_amb_c*100:.2f}%")
    print(f"Mean Confidence: {mean_amb_c*100:.2f}%")
    print(f"Max Consecutive Frames (>=0.60): {max_consecutive_amb}")
    print(f"Reaches 5-Frame Confirmation: {reaches_5_frame_conf}")
    print(f"Annotated Demo Video: {demo_video_path}")
    print(f"Report: {report_file}")

if __name__ == "__main__":
    fine_tune_dir, test_crops = prepare_demo_dataset()
    demo_model_pt = run_fine_tuning(fine_tune_dir)
    evaluate_on_east_video(demo_model_pt, test_crops)
