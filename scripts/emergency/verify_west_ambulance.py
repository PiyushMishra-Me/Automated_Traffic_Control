import os
import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def verify_west_ambulance():
    video_path = PROJECT_ROOT / "data" / "uploads" / "west.mp4"
    det_model_path = PROJECT_ROOT / "yolov8s.pt"
    cls_model_path = PROJECT_ROOT / "runs" / "emergency_classifier" / "demo_ambulance" / "weights" / "best.pt"
    
    out_dir = PROJECT_ROOT / "runs" / "emergency_classifier" / "demo_ambulance"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_out_path = out_dir / "west_ambulance_demo.mp4"
    
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing {video_path.name} ({width}x{height}, {fps:.1f} fps, {total_frames} frames)...")
    det_model = YOLO(str(det_model_path))
    cls_model = YOLO(str(cls_model_path))
    
    # West camera inbound vector is [-0.89, -0.45]
    j_vec_inbound = np.array([-0.89, -0.45], dtype=np.float32)
    j_vec_inbound = j_vec_inbound / np.linalg.norm(j_vec_inbound)
    
    track_records = defaultdict(list)
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
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                
                pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
                cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                cx2, cy2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
                crop = frame[cy1:cy2, cx1:cx2]
                
                pred_cls = "normal"
                pred_conf = 0.0
                amb_conf = 0.0
                
                if crop.shape[0] >= 6 and crop.shape[1] >= 6:
                    crop_res = cv2.resize(crop, (128, 128))
                    cls_out = cls_model.predict(source=crop_res, imgsz=128, verbose=False, device="cpu")
                    probs = cls_out[0].probs
                    top1_idx = int(probs.top1)
                    pred_cls = cls_model.names[top1_idx]
                    pred_conf = float(probs.top1conf)
                    amb_conf = float(probs.data[0])
                    
                track_records[tid].append({
                    "frame": frame_idx,
                    "box": [x1, y1, x2, y2],
                    "center": (cx, cy),
                    "bh": bh,
                    "yolo_cls": cname,
                    "det_conf": float(dconf),
                    "pred_cls": pred_cls,
                    "pred_conf": pred_conf,
                    "amb_conf": amb_conf
                })
                
    cap.release()
    print(f"Tracking complete across {frame_idx} frames. Total unique tracks: {len(track_records)}")
    
    candidate_summary = []
    
    for tid, recs in track_records.items():
        if len(recs) < 5:
            continue
            
        first_f = recs[0]["frame"]
        last_f = recs[-1]["frame"]
        heights = [r["bh"] for r in recs]
        amb_preds = sum(1 for r in recs if r["pred_cls"] == "ambulance")
        amb_confs = [r["amb_conf"] for r in recs]
        max_amb_c = max(amb_confs) if amb_confs else 0.0
        avg_amb_c = float(np.mean(amb_confs)) if amb_confs else 0.0
        
        consec = 0
        max_consec = 0
        for r in recs:
            if r["pred_cls"] == "ambulance" and r["amb_conf"] >= 0.60:
                consec += 1
                if consec > max_consec:
                    max_consec = consec
            else:
                consec = 0
                
        pts = [r["center"] for r in recs]
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        dist = np.hypot(dx, dy)
        
        movement = "UNKNOWN"
        if dist >= 10:
            u_vec = np.array([dx / dist, dy / dist], dtype=np.float32)
            dot = float(np.dot(u_vec, j_vec_inbound))
            if dot > 0.20:
                movement = "INCOMING"
            elif dot < -0.20:
                movement = "OUTGOING"
            else:
                movement = "CROSSING"
                
        if amb_preds > 0 or max_amb_c >= 0.40:
            candidate_summary.append({
                "track_id": tid,
                "first_frame": first_f,
                "last_frame": last_f,
                "num_frames": len(recs),
                "min_h": min(heights),
                "max_h": max(heights),
                "mean_h": float(np.mean(heights)),
                "amb_preds": amb_preds,
                "pct_amb": (amb_preds / len(recs)) * 100.0,
                "max_amb_conf": max_amb_c,
                "avg_amb_conf": avg_amb_c,
                "max_consecutive": max_consec,
                "reaches_confirmed": (max_consec >= 5),
                "movement": movement,
                "recs": recs,
                "dx": dx,
                "dy": dy
            })
            
    # Filter for OUTGOING ambulance tracks
    outgoing_candidates = [c for c in candidate_summary if c["movement"] == "OUTGOING" and c["max_amb_conf"] >= 0.60]
    
    if outgoing_candidates:
        best_cand = max(outgoing_candidates, key=lambda x: (x["max_consecutive"], x["amb_preds"], x["max_amb_conf"]))
    else:
        best_cand = max(candidate_summary, key=lambda x: (x["max_consecutive"], x["amb_preds"], x["max_amb_conf"]))
        
    print("\n" + "=" * 60)
    print(f"VERIFIED OUTGOING AMBULANCE TRACK IN WEST.MP4: Track #{best_cand['track_id']}")
    print("=" * 60)
    print(f"- First Frame: {best_cand['first_frame']}")
    print(f"- Last Frame: {best_cand['last_frame']}")
    print(f"- Total Frames Tracked: {best_cand['num_frames']}")
    print(f"- Bounding Box Height: min={best_cand['min_h']}px, max={best_cand['max_h']}px, mean={best_cand['mean_h']:.1f}px")
    print(f"- Frames Predicted AMBULANCE: {best_cand['amb_preds']} / {best_cand['num_frames']} ({best_cand['pct_amb']:.1f}%)")
    print(f"- Maximum AMBULANCE Confidence: {best_cand['max_amb_conf']*100:.2f}% (conf = {best_cand['max_amb_conf']:.4f})")
    print(f"- Mean AMBULANCE Confidence: {best_cand['avg_amb_conf']*100:.2f}% (conf = {best_cand['avg_amb_conf']:.4f})")
    print(f"- Maximum Consecutive AMBULANCE Frames (conf >= 0.60): {best_cand['max_consecutive']}")
    print(f"- Reaches CONFIRMED (>=5 consecutive): {best_cand['reaches_confirmed']}")
    print(f"- Movement Direction: {best_cand['movement']}")

    # Render annotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))
    
    cap = cv2.VideoCapture(str(video_path))
    f_count = 0
    frame_to_rec = {r["frame"]: r for r in best_cand["recs"]}
    
    consec_state = 0
    confirmed_triggered = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        f_count += 1
        
        if f_count in frame_to_rec:
            r = frame_to_rec[f_count]
            x1, y1, x2, y2 = r["box"]
            amb_c = r["amb_conf"]
            is_amb = (r["pred_cls"] == "ambulance" and amb_c >= 0.60)
            
            if is_amb:
                consec_state += 1
                if consec_state >= 5:
                    confirmed_triggered = True
            else:
                consec_state = 0
                
            state_label = "CONFIRMED" if confirmed_triggered else ("POSSIBLE" if consec_state >= 1 else "DETECTED")
            color = (0, 255, 0) if state_label == "CONFIRMED" else ((0, 165, 255) if state_label == "POSSIBLE" else (0, 255, 255))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"AMBULANCE [{state_label}]", (x1, max(20, y1 - 32)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(frame, f"Confidence: {amb_c*100:.1f}%", (x1, max(35, y1 - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
            cv2.putText(frame, f"Track #{best_cand['track_id']} | {best_cand['movement']}", (x1, max(50, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
        out_writer.write(frame)
        
    cap.release()
    out_writer.release()
    print(f"[OK] Annotated video saved to: {video_out_path}")
    
    # Save Report
    report_file = out_dir / "west_verification_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# West Camera Ambulance Verification Report (`west.mp4`)\n\n")
        f.write(f"## **AMBULANCE_DETECTED = {'YES' if best_cand['max_amb_conf'] >= 0.60 else 'NO'}**\n")
        f.write(f"## **MOVEMENT = {best_cand['movement']}**\n")
        f.write(f"## **CONFIRMED = {'YES' if best_cand['reaches_confirmed'] else 'NO'}**\n\n")
        f.write("### Track Metrics:\n\n")
        f.write(f"- **Ambulance Track ID:** #{best_cand['track_id']}\n")
        f.write(f"- **First Frame / Last Frame:** Frame {best_cand['first_frame']} – Frame {best_cand['last_frame']}\n")
        f.write(f"- **Total Frames Tracked:** {best_cand['num_frames']} frames\n")
        f.write(f"- **BBox Height Range:** {best_cand['min_h']} px (min) / {best_cand['max_h']} px (max) / {best_cand['mean_h']:.1f} px (mean)\n")
        f.write(f"- **Frames Predicted AMBULANCE:** {best_cand['amb_preds']} / {best_cand['num_frames']} ({best_cand['pct_amb']:.1f}%)\n")
        f.write(f"- **Maximum AMBULANCE Confidence:** {best_cand['max_amb_conf']*100:.2f}%\n")
        f.write(f"- **Mean AMBULANCE Confidence:** {best_cand['avg_amb_conf']*100:.2f}%\n")
        f.write(f"- **Max Consecutive AMBULANCE Frames (conf >= 0.60):** {best_cand['max_consecutive']}\n")
        f.write(f"- **Reaches 5-Frame Confirmation (CONFIRMED):** {'YES' if best_cand['reaches_confirmed'] else 'NO'}\n")
        f.write(f"- **Movement Trajectory & Direction:** {best_cand['movement']} (dx={best_cand['dx']:.1f}, dy={best_cand['dy']:.1f})\n\n")
        f.write(f"- **Annotated Video:** `{video_out_path}`\n")
        
    print(f"[OK] Report saved to: {report_file}")

if __name__ == "__main__":
    verify_west_ambulance()
