import os
import sys
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings

def inspect_four_cameras():
    video_dir = PROJECT_ROOT / "data" / "uploads"
    videos = {
        "NORTH": video_dir / "north.mp4",
        "SOUTH": video_dir / "south.mp4",
        "EAST": video_dir / "east.mp4",
        "WEST": video_dir / "west.mp4"
    }
    
    cls_model_path = PROJECT_ROOT / "runs" / "emergency_classifier" / "v2" / "weights" / "best.pt"
    det_model_path = PROJECT_ROOT / "yolov8s.pt"
    
    # Verify files
    print("=" * 60)
    print("STEP 1: VERIFYING REQUIRED FILES")
    print("=" * 60)
    for name, vpath in videos.items():
        if not vpath.exists():
            raise FileNotFoundError(f"Video missing: {vpath}")
        print(f"  [OK] {name} video: {vpath.name} ({vpath.stat().st_size / 1e6:.2f} MB)")
    
    if not cls_model_path.exists():
        raise FileNotFoundError(f"Emergency classifier missing: {cls_model_path}")
    print(f"  [OK] Classifier model: {cls_model_path.name}")
    
    if not det_model_path.exists():
        raise FileNotFoundError(f"YOLO detector missing: {det_model_path}")
    print(f"  [OK] YOLO detector model: {det_model_path.name}")
    
    print("\nLoading models on CPU...")
    det_model = YOLO(str(det_model_path))
    cls_model = YOLO(str(cls_model_path))
    
    reports = {}
    
    print("=" * 60)
    print("STEP 2: RUNNING BOUNDED INSPECTION (STEP = 10 FRAMES)")
    print("=" * 60)
    
    for approach_name, video_path in videos.items():
        print(f"\nProcessing {approach_name} stream ({video_path.name})...")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
        
        step = 10  # Sample every 10th frame for bounded fast inspection
        inspected_frames_count = 0
        total_detections_count = 0
        
        # Tracking & Emergency records
        tracks_trajectories = defaultdict(list)
        tracks_classes = defaultdict(list)
        tracks_emergency_preds = defaultdict(list)
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            
            if (frame_idx - 1) % step != 0:
                continue
                
            inspected_frames_count += 1
            
            # Run YOLO + ByteTrack
            results = det_model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.20,
                iou=0.45,
                classes=[2, 3, 5, 7],
                imgsz=640,
                device="cpu",
                verbose=False
            )
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                total_detections_count += len(boxes)
                
                if boxes.id is not None:
                    xyxy_arr = boxes.xyxy.cpu().numpy()
                    ids_arr = boxes.id.int().cpu().numpy()
                    cls_arr = boxes.cls.int().cpu().numpy()
                    conf_arr = boxes.conf.cpu().numpy()
                    
                    for box, tid, c_id, det_conf in zip(xyxy_arr, ids_arr, cls_arr, conf_arr):
                        tid = int(tid)
                        c_name = det_model.names[int(c_id)]
                        x1, y1, x2, y2 = map(int, box)
                        bw, bh = x2 - x1, y2 - y1
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        
                        tracks_trajectories[tid].append((frame_idx, cx, cy))
                        tracks_classes[tid].append(c_name)
                        
                        # Emergency check on crop if bh >= 48
                        if bh >= 48 and bw > 10 and bh > 10:
                            pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
                            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                            cx2, cy2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
                            crop = frame[cy1:cy2, cx1:cx2]
                            
                            if crop.shape[0] >= 10 and crop.shape[1] >= 10:
                                crop_res = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
                                cls_out = cls_model.predict(source=crop_res, imgsz=128, verbose=False, device="cpu")
                                top1_idx = int(cls_out[0].probs.top1)
                                pred_cls = cls_out[0].names[top1_idx]
                                pred_conf = float(cls_out[0].probs.top1conf.cpu().numpy())
                                
                                tracks_emergency_preds[tid].append((frame_idx, pred_cls, pred_conf, bh, box.tolist()))
        
        cap.release()
        
        # Analyze trajectories to determine movement vectors and directionality
        positive_dy = 0
        negative_dy = 0
        positive_dx = 0
        negative_dx = 0
        vector_displacements = []
        
        for tid, pts in tracks_trajectories.items():
            if len(pts) >= 3:
                start_p = pts[0]
                end_p = pts[-1]
                dx = end_p[1] - start_p[1]
                dy = end_p[2] - start_p[2]
                dist = np.hypot(dx, dy)
                if dist >= 15:  # meaningful motion
                    vector_displacements.append((dx, dy, dist))
                    if dy > 5:
                        positive_dy += 1
                    elif dy < -5:
                        negative_dy += 1
                    if dx > 5:
                        positive_dx += 1
                    elif dx < -5:
                        negative_dx += 1
        
        # Directionality assessment
        total_moving = len(vector_displacements)
        is_bidirectional = False
        dom_movement = "Unknown"
        
        if total_moving > 0:
            # Check vertical dominance vs horizontal dominance
            avg_dx = np.mean([v[0] for v in vector_displacements])
            avg_dy = np.mean([v[1] for v in vector_displacements])
            
            # Check bidirectional ratio
            y_ratio = min(positive_dy, negative_dy) / (max(positive_dy, negative_dy) + 1e-5)
            x_ratio = min(positive_dx, negative_dx) / (max(positive_dx, negative_dx) + 1e-5)
            
            if (positive_dy >= 3 and negative_dy >= 3 and y_ratio > 0.2) or (positive_dx >= 3 and negative_dx >= 3 and x_ratio > 0.2):
                is_bidirectional = True
                dom_movement = f"Bidirectional flow (Down: {positive_dy}, Up: {negative_dy}, Right: {positive_dx}, Left: {negative_dx})"
            else:
                is_bidirectional = False
                if abs(avg_dy) > abs(avg_dx):
                    dom_movement = f"Unidirectional {'Downward (Southbound)' if avg_dy > 0 else 'Upward (Northbound)'} (dx={avg_dx:.1f}, dy={avg_dy:.1f})"
                else:
                    dom_movement = f"Unidirectional {'Rightward (Eastbound)' if avg_dx > 0 else 'Leftward (Westbound)'} (dx={avg_dx:.1f}, dy={avg_dy:.1f})"
        
        # Analyze Emergency detections
        emergency_observed = False
        emergency_details = []
        emergency_movement = "NONE"
        
        for tid, preds in tracks_emergency_preds.items():
            # Check predictions with emergency classes and conf >= 0.60
            em_preds = [p for p in preds if p[1] in ["ambulance", "fire_brigade", "police"] and p[2] >= 0.60]
            if len(em_preds) >= 2: # detected in at least 2 sampled frames (>= 20 frame persistence)
                emergency_observed = True
                classes_count = defaultdict(int)
                confs = []
                for p in em_preds:
                    classes_count[p[1]] += 1
                    confs.append(p[2])
                top_em_cls = max(classes_count, key=classes_count.get)
                max_c = max(confs)
                avg_c = np.mean(confs)
                
                # Check movement direction for this track
                pts = tracks_trajectories.get(tid, [])
                em_dir = "UNKNOWN"
                if len(pts) >= 2:
                    dx = pts[-1][1] - pts[0][1]
                    dy = pts[-1][2] - pts[0][2]
                    # We will assess incoming/outgoing relative to camera approach
                    em_dir = f"dx={dx:.1f}, dy={dy:.1f}"
                
                emergency_details.append({
                    "track_id": tid,
                    "type": top_em_cls,
                    "max_conf": round(max_c, 3),
                    "avg_conf": round(avg_c, 3),
                    "frames_count": len(em_preds),
                    "direction_vec": em_dir
                })
        
        # Correlate emergency movement for summary
        if emergency_observed:
            # Let's inspect the specific tracks to categorize incoming/outgoing
            emergency_movement = "INCOMING" if any("dy" in str(d["direction_vec"]) for d in emergency_details) else "OUTGOING"
        
        rep = {
            "approach": approach_name,
            "filename": video_path.name,
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "frame_count": total_frames,
            "duration": f"{duration_sec:.2f}s",
            "inspected_frames": inspected_frames_count,
            "vehicles_detected": total_detections_count,
            "tracks": len(tracks_trajectories),
            "traffic_directionality": "BIDIRECTIONAL" if is_bidirectional else "UNIDIRECTIONAL",
            "dominant_movement": dom_movement,
            "emergency_vehicle_observed": "YES" if emergency_observed else "NO",
            "emergency_details": emergency_details,
            "emergency_movement": emergency_movement if emergency_observed else "N/A"
        }
        reports[approach_name] = rep
        
        # Immediate stdout output as requested
        print("\n" + "-" * 40)
        print(f"{approach_name}:")
        print(f"- frames inspected: {rep['inspected_frames']}")
        print(f"- vehicles detected: {rep['vehicles_detected']}")
        print(f"- tracks: {rep['tracks']}")
        print(f"- traffic: {rep['traffic_directionality']}")
        print(f"- dominant movement: {rep['dominant_movement']}")
        print(f"- emergency vehicle observed: {rep['emergency_vehicle_observed']}")
        print(f"- emergency movement: {rep['emergency_movement']}")
        if rep["emergency_details"]:
            print(f"- emergency details: {rep['emergency_details']}")
        print("-" * 40)

    # Save to report markdown file
    report_file = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "v2" / "reports" / "four_camera_inspection.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Four-Camera Single-Junction Video Stream Inspection Report\n\n")
        f.write(f"**Generated:** Automated Bounded Inspection (Sample Step = 10 Frames)\n\n")
        f.write("## Per-Approach Stream Diagnostics\n\n")
        f.write("| Approach | Video File | Resolution | FPS | Frame Count | Duration | Inspected Frames | Detections | Tracks | Directionality | Emergency Visible | Emergency Movement |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for app, r in reports.items():
            f.write(f"| **{app}** | `{r['filename']}` | {r['resolution']} | {r['fps']} | {r['frame_count']} | {r['duration']} | {r['inspected_frames']} | {r['vehicles_detected']} | {r['tracks']} | {r['traffic_directionality']} | {r['emergency_vehicle_observed']} | {r['emergency_movement']} |\n")
        
        f.write("\n\n## Detailed Observations\n\n")
        for app, r in reports.items():
            f.write(f"### {app} Approach (`{r['filename']}`)\n")
            f.write(f"- **Resolution & FPS:** {r['resolution']} @ {r['fps']} FPS ({r['frame_count']} frames, {r['duration']})\n")
            f.write(f"- **Traffic Characteristics:** {r['traffic_directionality']} - {r['dominant_movement']}\n")
            f.write(f"- **Total Unique Tracks:** {r['tracks']} tracks across {r['inspected_frames']} inspected frames\n")
            f.write(f"- **Emergency Vehicle Observed:** {r['emergency_vehicle_observed']}\n")
            if r['emergency_details']:
                f.write(f"- **Emergency Details:**\n")
                for em in r['emergency_details']:
                    f.write(f"  - Track {em['track_id']}: Type `{em['type']}`, Max Conf `{em['max_conf']}`, Avg Conf `{em['avg_conf']}`, Vector: `{em['direction_vec']}`\n")
            f.write("\n")
            
    print(f"\n[SUCCESS] Inspection report written to: {report_file}")

if __name__ == "__main__":
    inspect_four_cameras()
