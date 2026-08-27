import os
import sys
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def extract_west_ambulances():
    video_path = PROJECT_ROOT / "data" / "uploads" / "west.mp4"
    det_model_path = PROJECT_ROOT / "yolov8s.pt"
    
    out_dir = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "west_ambulance_validation"
    amb1_dir = out_dir / "ambulance_1"
    amb2_dir = out_dir / "ambulance_2"
    
    out_dir.mkdir(parents=True, exist_ok=True)
    amb1_dir.mkdir(parents=True, exist_ok=True)
    amb2_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    
    det_model = YOLO(str(det_model_path))
    
    # -------------------------------------------------------------
    # Expected spatial trajectories in west.mp4 (moving up-left):
    # Ambulance 1 (Frames 605-685):
    #   F610: x~410..520, y~330..410
    #   F620: x~360..450, y~300..380
    #   F630: x~315..405, y~270..345
    #   F640: x~280..360, y~250..315
    #   F650: x~255..330, y~230..290
    #   F660: x~230..300, y~215..270
    #   F670: x~210..275, y~200..250
    #
    # Ambulance 2 (Frames 280-430):
    #   F290: x~240..340, y~330..430
    #   F300: x~220..325, y~315..425
    #   F310: x~205..300, y~295..395
    #   F320: x~190..280, y~280..370
    #   F330: x~175..260, y~265..350
    #   F340: x~160..245, y~250..330
    #   F350: x~150..230, y~240..315
    #   F360: x~140..215, y~230..295
    #   F370: x~130..200, y~220..280
    # -------------------------------------------------------------
    
    amb1_target_frames = [610, 620, 630, 640, 650, 660, 670]
    amb2_target_frames = [290, 300, 310, 320, 330, 340, 350, 360, 370]
    
    manifest_records = []
    failed_frames = []
    contact_sheet_crops = []
    
    # Process Ambulance 1
    print("Extracting crops for Ambulance 1 (Frames 610-670)...")
    for f_idx in amb1_target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            failed_frames.append({"ambulance_id": "ambulance_1", "frame": f_idx, "reason": "Frame read error"})
            continue
            
        res = det_model.predict(source=frame, conf=0.15, classes=[2, 3, 5, 7], imgsz=640, verbose=False, device="cpu")
        best_box = None
        
        if res and res[0].boxes is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
            # Find the ambulance box based on visual lane trajectory
            for b in boxes:
                bx1, by1, bx2, by2 = b
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                bw, bh = bx2 - bx1, by2 - by1
                
                # Spatial gating for Ambulance 1 along left carriageway
                if 200 <= cx <= 520 and 200 <= cy <= 410 and bh >= 30:
                    # Trajectory consistency check
                    if f_idx == 610 and (380 <= cx <= 490 and 310 <= cy <= 400):
                        best_box = b
                    elif f_idx == 620 and (350 <= cx <= 440 and 290 <= cy <= 375):
                        best_box = b
                    elif f_idx == 630 and (310 <= cx <= 395 and 270 <= cy <= 340):
                        best_box = b
                    elif f_idx == 640 and (275 <= cx <= 350 and 245 <= cy <= 315):
                        best_box = b
                    elif f_idx == 650 and (250 <= cx <= 325 and 225 <= cy <= 290):
                        best_box = b
                    elif f_idx == 660 and (225 <= cx <= 300 and 210 <= cy <= 270):
                        best_box = b
                    elif f_idx == 670 and (200 <= cx <= 275 and 195 <= cy <= 255):
                        best_box = b
                        
        if best_box is not None:
            bx1, by1, bx2, by2 = best_box
            bw, bh = bx2 - bx1, by2 - by1
            pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
            cx1, cy1 = max(0, bx1 - pad_x), max(0, by1 - pad_y)
            cx2, cy2 = min(width, bx2 + pad_x), min(height, by2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            
            crop_name = f"amb1_f{f_idx:04d}_h{bh}px.jpg"
            crop_path = amb1_dir / crop_name
            cv2.imwrite(str(crop_path), crop)
            
            manifest_records.append({
                "source_video": "west.mp4",
                "ambulance_id": "ambulance_1",
                "frame_number": f_idx,
                "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
                "crop_bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
                "bbox_width": int(bw),
                "bbox_height": int(bh),
                "crop_filename": crop_name,
                "relative_path": f"ambulance_1/{crop_name}"
            })
            
            # Label for contact sheet
            sheet_img = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
            cv2.putText(sheet_img, f"Amb 1 F{f_idx} ({bh}px)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
            contact_sheet_crops.append(sheet_img)
            print(f"  [OK] Ambulance 1 Frame {f_idx}: bbox={best_box.tolist()}, h={bh}px")
        else:
            failed_frames.append({"ambulance_id": "ambulance_1", "frame": f_idx, "reason": "Ambulance not detected by YOLO"})
            print(f"  [FAILED] Ambulance 1 Frame {f_idx}: No matching detection")

    # Process Ambulance 2
    print("\nExtracting crops for Ambulance 2 (Frames 290-370)...")
    for f_idx in amb2_target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            failed_frames.append({"ambulance_id": "ambulance_2", "frame": f_idx, "reason": "Frame read error"})
            continue
            
        res = det_model.predict(source=frame, conf=0.15, classes=[2, 3, 5, 7], imgsz=640, verbose=False, device="cpu")
        best_box = None
        
        if res and res[0].boxes is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy().astype(int)
            for b in boxes:
                bx1, by1, bx2, by2 = b
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                bw, bh = bx2 - bx1, by2 - by1
                
                # Spatial gating for Ambulance 2 in left carriageway
                if 120 <= cx <= 350 and 220 <= cy <= 435 and bh >= 35:
                    if f_idx == 290 and (220 <= cx <= 320 and 320 <= cy <= 420):
                        best_box = b
                    elif f_idx == 300 and (210 <= cx <= 300 and 310 <= cy <= 410):
                        best_box = b
                    elif f_idx == 310 and (195 <= cx <= 280 and 290 <= cy <= 390):
                        best_box = b
                    elif f_idx == 320 and (180 <= cx <= 265 and 275 <= cy <= 370):
                        best_box = b
                    elif f_idx == 330 and (165 <= cx <= 250 and 260 <= cy <= 350):
                        best_box = b
                    elif f_idx == 340 and (155 <= cx <= 235 and 245 <= cy <= 335):
                        best_box = b
                    elif f_idx == 350 and (145 <= cx <= 220 and 235 <= cy <= 320):
                        best_box = b
                    elif f_idx == 360 and (135 <= cx <= 210 and 225 <= cy <= 305):
                        best_box = b
                    elif f_idx == 370 and (125 <= cx <= 200 and 215 <= cy <= 290):
                        best_box = b
                        
        if best_box is not None:
            bx1, by1, bx2, by2 = best_box
            bw, bh = bx2 - bx1, by2 - by1
            pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
            cx1, cy1 = max(0, bx1 - pad_x), max(0, by1 - pad_y)
            cx2, cy2 = min(width, bx2 + pad_x), min(height, by2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            
            crop_name = f"amb2_f{f_idx:04d}_h{bh}px.jpg"
            crop_path = amb2_dir / crop_name
            cv2.imwrite(str(crop_path), crop)
            
            manifest_records.append({
                "source_video": "west.mp4",
                "ambulance_id": "ambulance_2",
                "frame_number": f_idx,
                "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
                "crop_bbox": [int(cx1), int(cy1), int(cx2), int(cy2)],
                "bbox_width": int(bw),
                "bbox_height": int(bh),
                "crop_filename": crop_name,
                "relative_path": f"ambulance_2/{crop_name}"
            })
            
            sheet_img = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
            cv2.putText(sheet_img, f"Amb 2 F{f_idx} ({bh}px)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            contact_sheet_crops.append(sheet_img)
            print(f"  [OK] Ambulance 2 Frame {f_idx}: bbox={best_box.tolist()}, h={bh}px")
        else:
            failed_frames.append({"ambulance_id": "ambulance_2", "frame": f_idx, "reason": "Ambulance not detected by YOLO"})
            print(f"  [FAILED] Ambulance 2 Frame {f_idx}: No matching detection")
            
    cap.release()
    
    # Save JSON Manifest
    manifest_json_path = out_dir / "manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "manifest": manifest_records,
            "failed_frames": failed_frames
        }, f, indent=2)
        
    # Save CSV Manifest
    manifest_csv_path = out_dir / "manifest.csv"
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_video", "ambulance_id", "frame_number", "bbox",
            "crop_bbox", "bbox_width", "bbox_height", "crop_filename", "relative_path"
        ])
        writer.writeheader()
        writer.writerows(manifest_records)
        
    # Create unified Contact Sheet
    if contact_sheet_crops:
        cols = 4
        rows = int(np.ceil(len(contact_sheet_crops) / cols))
        cell_w, cell_h = 160, 160
        grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
        
        for idx, c_img in enumerate(contact_sheet_crops):
            r = idx // cols
            c = idx % cols
            grid[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = c_img
            
        contact_sheet_path = out_dir / "contact_sheet.jpg"
        cv2.imwrite(str(contact_sheet_path), grid)
        print(f"\n[OK] Contact sheet saved to: {contact_sheet_path}")
        
    # Summary Metrics
    amb1_records = [m for m in manifest_records if m["ambulance_id"] == "ambulance_1"]
    amb2_records = [m for m in manifest_records if m["ambulance_id"] == "ambulance_2"]
    
    amb1_heights = [m["bbox_height"] for m in amb1_records]
    amb2_heights = [m["bbox_height"] for m in amb2_records]
    
    print("\n" + "=" * 60)
    print("WEST AMBULANCE VALIDATION EXTRACTION SUMMARY:")
    print("=" * 60)
    print(f"Ambulance 1 Crops Extracted: {len(amb1_records)} / {len(amb1_target_frames)}")
    if amb1_heights:
        print(f"Ambulance 1 BBox Height Range: {min(amb1_heights)} px – {max(amb1_heights)} px (mean: {np.mean(amb1_heights):.1f} px)")
    print(f"Ambulance 2 Crops Extracted: {len(amb2_records)} / {len(amb2_target_frames)}")
    if amb2_heights:
        print(f"Ambulance 2 BBox Height Range: {min(amb2_heights)} px – {max(amb2_heights)} px (mean: {np.mean(amb2_heights):.1f} px)")
    print(f"Failed Frames Count: {len(failed_frames)}")
    if failed_frames:
        print(f"Failed Frames: {failed_frames}")
    print(f"Output Directory: {out_dir}")

if __name__ == "__main__":
    extract_west_ambulances()
