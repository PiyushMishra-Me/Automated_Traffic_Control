import os
import sys
import csv
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def extract_east_ambulance_dataset():
    video_path = PROJECT_ROOT / "data" / "uploads" / "east.mp4"
    det_model_path = PROJECT_ROOT / "yolov8s.pt"
    
    out_dir = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "demo_ambulance" / "east_track_373"
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    
    det_model = YOLO(str(det_model_path))
    
    # 1. Run tracking across frames 650 to 760 to isolate Track #373
    all_track_detections = []
    frame_idx = 0
    
    print("Tracking vehicles in east.mp4 to isolate Track #373...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        if frame_idx > 760:
            break
            
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
            
            for box, tid, cid in zip(boxes, ids, clss):
                if tid == 373:
                    x1, y1, x2, y2 = box
                    bw, bh = x2 - x1, y2 - y1
                    
                    # 8% crop padding
                    pad_x, pad_y = int(bw * 0.08), int(bh * 0.08)
                    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    cx2, cy2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
                    crop = frame[cy1:cy2, cx1:cx2]
                    
                    all_track_detections.append({
                        "frame": frame_idx,
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "crop_box": [int(cx1), int(cy1), int(cx2), int(cy2)],
                        "bw": int(bw),
                        "bh": int(bh),
                        "crop": crop.copy()
                    })
                    
    cap.release()
    print(f"Captured {len(all_track_detections)} frames for Track #373.")
    
    if len(all_track_detections) == 0:
        raise RuntimeError("Track #373 not captured in frame window.")
        
    # 2. Select approximately 30 evenly-spaced representative frames
    target_count = min(30, len(all_track_detections))
    selected_indices = np.linspace(0, len(all_track_detections) - 1, target_count, dtype=int)
    # Deduplicate indices if any
    selected_indices = sorted(list(dict.fromkeys(selected_indices)))
    
    selected_samples = [all_track_detections[i] for i in selected_indices]
    
    manifest_records = []
    crop_images_for_contact_sheet = []
    
    for i, sample in enumerate(selected_samples):
        f_num = sample["frame"]
        crop = sample["crop"]
        bh = sample["bh"]
        bw = sample["bw"]
        box = sample["box"]
        
        crop_filename = f"east_ambulance_f{f_num:04d}_h{bh}px.jpg"
        crop_path = crops_dir / crop_filename
        cv2.imwrite(str(crop_path), crop)
        
        # Prepare resized version for contact sheet
        resized_for_sheet = cv2.resize(crop, (150, 150), interpolation=cv2.INTER_AREA)
        # Put small label on top
        cv2.putText(resized_for_sheet, f"F{f_num} ({bh}px)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        crop_images_for_contact_sheet.append(resized_for_sheet)
        
        manifest_records.append({
            "source_video": "east.mp4",
            "frame_number": f_num,
            "track_id": 373,
            "bbox_coordinates": box,
            "crop_width_px": bw,
            "crop_height_px": bh,
            "crop_filename": crop_filename,
            "relative_path": f"crops/{crop_filename}"
        })
        
    # 3. Save JSON Manifest
    manifest_json_path = out_dir / "manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
        
    # 4. Save CSV Manifest
    manifest_csv_path = out_dir / "manifest.csv"
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_video", "frame_number", "track_id", "bbox_coordinates",
            "crop_width_px", "crop_height_px", "crop_filename", "relative_path"
        ])
        writer.writeheader()
        writer.writerows(manifest_records)
        
    # 5. Generate Contact Sheet
    # Arrange in a 5 x 6 grid (or appropriate dimension)
    cols = 6
    rows = int(np.ceil(len(crop_images_for_contact_sheet) / cols))
    cell_w, cell_h = 150, 150
    contact_sheet = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    
    for idx, crop_img in enumerate(crop_images_for_contact_sheet):
        r = idx // cols
        c = idx % cols
        contact_sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = crop_img
        
    contact_sheet_path = out_dir / "contact_sheet.jpg"
    cv2.imwrite(str(contact_sheet_path), contact_sheet)
    print(f"Contact sheet saved to: {contact_sheet_path}")
    
    # 6. Compute statistics
    heights = [m["crop_height_px"] for m in manifest_records]
    frames = [m["frame_number"] for m in manifest_records]
    
    summary = {
        "num_crops": len(manifest_records),
        "frame_range": f"Frame {min(frames)} – Frame {max(frames)}",
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_height": float(np.mean(heights)),
        "output_dir": str(out_dir),
        "contact_sheet": str(contact_sheet_path)
    }
    
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY:")
    print("=" * 60)
    print(f"Number of Crops Extracted: {summary['num_crops']}")
    print(f"Frame Range: {summary['frame_range']}")
    print(f"Minimum Crop Height: {summary['min_height']} px")
    print(f"Maximum Crop Height: {summary['max_height']} px")
    print(f"Average Crop Height: {summary['avg_height']:.1f} px")
    print(f"Output Directory: {summary['output_dir']}")
    print(f"Contact Sheet: {summary['contact_sheet']}")

if __name__ == "__main__":
    extract_east_ambulance_dataset()
