import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.vision.tracker import VehicleTracker
from backend.models.traffic_schemas import CameraConfig, ApproachEnum

def analyze_video(video_path: Path, title: str, camera_config: CameraConfig = None):
    print("=" * 65)
    print(f"ANALYZING CCTV VIDEO: {title}")
    print(f"File: {video_path}")
    print("=" * 65)
    
    if not video_path.exists():
        print(f"Error: {video_path} not found!")
        return

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Resolution: {width}x{height} | FPS: {fps:.1f} | Total Frames: {total_frames}")

    tracker = VehicleTracker()
    tracker.reset()

    crop_sizes_by_class = {"car": [], "motorcycle": [], "bus": [], "truck": []}
    track_max_crops = {} # track_id -> {'class': str, 'max_h': int, 'max_w': int, 'min_h': int, 'min_w': int, 'frames': int}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        tracked = tracker.track(frame)
        for v in tracked:
            w_box = v.xyxy[2] - v.xyxy[0]
            h_box = v.xyxy[3] - v.xyxy[1]
            c_name = v.class_name
            if c_name in crop_sizes_by_class:
                crop_sizes_by_class[c_name].append((w_box, h_box))

            if v.track_id not in track_max_crops:
                track_max_crops[v.track_id] = {
                    'class': c_name,
                    'max_h': h_box,
                    'max_w': w_box,
                    'min_h': h_box,
                    'min_w': w_box,
                    'frames': 1
                }
            else:
                info = track_max_crops[v.track_id]
                info['max_h'] = max(info['max_h'], h_box)
                info['max_w'] = max(info['max_w'], w_box)
                info['min_h'] = min(info['min_h'], h_box)
                info['min_w'] = min(info['min_w'], w_box)
                info['frames'] += 1

    cap.release()

    print(f"\nProcessed {frame_idx} frames. Total unique tracked vehicles: {len(track_max_crops)}")
    print("\n--- CROP SIZE DISTRIBUTION (Width x Height in Pixels) ---")
    for c_name, sizes in crop_sizes_by_class.items():
        if not sizes:
            continue
        ws = [s[0] for s in sizes]
        hs = [s[1] for s in sizes]
        print(f"[{c_name.upper()}] (n={len(sizes)} bounding box instances):")
        print(f"  Width:  Min={min(ws):.1f}px, Median={np.median(ws):.1f}px, Max={max(ws):.1f}px, Mean={np.mean(ws):.1f}px")
        print(f"  Height: Min={min(hs):.1f}px, Median={np.median(hs):.1f}px, Max={max(hs):.1f}px, Mean={np.mean(hs):.1f}px")

    print("\n--- DISTANT vs CLOSE VEHICLE TRACK RANGES ---")
    distant_tracks = [t for t, d in track_max_crops.items() if d['min_h'] < 40]
    mid_tracks = [t for t, d in track_max_crops.items() if 40 <= d['min_h'] <= 100]
    large_tracks = [t for t, d in track_max_crops.items() if d['min_h'] > 100]
    print(f"Tracks entering as distant (<40px height): {len(distant_tracks)} / {len(track_max_crops)}")
    print(f"Tracks in mid range (40px - 100px height): {len(mid_tracks)} / {len(track_max_crops)}")
    print(f"Tracks only seen up close (>100px height): {len(large_tracks)} / {len(track_max_crops)}")

if __name__ == "__main__":
    my_traffic = PROJECT_ROOT / "data" / "uploads" / "my_traffic.mp4"
    bidirectional = PROJECT_ROOT / "data" / "uploads" / "bidirectional.mp4"

    analyze_video(my_traffic, "my_traffic.mp4 (768x432)")
    
    bi_config = CameraConfig(
        camera_id="CAM-TEST-BI",
        junction_id="J-TEST",
        approach=ApproachEnum.NORTH,
        roi=[0.0, 0.0, 1.0, 1.0],
        junction_vector=[0.0, 1.0],
        is_bidirectional=True
    )
    analyze_video(bidirectional, "bidirectional.mp4 (768x432)", camera_config=bi_config)
