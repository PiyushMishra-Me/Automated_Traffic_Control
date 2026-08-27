import sys
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.vision.tracker import VehicleTracker
from backend.models.traffic_schemas import ApproachEnum, CameraConfig

def compute_detailed_cctv_stats(video_path: Path, video_name: str, is_bidirectional: bool = False):
    print("=" * 75)
    print(f"CALCULATING EXACT BOUNDING BOX / CROP STATISTICS FOR: {video_name}")
    print(f"File Path: {video_path}")
    print("=" * 75)

    if not video_path.exists():
        print(f"Error: {video_path} does not exist!")
        return

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height} | FPS: {fps:.1f} | Total Frames: {total_frames}")

    # Use tracker with full frame or north ROI as per production
    if is_bidirectional:
        # Full frame evaluation for bidirectional camera
        camera_config = CameraConfig(
            camera_id="CAM-TEST-BI",
            junction_id="J-TEST",
            approach=ApproachEnum.NORTH,
            roi=None,
            junction_vector=[0.0, 1.0],
            is_bidirectional=True
        )
        tracker = VehicleTracker(roi=None, camera_config=camera_config)
    else:
        tracker = VehicleTracker(approach=ApproachEnum.NORTH)

    # Monkey patch model track to use device='cpu' if torch cuda is unavailable
    orig_track = tracker.model.track
    def cpu_track(*args, **kwargs):
        kwargs['device'] = 'cpu'
        return orig_track(*args, **kwargs)
    tracker.model.track = cpu_track

    tracker.reset()

    # Track metrics
    # Per-class bounding box heights across all frame observations
    class_all_heights = {"car": [], "motorcycle": [], "bus": [], "truck": []}
    class_all_widths = {"car": [], "motorcycle": [], "bus": [], "truck": []}
    
    # Per-track first observed height and max observed height
    track_first_heights = {} # track_id -> (class_name, first_h, first_w)
    track_max_heights = {}   # track_id -> (class_name, max_h, max_w)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        tracked_vehicles = tracker.track(frame, fps=fps)
        for v in tracked_vehicles:
            w_box = float(v.xyxy[2] - v.xyxy[0])
            h_box = float(v.xyxy[3] - v.xyxy[1])
            c_name = v.class_name

            if c_name in class_all_heights:
                class_all_heights[c_name].append(h_box)
                class_all_widths[c_name].append(w_box)

            tid = v.track_id
            if tid not in track_first_heights:
                track_first_heights[tid] = (c_name, h_box, w_box)
                track_max_heights[tid] = (c_name, h_box, w_box)
            else:
                curr_max_h = track_max_heights[tid][1]
                curr_max_w = track_max_heights[tid][2]
                if h_box > curr_max_h:
                    track_max_heights[tid] = (c_name, h_box, max(w_box, curr_max_w))

    cap.release()

    print(f"\nProcessed {frame_idx} frames. Total unique tracked vehicles: {len(track_first_heights)}")

    print("\n" + "-" * 75)
    print("1. OVERALL BOUNDING BOX OBSERVATION STATISTICS (ALL DETECTIONS ACROSS ALL FRAMES)")
    print("-" * 75)
    print(f"{'Class':<12} | {'Count':<8} | {'Min (px)':<9} | {'Median (px)':<12} | {'Max (px)':<9} | {'Mean (px)':<9}")
    print("-" * 75)
    for c_name in ["car", "motorcycle", "bus", "truck"]:
        hs = class_all_heights[c_name]
        if hs:
            print(f"{c_name.capitalize():<12} | {len(hs):<8} | {min(hs):<9.1f} | {np.median(hs):<12.1f} | {max(hs):<9.1f} | {np.mean(hs):<9.1f}")
        else:
            print(f"{c_name.capitalize():<12} | 0        | N/A       | N/A          | N/A       | N/A")

    print("\n" + "-" * 75)
    print("2. TRACK INITIAL ENTRY HEIGHT PERCENTAGES (WHEN VEHICLE FIRST APPEARS)")
    print("-" * 75)
    print(f"{'Class':<12} | {'Total Tracks':<13} | {'< 32px':<10} | {'< 36px':<10} | {'< 40px':<10} | {'> 64px':<10} | {'> 100px':<10}")
    print("-" * 75)

    all_initial_hs = []
    for c_name in ["car", "motorcycle", "bus", "truck"]:
        c_tracks = [h for tid, (cls, h, w) in track_first_heights.items() if cls == c_name]
        all_initial_hs.extend(c_tracks)
        n = len(c_tracks)
        if n > 0:
            p_lt_32 = sum(1 for h in c_tracks if h < 32.0) / n * 100.0
            p_lt_36 = sum(1 for h in c_tracks if h < 36.0) / n * 100.0
            p_lt_40 = sum(1 for h in c_tracks if h < 40.0) / n * 100.0
            p_gt_64 = sum(1 for h in c_tracks if h > 64.0) / n * 100.0
            p_gt_100 = sum(1 for h in c_tracks if h > 100.0) / n * 100.0
            print(f"{c_name.capitalize():<12} | {n:<13} | {p_lt_32:<9.1f}% | {p_lt_36:<9.1f}% | {p_lt_40:<9.1f}% | {p_gt_64:<9.1f}% | {p_gt_100:<9.1f}%")
        else:
            print(f"{c_name.capitalize():<12} | 0             | 0.0%      | 0.0%      | 0.0%      | 0.0%      | 0.0%")

    total_all_tracks = len(track_first_heights)
    if total_all_tracks > 0:
        tot_lt_32 = sum(1 for h in all_initial_hs if h < 32.0) / total_all_tracks * 100.0
        tot_lt_36 = sum(1 for h in all_initial_hs if h < 36.0) / total_all_tracks * 100.0
        tot_lt_40 = sum(1 for h in all_initial_hs if h < 40.0) / total_all_tracks * 100.0
        tot_gt_64 = sum(1 for h in all_initial_hs if h > 64.0) / total_all_tracks * 100.0
        tot_gt_100 = sum(1 for h in all_initial_hs if h > 100.0) / total_all_tracks * 100.0
        print("-" * 75)
        print(f"{'ALL VEHICLES':<12} | {total_all_tracks:<13} | {tot_lt_32:<9.1f}% | {tot_lt_36:<9.1f}% | {tot_lt_40:<9.1f}% | {tot_gt_64:<9.1f}% | {tot_gt_100:<9.1f}%")
    print("-" * 75)

    print("\n" + "-" * 75)
    print("3. TRACK MAXIMUM OBSERVED HEIGHT DISTRIBUTION (WHEN VEHICLE IS CLOSEST/LARGEST)")
    print("-" * 75)
    for c_name in ["car", "motorcycle", "bus", "truck"]:
        c_max_hs = [h for tid, (cls, h, w) in track_max_heights.items() if cls == c_name]
        if c_max_hs:
            print(f"{c_name.capitalize():<12} | Tracks={len(c_max_hs):<3} | Min={min(c_max_hs):<5.1f}px | Median={np.median(c_max_hs):<5.1f}px | Max={max(c_max_hs):<5.1f}px | >64px={sum(1 for h in c_max_hs if h>64)/len(c_max_hs)*100:.1f}% | >100px={sum(1 for h in c_max_hs if h>100)/len(c_max_hs)*100:.1f}%")
    print("-" * 75 + "\n")

if __name__ == "__main__":
    my_traffic = PROJECT_ROOT / "data" / "uploads" / "my_traffic.mp4"
    bidirectional = PROJECT_ROOT / "data" / "uploads" / "bidirectional.mp4"

    compute_detailed_cctv_stats(my_traffic, "my_traffic.mp4 (768x432)", is_bidirectional=False)
    compute_detailed_cctv_stats(bidirectional, "bidirectional.mp4 (768x432)", is_bidirectional=True)
