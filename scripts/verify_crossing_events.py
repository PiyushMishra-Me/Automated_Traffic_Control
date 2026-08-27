import sys
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.vision.tracker import VehicleTracker
from backend.core.analytics.traffic_metrics import intersect

def analyze_crossings():
    tracker = VehicleTracker()
    tracker.reset()

    video_path = PROJECT_ROOT / "data" / "uploads" / "my_traffic.mp4"
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    p1 = (0.1 * w, 0.65 * h)
    p2 = (0.9 * w, 0.65 * h)
    line_y = 0.65 * h # 280.8 px
    line_x1 = 0.1 * w # 76.8 px
    line_x2 = 0.9 * w # 691.2 px

    print("=" * 80)
    print(f"COUNTING LINE ANALYSIS FOR {video_path.name} ({w}x{h}, {total_frames} frames)")
    print(f"Counting Line: y={line_y:.1f}px, x ∈ [{line_x1:.1f}, {line_x2:.1f}]")
    print("=" * 80)

    crossings = []
    crossed_set = set()
    all_tracks = defaultdict(list)
    all_raw_frames = {}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        all_raw_frames[frame_idx] = frame

        vehicles = tracker.track(frame)
        for v in vehicles:
            all_tracks[v.track_id].append({
                "frame": frame_idx,
                "center": v.center,
                "xyxy": v.xyxy,
                "class": v.class_name,
                "conf": v.confidence
            })
            if v.track_id not in crossed_set and v.previous_center is not None:
                if intersect(v.previous_center, v.center, p1, p2):
                    crossed_set.add(v.track_id)
                    crossings.append({
                        "tid": v.track_id,
                        "class": v.class_name,
                        "frame": frame_idx,
                        "prev_center": v.previous_center,
                        "curr_center": v.center,
                        "xyxy": v.xyxy,
                        "conf": v.confidence
                    })

    cap.release()

    print(f"\n1. DETAILS OF ALL {len(crossings)} COUNTED CROSSING EVENTS:")
    print("-" * 80)
    print(f"{'#':2s} | {'ID':4s} | {'Class':10s} | {'Frame':5s} | {'Lifespan (Frames)':22s} | {'Crossing Transition (y)':24s} | {'x Position':10s}")
    print("-" * 80)
    for i, c in enumerate(crossings, 1):
        tid = c["tid"]
        history = all_tracks[tid]
        f_start = history[0]["frame"]
        f_end = history[-1]["frame"]
        dur = len(history)
        prev_y = c["prev_center"][1]
        curr_y = c["curr_center"][1]
        curr_x = c["curr_center"][0]
        print(f"{i:2d} | #{tid:3d} | {c['class']:10s} | f{c['frame']:03d} | f{f_start:03d} -> f{f_end:03d} ({dur:3d}f) | y={prev_y:5.1f} -> y={curr_y:5.1f} | x={curr_x:5.1f}px")

    # 2. Check for vehicles that spanned across line_y but were NOT registered as crossing
    print("\n" + "=" * 80)
    print("2. VERIFICATION: DID ANY TRACK CROSS y=280.8 WITHOUT TRIGGERING INTERSECT?")
    print("=" * 80)
    missed_crossings = []
    for tid, history in all_tracks.items():
        if tid in crossed_set:
            continue
        # Check if min_y < line_y and max_y > line_y
        ys = [h["center"][1] for h in history]
        min_y, max_y = min(ys), max(ys)
        if min_y < line_y and max_y > line_y:
            missed_crossings.append((tid, history[0]["class"], min_y, max_y, len(history)))

    if missed_crossings:
        print(f"Found {len(missed_crossings)} tracks that traversed across y={line_y:.1f} without intersecting:")
        for m in missed_crossings:
            print(f"   - Track #{m[0]:3d} ({m[1]}): y spans [{m[2]:.1f}, {m[3]:.1f}] across {m[4]} frames (check if outside x-bounds [{line_x1:.1f}, {line_x2:.1f}])")
    else:
        print("None! All tracks that traversed across y=280.8 within the roadway bounds were registered.")

    # 3. Check for vehicles near the bottom / exiting (y > 280.8) vs vehicles queued above line (y < 280.8)
    print("\n" + "=" * 80)
    print("3. ROADWAY TRAFFIC REGIONS BREAKDOWN:")
    print("=" * 80)
    above_line_only = []
    below_line_only = []
    cross_line = []
    for tid, history in all_tracks.items():
        ys = [h["center"][1] for h in history]
        min_y, max_y = min(ys), max(ys)
        if tid in crossed_set:
            cross_line.append(tid)
        elif max_y <= line_y:
            above_line_only.append(tid)
        elif min_y >= line_y:
            below_line_only.append(tid)

    print(f"• Total Unique Track IDs:                  {len(all_tracks)}")
    print(f"• Crossed the Counting Line (Flow):        {len(cross_line)} tracks")
    print(f"• Stayed exclusively ABOVE line (Queued):  {len(above_line_only)} tracks (Distant & mid-road traffic/queue)")
    print(f"• Started exclusively BELOW line:          {len(below_line_only)} tracks (Initial foreground vehicles already past line at f1)")

    print("\n• Initial foreground tracks starting below line at frame 1:")
    for tid in below_line_only:
        h = all_tracks[tid]
        if h[0]["frame"] <= 5:
            print(f"   - Track #{tid:3d} ({h[0]['class']}): started at f{h[0]['frame']} at y={h[0]['center'][1]:.1f}px (already past count line at start of video)")

    # 4. Check for motorcycle track fragmentation / behavior
    print("\n" + "=" * 80)
    print("4. MOTORCYCLE TRACKING & CROSSING AUDIT:")
    print("=" * 80)
    motorcycle_tracks = {tid: h for tid, h in all_tracks.items() if h[-1]["class"] == "motorcycle"}
    print(f"• Total Motorcycle Track IDs: {len(motorcycle_tracks)}")
    print(f"• Motorcycles that crossed line: {sum(1 for tid in motorcycle_tracks if tid in crossed_set)}")
    for tid, h in motorcycle_tracks.items():
        crossed_str = f"CROSSED at f{first_crossing_frame(tid, crossings)}" if tid in crossed_set else "Queued/Moving in upper sector"
        print(f"   - Moto Track #{tid:3d} | f{h[0]['frame']:03d}->f{h[-1]['frame']:03d} ({len(h):3d}f) | y: [{min(x['center'][1] for x in h):5.1f}, {max(x['center'][1] for x in h):5.1f}]px | {crossed_str}")

def first_crossing_frame(tid, crossings):
    for c in crossings:
        if c["tid"] == tid:
            return c["frame"]
    return -1

if __name__ == "__main__":
    analyze_crossings()
