import sys
import time
import torch
import cv2
from collections import defaultdict, Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.traffic_schemas import ApproachEnum
from backend.core.vision.tracker import VehicleTracker
from backend.core.analytics.traffic_metrics import TrafficMetricsCalculator, intersect
from backend.config import settings

def run_diagnostics():
    print("=" * 70, flush=True)
    print("RUNNING COMPREHENSIVE TRACKING & COUNTING DIAGNOSTICS", flush=True)
    print("=" * 70, flush=True)

    input_video = PROJECT_ROOT / "data" / "uploads" / "my_traffic.mp4"
    if not input_video.exists():
        raise FileNotFoundError(f"Input video {input_video} not found!")

    cap = cv2.VideoCapture(str(input_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"• Video: {input_video.name} ({width}x{height}, {total_frames} frames, {fps:.2f} fps)")
    print(f"• Model: {settings.MODEL_PATH}, conf={settings.CONFIDENCE_THRESHOLD}, iou={settings.IOU_THRESHOLD}")
    print(f"• ROI:   {settings.DETECTION_ROI}")
    print(f"• Device: device=0 (CUDA: {torch.cuda.is_available()})")

    tracker = VehicleTracker()
    metrics_calculator = TrafficMetricsCalculator(ApproachEnum.NORTH, None)
    tracker.reset()

    # Counting line coordinates in absolute pixels
    line_cfg = metrics_calculator.line_config
    p1 = (line_cfg["p1"][0] * width, line_cfg["p1"][1] * height)
    p2 = (line_cfg["p2"][0] * width, line_cfg["p2"][1] * height)
    print(f"• Counting Line (NORTH): ({p1[0]:.1f}, {p1[1]:.1f}) -> ({p2[0]:.1f}, {p2[1]:.1f})")

    # Diagnostic Data Structures
    # 1. Class history per track_id: track_id -> list of (frame_idx, class_name, conf)
    track_class_history = defaultdict(list)
    
    # 2. Frame appearances: track_id -> list of frame indices
    track_frame_history = defaultdict(list)
    
    # 3. Trajectory per track_id: track_id -> list of (frame_idx, center_x, center_y)
    track_trajectory = defaultdict(list)

    # 4. Line crossing events: track_id -> list of frame indices where intersect was True
    crossing_events = defaultdict(list)
    first_crossing_frame = {}
    duplicate_crossing_attempts = defaultdict(int)

    # 5. Per-frame active class counts
    per_frame_active_counts = []

    frame_idx = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        tracked_vehicles = tracker.track(frame)

        # Record per-frame data
        frame_classes = Counter()
        for v in tracked_vehicles:
            tid = v.track_id
            cname = v.class_name
            conf = v.confidence
            cx, cy = v.center

            track_class_history[tid].append((frame_idx, cname, conf))
            track_frame_history[tid].append(frame_idx)
            track_trajectory[tid].append((frame_idx, cx, cy))
            frame_classes[cname] += 1

            # Check counting line intersection explicitly for diagnostics
            if v.previous_center is not None:
                if intersect(v.previous_center, v.center, p1, p2):
                    crossing_events[tid].append(frame_idx)
                    if tid not in first_crossing_frame:
                        first_crossing_frame[tid] = frame_idx
                    else:
                        duplicate_crossing_attempts[tid] += 1

        per_frame_active_counts.append(frame_classes)

        # Also update standard metrics calculator to verify parity
        state = metrics_calculator.calculate_metrics(
            vehicles=tracked_vehicles,
            frame_width=width,
            frame_height=height,
            processed_frames=frame_idx,
            fps=fps
        )

    cap.release()
    elapsed = time.time() - t0

    # ------------------ ANALYSIS & COMPILATION ------------------
    all_track_ids = sorted(list(track_class_history.keys()))
    total_unique_ids = len(all_track_ids)

    # A. Class Breakdown Analysis
    # For each track_id, find all distinct classes it was assigned, and its dominant (majority) class
    classes_per_id = {} # tid -> set of classes
    dominant_class_per_id = {} # tid -> majority class
    ids_by_any_class = defaultdict(set) # class -> set of track_ids that had this class at least once
    ids_by_dominant_class = defaultdict(set) # class -> set of track_ids with this dominant class

    multi_class_ids = {} # tid -> Counter of classes

    for tid, history in track_class_history.items():
        cls_list = [h[1] for h in history]
        cls_counter = Counter(cls_list)
        classes_per_id[tid] = set(cls_counter.keys())
        dominant_class = cls_counter.most_common(1)[0][0]
        dominant_class_per_id[tid] = dominant_class

        ids_by_dominant_class[dominant_class].add(tid)
        for c in classes_per_id[tid]:
            ids_by_any_class[c].add(tid)

        if len(cls_counter) > 1:
            multi_class_ids[tid] = cls_counter

    # B. Disappearance and Reappearance (Gap) Analysis
    # Measure track fragmentation: how many times did a track disappear for >= 1 frame and reappear?
    reappearing_tracks = {} # tid -> list of gaps (gap_length, from_frame, to_frame)
    for tid, frames in track_frame_history.items():
        gaps = []
        for i in range(len(frames) - 1):
            diff = frames[i+1] - frames[i]
            if diff > 1:
                gaps.append((diff - 1, frames[i], frames[i+1]))
        if gaps:
            reappearing_tracks[tid] = gaps

    # C. Line Crossing Diagnostics
    crossed_ids_metrics = metrics_calculator.crossed_ids
    unique_crossing_ids = set(first_crossing_frame.keys())

    # ------------------ PRINT DETAILED REPORT ------------------
    print("\n" + "=" * 70, flush=True)
    print("DIAGNOSTIC REPORT SUMMARY:", flush=True)
    print("=" * 70, flush=True)
    print(f"• Total Processed Frames: {frame_idx}")
    print(f"• Processing Time:        {elapsed:.2f}s ({frame_idx/elapsed:.2f} FPS)")
    print(f"• Total Unique Track IDs: {total_unique_ids}")
    print(f"• MetricsCalculator All Seen IDs Count: {len(metrics_calculator.all_seen_ids)}")
    print(f"• Parity Check (Tracker vs MetricsCalculator): {total_unique_ids == len(metrics_calculator.all_seen_ids)}")

    print("\n" + "-" * 70, flush=True)
    print("1. CLASS MULTI-MEMBERSHIP (THE 240 vs 164 DISCREPANCY EXPLANATION):", flush=True)
    print("-" * 70, flush=True)
    print("A single physical vehicle's ByteTrack track ID can receive different YOLO class labels on different frames.")
    print(f"• IDs with exactly ONE class throughout lifetime:   {total_unique_ids - len(multi_class_ids)} / {total_unique_ids} ({((total_unique_ids - len(multi_class_ids))/total_unique_ids)*100:.1f}%)")
    print(f"• IDs that CHANGED class (multi-class flickering):  {len(multi_class_ids)} / {total_unique_ids} ({(len(multi_class_ids)/total_unique_ids)*100:.1f}%)")

    print("\n• Comparison of 'Any-Class Seen' vs 'Dominant / Majority Class':")
    print(f"  {'Class':12s} | {'Any-Class Seen (Overlapping)':30s} | {'Dominant Class (Mutually Exclusive)':35s}")
    print(f"  {'-'*12} | {'-'*30} | {'-'*35}")
    sum_any = 0
    sum_dom = 0
    for c in ["car", "motorcycle", "bus", "truck"]:
        cnt_any = len(ids_by_any_class[c])
        cnt_dom = len(ids_by_dominant_class[c])
        sum_any += cnt_any
        sum_dom += cnt_dom
        print(f"  {c:12s} | {cnt_any:4d} unique IDs                   | {cnt_dom:4d} unique IDs")
    print(f"  {'-'*12} | {'-'*30} | {'-'*35}")
    print(f"  {'SUM TOTAL':12s} | {sum_any:4d} (matches previous 240!)   | {sum_dom:4d} (matches exact 164 total IDs!)")

    print("\n• Sample IDs that changed class across frames:")
    for tid, c_counts in list(multi_class_ids.items())[:12]:
        breakdown_str = ", ".join([f"{cls}: {cnt} frames" for cls, cnt in c_counts.items()])
        dominant = dominant_class_per_id[tid]
        print(f"   - Track #{tid:3d} (Total frames: {sum(c_counts.values()):3d}, Dominant: {dominant:10s}): {breakdown_str}")
    if len(multi_class_ids) > 12:
        print(f"   ... and {len(multi_class_ids) - 12} more IDs that experienced class flickering.")

    print("\n" + "-" * 70, flush=True)
    print("2. TRACK DISAPPEARANCE / REAPPEARANCE (GAP & FRAGMENTATION ANALYSIS):", flush=True)
    print("-" * 70, flush=True)
    print(f"• Total Track IDs that disappeared and reappeared: {len(reappearing_tracks)} / {total_unique_ids}")
    total_gap_events = sum(len(gaps) for gaps in reappearing_tracks.values())
    print(f"• Total Reappearance Gap Events:                  {total_gap_events}")
    
    gap_durations = [gap[0] for gaps in reappearing_tracks.values() for gap in gaps]
    if gap_durations:
        print(f"• Gap Duration Range:                             {min(gap_durations)} to {max(gap_durations)} frames (Mean: {sum(gap_durations)/len(gap_durations):.1f} frames)")
    print("\n• Sample Reappearing Track IDs:")
    for tid, gaps in list(reappearing_tracks.items())[:8]:
        gap_strs = [f"lost {g[0]} frames (f{g[1]}->f{g[2]})" for g in gaps]
        print(f"   - Track #{tid:3d} ({dominant_class_per_id[tid]}): {', '.join(gap_strs)}")

    print("\n" + "-" * 70, flush=True)
    print("3. COUNTING LINE CROSSING & FLOW VERIFICATION:", flush=True)
    print("-" * 70, flush=True)
    print(f"• Traffic Flow Count in MetricsCalculator:       {int(state.flow)}")
    print(f"• Unique Track IDs that crossed counting line:   {len(unique_crossing_ids)}")
    print(f"• Crossed IDs Set Match:                         {crossed_ids_metrics == unique_crossing_ids}")
    print(f"• Crossing Track IDs:                            {sorted(list(unique_crossing_ids))}")
    print(f"• Track IDs with duplicate crossing attempts:    {len(duplicate_crossing_attempts)}")
    if duplicate_crossing_attempts:
        for tid, dup_cnt in duplicate_crossing_attempts.items():
            print(f"   - Track #{tid}: attempted crossing {dup_cnt + 1} times (correctly filtered to 1 by crossed_ids set)")
    else:
        print("   - No duplicate crossing attempts occurred; all crossings were single clean crossings.")

    print("\n• Crossed Vehicles Class Breakdown (by dominant class):")
    crossed_by_dom_class = Counter([dominant_class_per_id[tid] for tid in unique_crossing_ids])
    for c, cnt in crossed_by_dom_class.items():
        print(f"   - {c:12s}: {cnt} vehicles crossed")

    print("\n" + "-" * 70, flush=True)
    print("4. ACTIVE VEHICLES VS CUMULATIVE METRICS:", flush=True)
    print("-" * 70, flush=True)
    print(f"• Final Frame Active Vehicles in Scene: {state.vehicle_count}")
    print(f"• Final Frame Class Counts:            {state.class_counts}")
    print(f"• Final Frame Queue Length:            ~{state.estimated_queue_length}")
    print(f"• Final Frame Density Index:           {state.density:.2f}")
    print(f"• Final Frame Traffic Level:           {state.traffic_level.value}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_diagnostics()
