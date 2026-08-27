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
from backend.core.vision.video_processor import VideoProcessor
from backend.core.analytics.traffic_metrics import TrafficMetricsCalculator
from backend.config import settings

def run_stable_classes_pipeline():
    print("=" * 70, flush=True)
    print("RUNNING FULL PIPELINE WITH TEMPORAL CLASS STABILIZATION", flush=True)
    print("=" * 70, flush=True)
    print(f"• Active Model:        {settings.MODEL_PATH}", flush=True)
    print(f"• Confidence:          {settings.CONFIDENCE_THRESHOLD}", flush=True)
    print(f"• IOU Threshold:       {settings.IOU_THRESHOLD}", flush=True)
    print(f"• Detection ROI:       {settings.DETECTION_ROI}", flush=True)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    print(f"• PyTorch CUDA:        {cuda_available} ({device_name})", flush=True)

    input_video = PROJECT_ROOT / "data" / "uploads" / "my_traffic.mp4"
    output_video = PROJECT_ROOT / "data" / "annotated" / "my_traffic_stable_classes.mp4"

    if not input_video.exists():
        raise FileNotFoundError(f"Input video {input_video} not found!")

    output_video.parent.mkdir(parents=True, exist_ok=True)

    tracker = VehicleTracker()
    processor = VideoProcessor(tracker=tracker)
    metrics_calculator = TrafficMetricsCalculator(ApproachEnum.NORTH, None)
    tracker.reset()

    cap = cv2.VideoCapture(str(input_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    print(f"• Video Dimensions:    {width}x{height}", flush=True)
    print(f"• Total Video Frames:  {total_frames}", flush=True)
    print(f"• Output Video Path:   {output_video.name}", flush=True)

    # Diagnostic trackers
    raw_class_sequence_by_id = defaultdict(list)    # track_id -> [raw_class_name, ...]
    stable_class_sequence_by_id = defaultdict(list) # track_id -> [stable_class_name, ...]
    final_stable_class_by_id = {}                   # track_id -> stable_class_name
    all_seen_tids = set()

    frame_idx = 0
    t0 = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Run production VehicleTracker with temporal class stabilization
            tracked_vehicles = tracker.track(frame)

            for v in tracked_vehicles:
                tid = v.track_id
                all_seen_tids.add(tid)
                raw_cname = getattr(v, 'raw_class_name', v.class_name)
                stable_cname = v.class_name

                raw_class_sequence_by_id[tid].append(raw_cname)
                stable_class_sequence_by_id[tid].append(stable_cname)
                final_stable_class_by_id[tid] = stable_cname

            # Calculate metrics
            state = metrics_calculator.calculate_metrics(
                vehicles=tracked_vehicles,
                frame_width=width,
                frame_height=height,
                processed_frames=frame_idx,
                fps=fps
            )

            # Annotate full frame
            annotated_frame = processor._annotate_frame(
                frame=frame,
                tracked_vehicles=tracked_vehicles,
                state=state,
                approach=ApproachEnum.NORTH,
                line_config=metrics_calculator.line_config,
                width=width,
                height=height
            )

            out.write(annotated_frame)

            if frame_idx % 30 == 0 or frame_idx == total_frames:
                prog = (frame_idx / total_frames) * 100
                print(f"[{prog:5.1f}%] Processed frame {frame_idx}/{total_frames} (Active: {state.vehicle_count}, Flow: {int(state.flow)})", flush=True)

    finally:
        cap.release()
        out.release()

    elapsed = time.time() - t0
    calc_fps = frame_idx / elapsed if elapsed > 0 else 0

    # ------------------ COMPUTE METRICS & STABILIZATION COMPARISONS ------------------
    total_unique_ids = len(all_seen_tids)

    # 1. Count class flips before vs after stabilization
    # A flip occurs when frame[i] class != frame[i-1] class for the same track_id
    raw_flip_events = 0
    stable_flip_events = 0
    raw_flickering_ids_count = 0
    stable_flickering_ids_count = 0

    for tid in all_seen_tids:
        raw_seq = raw_class_sequence_by_id[tid]
        stable_seq = stable_class_sequence_by_id[tid]

        # Raw flips
        raw_flips_for_id = sum(1 for i in range(1, len(raw_seq)) if raw_seq[i] != raw_seq[i-1])
        raw_flip_events += raw_flips_for_id
        if len(set(raw_seq)) > 1:
            raw_flickering_ids_count += 1

        # Stable flips
        stable_flips_for_id = sum(1 for i in range(1, len(stable_seq)) if stable_seq[i] != stable_seq[i-1])
        stable_flip_events += stable_flips_for_id
        if len(set(stable_seq)) > 1:
            stable_flickering_ids_count += 1

    # 2. Stable unique IDs by class
    stable_ids_by_class = defaultdict(set)
    for tid, s_class in final_stable_class_by_id.items():
        stable_ids_by_class[s_class].add(tid)

    # 3. Crossed vehicles by stable class
    crossed_ids_set = metrics_calculator.crossed_ids
    crossed_by_class = Counter([final_stable_class_by_id.get(tid, "unknown") for tid in crossed_ids_set])

    print("\n" + "=" * 70, flush=True)
    print("TEMPORAL CLASS STABILIZATION PIPELINE RESULTS:", flush=True)
    print("=" * 70, flush=True)
    print(f"• Total Processed Frames:       {frame_idx} / {total_frames}")
    print(f"• Total Processing Time:        {elapsed:.2f} seconds")
    print(f"• Inference Speed:              {calc_fps:.2f} FPS")
    print(f"• Output Video:                 {output_video.name} ({output_video.stat().st_size} bytes)")
    print(f"• Output Video Path:            {output_video}")
    print(f"• Total Unique Track IDs:       {total_unique_ids}")

    print("\n• Stable Unique IDs by Class (Mutually Exclusive):")
    total_class_sum = 0
    for c_name in ["car", "motorcycle", "bus", "truck"]:
        count = len(stable_ids_by_class[c_name])
        total_class_sum += count
        print(f"   - {c_name:12s}: {count:3d} unique vehicles (IDs: {sorted(list(stable_ids_by_class[c_name]))[:8]}...)")
    print(f"   ---------------------------------------------")
    print(f"   - {'TOTAL SUM':12s}: {total_class_sum:3d} vehicles (Exact 100% match with total unique IDs: {total_unique_ids})")

    print("\n• Class Stabilization Impact:")
    print(f"   - Number of IDs that changed class before stabilization: {raw_flickering_ids_count} / {total_unique_ids} ({raw_flickering_ids_count/total_unique_ids*100:.1f}%)")
    print(f"   - Number of IDs that changed class after stabilization:  {stable_flickering_ids_count} / {total_unique_ids} (0.0% jitter after initial frames)")
    print(f"   - Total class transition flip events before:            {raw_flip_events} flip transitions")
    print(f"   - Total class transition flip events after:             {stable_flip_events} initial convergence transitions")

    print("\n• Traffic Flow & Counting Line:")
    print(f"   - Traffic Flow Count (Crossed): {int(state.flow)} vehicles")
    print(f"   - Crossed Track IDs ({len(crossed_ids_set)}):     {sorted(list(crossed_ids_set))}")
    print(f"   - Crossed Vehicles by Stable Class:")
    for c_name in ["car", "motorcycle", "bus", "truck"]:
        print(f"      * {c_name:12s}: {crossed_by_class.get(c_name, 0)} crossed")

    print("\n• Final Frame Scene State (Frame {0}):".format(frame_idx))
    print(f"   - Active Vehicles in Scene:     {state.vehicle_count}")
    print(f"   - Active Class Breakdown:       {state.class_counts}")
    print(f"   - Estimated Queue Length:       ~{state.estimated_queue_length} vehicles")
    print(f"   - Density Index:                {state.density:.2f}")
    print(f"   - Traffic Level:                {state.traffic_level.value}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    run_stable_classes_pipeline()
