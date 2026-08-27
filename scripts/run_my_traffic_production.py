import sys
import time
import torch
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.traffic_schemas import ApproachEnum
from backend.core.vision.tracker import VehicleTracker
from backend.core.vision.video_processor import VideoProcessor
from backend.core.analytics.traffic_metrics import TrafficMetricsCalculator
from backend.config import settings

def run_production_pipeline():
    print("=" * 65, flush=True)
    print("RUNNING FULL PRODUCTION PIPELINE (INTEGRATED ROI TRACKER)", flush=True)
    print("=" * 65, flush=True)
    print(f"• Active Model Config:    {settings.MODEL_PATH}", flush=True)
    print(f"• Confidence Threshold:   {settings.CONFIDENCE_THRESHOLD}", flush=True)
    print(f"• IOU Threshold:          {settings.IOU_THRESHOLD}", flush=True)
    print(f"• Detection ROI Config:   {settings.DETECTION_ROI}", flush=True)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    print(f"• PyTorch CUDA Available: {cuda_available}", flush=True)
    if cuda_available:
        print(f"• CUDA Device Name:       {device_name}", flush=True)

    input_video = PROJECT_ROOT / "data" / "uploads" / "my_traffic.mp4"
    output_video = PROJECT_ROOT / "data" / "annotated" / "my_traffic_roi_production.mp4"

    if not input_video.exists():
        raise FileNotFoundError(f"Input video {input_video} not found!")

    output_video.parent.mkdir(parents=True, exist_ok=True)

    tracker = VehicleTracker()
    processor = VideoProcessor(tracker=tracker)

    cap = cv2.VideoCapture(str(input_video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    metrics_calculator = TrafficMetricsCalculator(ApproachEnum.NORTH, None)
    tracker.reset()

    print(f"• Total Video Frames:     {total_frames}", flush=True)
    print(f"• Frame Dimensions:       {width}x{height}", flush=True)
    print(f"• Output Video:           {output_video.name}", flush=True)

    tracked_ids_by_class = {} # class_name -> set of track_ids
    distant_motorcycles = []  # list of (frame, track_id, y, conf)

    frame_idx = 0
    t0 = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Run production VehicleTracker (with automatic ROI & translation)
            tracked_vehicles = tracker.track(frame)

            # Record track IDs and distant motorcycles
            for v in tracked_vehicles:
                if v.class_name not in tracked_ids_by_class:
                    tracked_ids_by_class[v.class_name] = set()
                tracked_ids_by_class[v.class_name].add(v.track_id)

                if v.class_name == "motorcycle" and v.center[1] < 150:
                    distant_motorcycles.append((frame_idx, v.track_id, v.center[1], v.confidence))

            # Compute traffic metrics for full frame
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

    print("\n" + "=" * 65, flush=True)
    print("FULL PRODUCTION PIPELINE RESULTS (my_traffic_roi_production.mp4):", flush=True)
    print("=" * 65, flush=True)
    print(f"• Total Frames Processed: {frame_idx}", flush=True)
    print(f"• Total Processing Time:  {elapsed:.2f} seconds", flush=True)
    print(f"• Overall Speed (FPS):    {calc_fps:.2f} FPS", flush=True)
    print(f"• Output Video:           {output_video.name} ({output_video.stat().st_size} bytes)", flush=True)
    print(f"• Output Video Path:      {output_video}", flush=True)
    print(f"• Total Unique Track IDs: {state.total_unique_vehicles}", flush=True)
    print(f"• Traffic Flow (Crossed): {int(state.flow)} vehicles", flush=True)
    print(f"• Unique Tracks by Vehicle Class:")
    for c_name in ["car", "motorcycle", "bus", "truck"]:
        ids = tracked_ids_by_class.get(c_name, set())
        print(f"   - {c_name:12s}: {len(ids)} unique tracks (IDs: {sorted(list(ids))})", flush=True)
    
    distant_ids = set([d[1] for d in distant_motorcycles])
    print(f"\n• Distant Motorcycle Detections (top of road, y < 150px):", flush=True)
    print(f"   - Total Detection Instances: {len(distant_motorcycles)} across {frame_idx} frames", flush=True)
    print(f"   - Unique Distant Motorcycle IDs: {sorted(list(distant_ids))}", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    run_production_pipeline()
