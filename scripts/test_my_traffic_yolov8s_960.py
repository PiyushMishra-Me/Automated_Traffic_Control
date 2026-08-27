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

def run_test():
    print("=" * 60, flush=True)
    print("RUNNING YOLOv8s + CUDA + IMGSZ=960 SMALL INFERENCE TEST", flush=True)
    print("=" * 60, flush=True)
    print(f"• Active Model Config:    {settings.MODEL_PATH}", flush=True)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    print(f"• PyTorch CUDA Available: {cuda_available}", flush=True)
    if cuda_available:
        print(f"• CUDA Device Name:       {device_name}", flush=True)
    else:
        print("WARNING: CUDA is not available to PyTorch!", flush=True)

    input_video = PROJECT_ROOT / "data" / "uploads" / "my_traffic.mp4"
    output_video = PROJECT_ROOT / "data" / "annotated" / "my_traffic_test_yolov8s_960.mp4"

    if not input_video.exists():
        raise FileNotFoundError(f"Input video {input_video} not found!")

    output_video.parent.mkdir(parents=True, exist_ok=True)

    tracker = VehicleTracker()
    print(f"• Model Device setting:   device=0", flush=True)
    
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    metrics_calculator = TrafficMetricsCalculator(ApproachEnum.NORTH, None)
    tracker.reset()

    processor = VideoProcessor(tracker=tracker)

    MAX_TEST_FRAMES = 90
    frame_idx = 0
    t0 = time.time()
    cuda_used_during_inference = False
    tracked_ids_by_class = {} # class_name -> set of track_ids

    try:
        while cap.isOpened() and frame_idx < MAX_TEST_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Run tracking with imgsz=960, device=0, model=yolov8s.pt
            tracked_vehicles = tracker.track(frame)

            # Check if model is running on cuda
            if frame_idx == 1 and hasattr(tracker.model, 'device'):
                print(f"• YOLO Model PyTorch Device: {tracker.model.device}", flush=True)
                if 'cuda' in str(tracker.model.device):
                    cuda_used_during_inference = True

            state = metrics_calculator.calculate_metrics(
                vehicles=tracked_vehicles,
                frame_width=width,
                frame_height=height,
                processed_frames=frame_idx,
                fps=fps
            )

            # Record unique track IDs by class
            for v in tracked_vehicles:
                if v.class_name not in tracked_ids_by_class:
                    tracked_ids_by_class[v.class_name] = set()
                tracked_ids_by_class[v.class_name].add(v.track_id)

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
    finally:
        cap.release()
        out.release()

    elapsed = time.time() - t0
    calc_fps = frame_idx / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 60, flush=True)
    print("YOLOv8s TEST RESULTS (imgsz=960, device=0):", flush=True)
    print("=" * 60, flush=True)
    print(f"• Frames Tested:          {frame_idx}", flush=True)
    print(f"• Total Processing Time:  {elapsed:.2f} seconds", flush=True)
    print(f"• Processing Speed (FPS): {calc_fps:.2f} FPS", flush=True)
    print(f"• CUDA Used:              {cuda_used_during_inference} ({device_name})", flush=True)
    print(f"• Output Video:           {output_video.name} ({output_video.stat().st_size} bytes)", flush=True)
    print("• Tracked Vehicles Count by Class:", flush=True)
    for c_name in ["car", "motorcycle", "bus", "truck"]:
        ids = tracked_ids_by_class.get(c_name, set())
        print(f"   - {c_name:12s}: {len(ids)} unique tracked (IDs: {sorted(list(ids))})", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    run_test()
