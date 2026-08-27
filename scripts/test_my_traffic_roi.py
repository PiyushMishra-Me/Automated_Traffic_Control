import sys
import time
import torch
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.traffic_schemas import ApproachEnum
from backend.core.vision.tracker import VehicleTracker, TrackedVehicle
from backend.core.vision.video_processor import VideoProcessor
from backend.core.analytics.traffic_metrics import TrafficMetricsCalculator
from backend.config import settings

def run_roi_experiment():
    print("=" * 65, flush=True)
    print("RUNNING ROI-BASED VEHICLE DETECTION EXPERIMENT (YOLOv8s + IMGSZ=960)", flush=True)
    print("=" * 65, flush=True)
    print(f"• Active Model Config:    {settings.MODEL_PATH}", flush=True)
    print(f"• Confidence Threshold:   {settings.CONFIDENCE_THRESHOLD}", flush=True)
    print(f"• IOU Threshold:          {settings.IOU_THRESHOLD}", flush=True)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    print(f"• PyTorch CUDA Available: {cuda_available}", flush=True)
    if cuda_available:
        print(f"• CUDA Device Name:       {device_name}", flush=True)
    else:
        print("WARNING: CUDA is not available to PyTorch!", flush=True)

    input_video = PROJECT_ROOT / "data" / "uploads" / "my_traffic.mp4"
    output_video = PROJECT_ROOT / "data" / "annotated" / "my_traffic_test_roi.mp4"

    if not input_video.exists():
        raise FileNotFoundError(f"Input video {input_video} not found!")

    output_video.parent.mkdir(parents=True, exist_ok=True)

    # Instantiate tracker with yolov8s.pt, conf=0.20, imgsz=960, device=0, ByteTrack
    tracker = VehicleTracker(model_path="yolov8s.pt")
    print(f"• Model Device setting:   device=0", flush=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise ValueError(f"Could not open input video {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    # Roadway ROI definition:
    # my_traffic.mp4 is 768x432.
    # The left region (x < 220) contains only trees/foliage.
    # The entire roadway spans x: [220, 768] and y: [0, 432] from distant horizon to bottom exit.
    ROI_X_MIN = 220
    ROI_Y_MIN = 0
    ROI_X_MAX = 768
    ROI_Y_MAX = 432
    roi_w = ROI_X_MAX - ROI_X_MIN
    roi_h = ROI_Y_MAX - ROI_Y_MIN

    print(f"• Frame Dimensions:       {width}x{height}", flush=True)
    print(f"• Selected Roadway ROI:   x=[{ROI_X_MIN}, {ROI_X_MAX}], y=[{ROI_Y_MIN}, {ROI_Y_MAX}] (size: {roi_w}x{roi_h})", flush=True)
    print(f"• Resolution Gain:        ~{width / roi_w:.2f}x horizontal density focused on roadway", flush=True)

    metrics_calculator = TrafficMetricsCalculator(ApproachEnum.NORTH, None)
    tracker.reset()

    processor = VideoProcessor(tracker=tracker)

    MAX_TEST_FRAMES = 90
    frame_idx = 0
    t0 = time.time()
    cuda_used_during_inference = False
    tracked_ids_by_class = {}  # class_name -> set of track_ids
    raw_detections_by_class = {} # class_name -> count of raw detections
    distant_motorcycles_detected = [] # list of (frame, track_id, y, conf)

    try:
        while cap.isOpened() and frame_idx < MAX_TEST_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Extract Roadway ROI
            roi = frame[ROI_Y_MIN:ROI_Y_MAX, ROI_X_MIN:ROI_X_MAX]

            # Run tracking on the ROI (YOLOv8s, imgsz=960, conf=0.20, device=0, ByteTrack)
            tracked_roi_vehicles = tracker.track(roi)

            # Verify PyTorch CUDA device
            if frame_idx == 1 and hasattr(tracker.model, 'device'):
                print(f"• YOLO Model PyTorch Device: {tracker.model.device}", flush=True)
                if 'cuda' in str(tracker.model.device):
                    cuda_used_during_inference = True

            # Translate ROI coordinates back to original 768x432 frame coordinate space
            translated_vehicles = []
            for v in tracked_roi_vehicles:
                orig_xyxy = [
                    v.xyxy[0] + ROI_X_MIN,
                    v.xyxy[1] + ROI_Y_MIN,
                    v.xyxy[2] + ROI_X_MIN,
                    v.xyxy[3] + ROI_Y_MIN
                ]
                orig_center = (v.center[0] + ROI_X_MIN, v.center[1] + ROI_Y_MIN)
                orig_prev_center = (
                    (v.previous_center[0] + ROI_X_MIN, v.previous_center[1] + ROI_Y_MIN)
                    if v.previous_center is not None else None
                )

                translated_vehicles.append(
                    TrackedVehicle(
                        track_id=v.track_id,
                        xyxy=orig_xyxy,
                        confidence=v.confidence,
                        class_id=v.class_id,
                        class_name=v.class_name,
                        center=orig_center,
                        previous_center=orig_prev_center,
                        speed_px=v.speed_px,
                        stationary_frames=v.stationary_frames,
                        crossed_counting_line=v.crossed_counting_line
                    )
                )

                # Record stats
                if v.class_name not in tracked_ids_by_class:
                    tracked_ids_by_class[v.class_name] = set()
                tracked_ids_by_class[v.class_name].add(v.track_id)

                # Check if this is a distant motorcycle (e.g. y < 150)
                if v.class_name == "motorcycle" and orig_center[1] < 150:
                    distant_motorcycles_detected.append((frame_idx, v.track_id, orig_center[1], v.confidence))

            # Compute traffic metrics on full-frame dimensions with translated vehicles
            state = metrics_calculator.calculate_metrics(
                vehicles=translated_vehicles,
                frame_width=width,
                frame_height=height,
                processed_frames=frame_idx,
                fps=fps
            )

            # Annotate full 768x432 original frame with translated detections and HUD
            annotated_frame = processor._annotate_frame(
                frame=frame,
                tracked_vehicles=translated_vehicles,
                state=state,
                approach=ApproachEnum.NORTH,
                line_config=metrics_calculator.line_config,
                width=width,
                height=height
            )

            # Optional subtle indicator of ROI boundary on the full frame
            # cv2.rectangle(annotated_frame, (ROI_X_MIN, ROI_Y_MIN), (ROI_X_MAX, ROI_Y_MAX), (100, 100, 100), 1)

            out.write(annotated_frame)
    finally:
        cap.release()
        out.release()

    elapsed = time.time() - t0
    calc_fps = frame_idx / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 65, flush=True)
    print("ROI-BASED DETECTION EXPERIMENT RESULTS (YOLOv8s, imgsz=960, device=0):", flush=True)
    print("=" * 65, flush=True)
    print(f"• Frames Tested:          {frame_idx}", flush=True)
    print(f"• Total Processing Time:  {elapsed:.2f} seconds", flush=True)
    print(f"• Processing Speed (FPS): {calc_fps:.2f} FPS", flush=True)
    print(f"• CUDA Used:              {cuda_used_during_inference} ({device_name})", flush=True)
    print(f"• Output Video:           {output_video.name} ({output_video.stat().st_size} bytes)", flush=True)
    print(f"• Output Video Path:      {output_video}", flush=True)
    print("• Tracked Vehicles Count by Class:", flush=True)
    for c_name in ["car", "motorcycle", "bus", "truck"]:
        ids = tracked_ids_by_class.get(c_name, set())
        print(f"   - {c_name:12s}: {len(ids)} unique tracks (IDs: {sorted(list(ids))})", flush=True)
    
    print("\n• Distant Motorcycle Detections (top of roadway, y < 150):", flush=True)
    print(f"   - Distant motorcycle detection instances: {len(distant_motorcycles_detected)} across {frame_idx} frames", flush=True)
    distant_ids = set([d[1] for d in distant_motorcycles_detected])
    print(f"   - Unique distant motorcycle track IDs:    {sorted(list(distant_ids))}", flush=True)
    for sample in distant_motorcycles_detected[:8]:
        print(f"      Frame {sample[0]:02d} | Track ID {sample[1]} | y-coord: {sample[2]:.1f}px | Conf: {sample[3]:.2f}")
    if len(distant_motorcycles_detected) > 8:
        print(f"      ... and {len(distant_motorcycles_detected) - 8} more distant motorcycle detections")
    print("=" * 65, flush=True)

if __name__ == "__main__":
    run_roi_experiment()
