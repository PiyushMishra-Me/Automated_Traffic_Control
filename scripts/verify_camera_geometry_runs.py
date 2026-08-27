import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models.traffic_schemas import ApproachEnum, CameraConfig, MovementStateEnum
from backend.core.vision.tracker import VehicleTracker
from backend.core.vision.video_processor import VideoProcessor
from backend.config import settings

def run_camera_verification(name, video_file, camera_config, output_file):
    print(f"\n==================================================")
    print(f"VERIFYING: {name}")
    print(f"Video:  {video_file}")
    print(f"Camera: {camera_config.camera_id} (ROI: {camera_config.roi})")
    print(f"==================================================")

    tracker = VehicleTracker(camera_config=camera_config)
    processor = VideoProcessor(tracker=tracker)

    start_t = time.time()
    state = processor.process_video(
        video_path=video_file,
        camera_config=camera_config,
        output_path=output_file
    )
    elapsed = time.time() - start_t
    fps = state.processed_frames / elapsed if elapsed > 0 else 0

    print(f"Done in {elapsed:.2f}s ({fps:.2f} FPS).")
    print(f"Total Unique:      {state.total_unique_vehicles}")
    print(f"Active Vehicles:   {state.vehicle_count}")
    print(f"Incoming:          {state.incoming_count}")
    print(f"Outgoing:          {state.outgoing_count}")
    print(f"Stopped Incoming:  {state.stopped_incoming_count}")
    print(f"Stopped Outgoing:  {state.stopped_outgoing_count}")
    print(f"Parked:            {state.parked_count}")
    print(f"Unknown:           {state.unknown_direction_count}")
    print(f"Flow:              {int(state.flow)} (In: {int(state.incoming_flow)} | Out: {int(state.outgoing_flow)})")
    return state

def main():
    # 1. Camera 1: my_traffic.mp4 (configured with tree-clipping ROI [220, 0, 768, 432])
    cam_my_traffic = CameraConfig(
        camera_id="CAM-MY-TRAFFIC-N",
        junction_id="J-CENTRAL",
        approach=ApproachEnum.NORTH,
        roi=[220.0, 0.0, 768.0, 432.0],
        junction_vector=[0.0, 1.0],
        is_bidirectional=False
    )
    state1 = run_camera_verification(
        "my_traffic.mp4 (Corridor Camera)",
        settings.UPLOAD_DIR / "my_traffic.mp4",
        cam_my_traffic,
        settings.ANNOTATED_DIR / "my_traffic_geometry_verified.mp4"
    )

    # 2. Camera 2: bidirectional.mp4 (configured with Full-Frame ROI None)
    cam_bidirectional = CameraConfig(
        camera_id="CAM-BIDIRECTIONAL-N",
        junction_id="J-CENTRAL",
        approach=ApproachEnum.NORTH,
        roi=None, # Full frame
        junction_vector=[0.0, 1.0],
        is_bidirectional=True
    )
    state2 = run_camera_verification(
        "bidirectional.mp4 (Bidirectional Camera)",
        settings.UPLOAD_DIR / "bidirectional.mp4",
        cam_bidirectional,
        settings.ANNOTATED_DIR / "bidirectional_geometry_verified.mp4"
    )

    print("\n==================================================")
    print("ALL VERIFICATION RUNS COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
