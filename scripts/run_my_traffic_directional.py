import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models.traffic_schemas import ApproachEnum
from backend.core.vision.video_processor import VideoProcessor
from backend.core.vision.tracker import VehicleTracker
from backend.config import settings

def main():
    input_video = settings.UPLOAD_DIR / "my_traffic.mp4"
    output_video = settings.ANNOTATED_DIR / "my_traffic_directional.mp4"

    print("==================================================")
    print("RUNNING DIRECTIONAL STATE VIDEO PROCESSING")
    print(f"Input:  {input_video}")
    print(f"Output: {output_video}")
    print("==================================================")

    tracker = VehicleTracker(roi=settings.DETECTION_ROI, approach=ApproachEnum.NORTH)
    processor = VideoProcessor(tracker=tracker)

    start_time = time.time()
    
    def progress_callback(prog, msg):
        print(f"[{prog:5.1f}%] {msg}")

    final_state = processor.process_video(
        video_path=input_video,
        approach=ApproachEnum.NORTH,
        output_path=output_video,
        progress_callback=progress_callback
    )

    elapsed_time = time.time() - start_time
    fps = final_state.processed_frames / elapsed_time if elapsed_time > 0 else 0.0

    print("\n==================================================")
    print("PROCESSING COMPLETE - METRICS REPORT")
    print("==================================================")
    print(f"Processing Time:        {elapsed_time:.2f} seconds")
    print(f"Processing Speed:       {fps:.2f} FPS")
    print(f"Processed Frames:       {final_state.processed_frames}")
    print(f"Total Unique Tracks:    {final_state.total_unique_vehicles}")
    print(f"Active Vehicles:        {final_state.vehicle_count}")
    print(f"Incoming:               {final_state.incoming_count}")
    print(f"Outgoing:               {final_state.outgoing_count}")
    print(f"Stopped Incoming:       {final_state.stopped_incoming_count}")
    print(f"Stopped Outgoing:       {final_state.stopped_outgoing_count}")
    print(f"Parked:                 {final_state.parked_count}")
    print(f"Unknown:                {final_state.unknown_direction_count}")
    print(f"Traffic Flow (Total):   {int(final_state.flow)}")
    print(f"Incoming Flow:          {int(final_state.incoming_flow)}")
    print(f"Outgoing Flow:          {int(final_state.outgoing_flow)}")
    print(f"Density:                {final_state.density:.2f}")
    print(f"Traffic Level:          {final_state.traffic_level.value}")
    print(f"Cars:                   {final_state.class_counts.get('car', 0)}")
    print(f"Motorcycles:            {final_state.class_counts.get('motorcycle', 0)}")
    print(f"Buses:                  {final_state.class_counts.get('bus', 0)}")
    print(f"Trucks:                 {final_state.class_counts.get('truck', 0)}")
    print("==================================================")

if __name__ == "__main__":
    main()
