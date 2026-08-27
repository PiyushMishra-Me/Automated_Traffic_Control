import sys
import time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from backend.models.traffic_schemas import ApproachEnum
from backend.core.vision.video_processor import VideoProcessor
from backend.db.repositories.traffic_repo import traffic_repo
from backend.db.repositories.junction_repo import junction_repo

def test_trial_video_processing():
    uploads_dir = Path("data/uploads")
    annotated_dir = Path("data/annotated")
    
    videos = list(uploads_dir.glob("*.mp4"))
    if not videos:
        print("No trial video found in data/uploads")
        return

    sample_video = videos[0]
    output_video = annotated_dir / f"trial_verified_{sample_video.name}"
    
    print(f"=== TESTING TRIAL VIDEO INFERENCE & ANALYTICS PIPELINE ===")
    print(f"Input Video File: {sample_video} (Size: {sample_video.stat().st_size} bytes)")
    
    processor = VideoProcessor()
    
    t0 = time.time()
    result = processor.process_video(
        video_path=sample_video,
        approach=ApproachEnum.NORTH,
        output_path=output_video,
        progress_callback=lambda p, msg: print(f"  [Progress: {p:.1f}%] {msg}")
    )
    duration = time.time() - t0
    
    result.annotated_video_url = f"/api/videos/annotated/{output_video.name}"
    traffic_repo.save_observation("J-01", result)
    
    print(f"\n=== TRIAL VIDEO PROCESSING SUCCESSFUL ===")
    print(f"Time Taken:             {duration:.2f} seconds")
    print(f"Processed Frames:       {result.processed_frames}")
    print(f"Total Unique Vehicles:  {result.total_unique_vehicles}")
    print(f"Active Vehicles Count:  {result.vehicle_count}")
    print(f"Vehicle Class Breakdown:{result.class_counts}")
    print(f"Traffic Density:        {result.density:.2f}")
    print(f"Estimated Queue Length: {result.estimated_queue_length}")
    print(f"Traffic Flow Rate:      {result.flow}")
    print(f"Traffic Congestion Level:{result.traffic_level.value}")
    print(f"Annotated Video Saved:  {output_video.name} (Size: {output_video.stat().st_size} bytes)")
    print(f"State stored in repo:   J-01 NORTH state updated and available to frontend")

if __name__ == "__main__":
    test_trial_video_processing()
