import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.traffic_schemas import ApproachEnum, ApproachTrafficState
from backend.core.vision.video_processor import VideoProcessor
from backend.db.repositories.traffic_repo import traffic_repo
from backend.core.analytics.junction_aggregator import JunctionAggregator

def run_verification():
    print("=" * 60, flush=True)
    print("RUNNING PHASE 1 END-TO-END PIPELINE VERIFICATION", flush=True)
    print("=" * 60, flush=True)

    input_video = PROJECT_ROOT / "data" / "uploads" / "my_traffic.mp4"
    output_video = PROJECT_ROOT / "data" / "annotated" / "sample_north_annotated.mp4"

    if not input_video.exists():
        raise FileNotFoundError(f"Input test video {input_video} does not exist.")

    processor = VideoProcessor()

    def progress(p, msg):
        print(f"[{p:.1f}%] {msg}", flush=True)

    print(f"\n1. Processing {input_video.name} for Approach NORTH with YOLOv8n + ByteTrack...", flush=True)
    t0 = time.time()
    result_state = processor.process_video(
        video_path=input_video,
        approach=ApproachEnum.NORTH,
        output_path=output_video,
        progress_callback=progress
    )
    elapsed = time.time() - t0
    print(f"Video processing finished in {elapsed:.2f}s", flush=True)

    result_state.annotated_video_url = f"/api/videos/annotated/{output_video.name}"
    
    print("\n2. Storing traffic observation in Data Layer / MongoDB...", flush=True)
    saved_doc = traffic_repo.save_observation("J-01", result_state)
    print(f"Observation persisted with Approach: {saved_doc['approach']}, Level: {saved_doc['traffic_level']}", flush=True)

    print("\n3. Generating Unified 4-Way Junction Traffic State...", flush=True)
    all_states = traffic_repo.get_all_latest_for_junction("J-01")
    parsed_states = {k: ApproachTrafficState(**v) for k, v in all_states.items()}
    junction_state = JunctionAggregator.aggregate("J-01", parsed_states)

    print("\n" + "=" * 60, flush=True)
    print("PHASE 1 VERIFICATION RESULTS:", flush=True)
    print("=" * 60, flush=True)
    print(f"• Input Video:             {input_video.name} ({input_video.stat().st_size} bytes)", flush=True)
    print(f"• Annotated Output Video:  {output_video.name} ({output_video.stat().st_size} bytes)", flush=True)
    print(f"• Processed Frames:        {result_state.processed_frames}", flush=True)
    print(f"• Total Unique Tracked:    {result_state.total_unique_vehicles}", flush=True)
    print(f"• Active Vehicles in Scene:{result_state.vehicle_count}", flush=True)
    print(f"• Vehicle Classes:         {result_state.class_counts}", flush=True)
    print(f"• Traffic Flow (Crossed):  {int(result_state.flow)}", flush=True)
    print(f"• Estimated Queue Length:  ~{result_state.estimated_queue_length} vehicles", flush=True)
    print(f"• Density Index:           {result_state.density:.2f}", flush=True)
    print(f"• Traffic Level:           {result_state.traffic_level.value}", flush=True)
    print(f"• Aggregated Junction Level:{junction_state.aggregate_level.value}", flush=True)
    print("=" * 60, flush=True)
    print("SUCCESS: Full Phase 1 CV Pipeline Verified!", flush=True)

if __name__ == "__main__":
    run_verification()
