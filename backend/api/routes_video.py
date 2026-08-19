import os
import uuid
import threading
from pathlib import Path
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from backend.config import settings
from backend.models.traffic_schemas import ApproachEnum, ProcessingJobStatus, ApproachTrafficState
from backend.core.vision.video_processor import VideoProcessor
from backend.db.repositories.traffic_repo import traffic_repo
from backend.db.repositories.junction_repo import junction_repo

router = APIRouter(prefix="/api/videos", tags=["Video Processing"])

# In-memory dictionary for active job tracking
_jobs: dict[str, ProcessingJobStatus] = {}

# Shared processor instance
_processor = VideoProcessor()

def _run_processing_job(job_id: str, input_path: Path, output_path: Path, junction_id: str, approach: ApproachEnum):
    job = _jobs.get(job_id)
    if not job:
        return

    def update_progress(progress: float, msg: str):
        job.progress = progress
        job.message = msg

    try:
        job.status = "PROCESSING"
        job.message = "Running YOLOv8n vehicle detection and ByteTrack tracking..."
        
        # Get custom counting line if set for junction
        j_doc = junction_repo.get_junction(junction_id)
        counting_line = None
        if j_doc and "custom_counting_lines" in j_doc:
            counting_line = j_doc["custom_counting_lines"].get(approach.value)

        final_state: ApproachTrafficState = _processor.process_video(
            video_path=input_path,
            approach=approach,
            output_path=output_path,
            counting_line_config=counting_line,
            progress_callback=update_progress
        )

        final_state.annotated_video_url = f"/api/videos/annotated/{output_path.name}"
        
        # Save to database
        traffic_repo.save_observation(junction_id, final_state)

        job.status = "COMPLETED"
        job.progress = 100.0
        job.message = "Processing completed successfully."
        job.result = final_state
        job.annotated_filename = output_path.name

    except Exception as e:
        job.status = "FAILED"
        job.message = f"Error during processing: {str(e)}"
        print(f"Error processing job {job_id}: {e}")

@router.post("/upload", response_model=ProcessingJobStatus)
async def upload_and_process_video(
    background_tasks: BackgroundTasks,
    junction_id: str = Form(...),
    approach: ApproachEnum = Form(...),
    video: UploadFile = File(...)
):
    """
    Upload a traffic video assigned to a specific junction approach (NORTH, SOUTH, EAST, WEST)
    and launch vehicle detection and tracking.
    """
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    ext = Path(video.filename).suffix or ".mp4"
    safe_input_name = f"{junction_id}_{approach.value}_{job_id}{ext}"
    safe_output_name = f"{junction_id}_{approach.value}_{job_id}_annotated.mp4"

    input_path = settings.UPLOAD_DIR / safe_input_name
    output_path = settings.ANNOTATED_DIR / safe_output_name

    # Save uploaded file
    try:
        with open(input_path, "wb") as f:
            content = await video.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    job_status = ProcessingJobStatus(
        job_id=job_id,
        junction_id=junction_id,
        approach=approach,
        status="PENDING",
        progress=0.0,
        message="Queued for processing",
        video_filename=safe_input_name,
        annotated_filename=safe_output_name
    )
    _jobs[job_id] = job_status

    # Start processing in background thread
    threading.Thread(
        target=_run_processing_job,
        args=(job_id, input_path, output_path, junction_id, approach),
        daemon=True
    ).start()

    return job_status

@router.get("/status/{job_id}", response_model=ProcessingJobStatus)
def get_job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]

@router.get("/annotated/{filename}")
def stream_annotated_video(filename: str):
    video_path = settings.ANNOTATED_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not found")
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=filename
    )
