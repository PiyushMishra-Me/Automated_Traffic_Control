import os
import uuid
import threading
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse
from backend.config import settings
from backend.models.traffic_schemas import (
    ApproachEnum,
    ProcessingJobStatus,
    ApproachTrafficState,
    BatchUploadResponse,
    LiveStreamConfigRequest
)
from backend.core.vision.video_processor import VideoProcessor
from backend.db.repositories.traffic_repo import traffic_repo
from backend.db.repositories.junction_repo import junction_repo
from backend.core.control.emergency_orchestrator import emergency_orchestrator

router = APIRouter(prefix="/api/videos", tags=["Video Processing & Live Feeds"])

# In-memory dictionary for active job tracking
_jobs: dict[str, ProcessingJobStatus] = {}

# In-memory dictionary for live stream camera sources
_live_streams: dict[str, dict] = {}


def _run_processing_job(job_id: str, input_path: Path, output_path: Path, junction_id: str, approach: ApproachEnum):
    job = _jobs.get(job_id)
    if not job:
        return

    def update_progress(progress: float, msg: str):
        job.progress = progress
        job.message = msg

    try:
        job.status = "PROCESSING"
        job.message = "Running YOLO vehicle detection & ByteTrack tracking..."
        
        # Dedicated thread-safe processor instance per job
        processor = VideoProcessor()

        # Get custom counting line if set for junction
        j_doc = junction_repo.get_junction(junction_id)
        counting_line = None
        if j_doc and "custom_counting_lines" in j_doc:
            counting_line = j_doc["custom_counting_lines"].get(approach.value)

        # Get vision bridge for this junction camera
        bridge = emergency_orchestrator.get_or_create_camera_bridge(junction_id, approach)

        final_state: ApproachTrafficState = processor.process_video(
            video_path=input_path,
            approach=approach,
            output_path=output_path,
            counting_line_config=counting_line,
            progress_callback=update_progress,
            emergency_bridge=bridge
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


@router.post("/batch-upload", response_model=BatchUploadResponse)
async def batch_upload_videos(
    junction_id: str = Form(...),
    north_video: Optional[UploadFile] = File(None),
    south_video: Optional[UploadFile] = File(None),
    east_video: Optional[UploadFile] = File(None),
    west_video: Optional[UploadFile] = File(None),
):
    """
    Simultaneously upload and launch video processing for all 4 junction approaches (NORTH, SOUTH, EAST, WEST).
    Each approach runs in parallel in its own thread with an isolated VideoProcessor instance.
    """
    approach_files = [
        (ApproachEnum.NORTH, north_video),
        (ApproachEnum.SOUTH, south_video),
        (ApproachEnum.EAST, east_video),
        (ApproachEnum.WEST, west_video),
    ]

    spawned_jobs: List[ProcessingJobStatus] = []

    for approach, v_file in approach_files:
        if v_file is None or not v_file.filename:
            continue

        job_id = f"job_{uuid.uuid4().hex[:10]}"
        ext = Path(v_file.filename).suffix or ".mp4"
        safe_input_name = f"{junction_id}_{approach.value}_{job_id}{ext}"
        safe_output_name = f"{junction_id}_{approach.value}_{job_id}_annotated.mp4"

        input_path = settings.UPLOAD_DIR / safe_input_name
        output_path = settings.ANNOTATED_DIR / safe_output_name

        try:
            with open(input_path, "wb") as f:
                content = await v_file.read()
                f.write(content)
        except Exception as e:
            continue

        job_status = ProcessingJobStatus(
            job_id=job_id,
            junction_id=junction_id,
            approach=approach,
            status="PENDING",
            progress=0.0,
            message="Queued for parallel processing",
            video_filename=safe_input_name,
            annotated_filename=safe_output_name
        )
        _jobs[job_id] = job_status
        spawned_jobs.append(job_status)

        # Launch parallel background worker with dedicated VideoProcessor
        threading.Thread(
            target=_run_processing_job,
            args=(job_id, input_path, output_path, junction_id, approach),
            daemon=True
        ).start()

    return BatchUploadResponse(
        junction_id=junction_id,
        jobs=spawned_jobs,
        message=f"Queued {len(spawned_jobs)} approach videos for simultaneous inference"
    )


@router.get("/batch-status", response_model=List[ProcessingJobStatus])
def get_batch_job_status(
    job_ids: Optional[str] = Query(None, description="Comma-separated list of job IDs"),
    junction_id: Optional[str] = Query(None, description="Filter jobs by junction ID")
):
    """Query status of multiple processing jobs in a single request."""
    if job_ids:
        ids = [j.strip() for j in job_ids.split(",") if j.strip()]
        return [_jobs[jid] for jid in ids if jid in _jobs]

    if junction_id:
        return [job for job in _jobs.values() if job.junction_id == junction_id]

    return list(_jobs.values())[-10:]


@router.get("/status/{job_id}", response_model=ProcessingJobStatus)
def get_job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@router.post("/live-stream", response_model=dict)
def register_live_stream_feed(payload: LiveStreamConfigRequest):
    """
    Register or update a live camera stream feed (RTSP, HLS, WebRTC, Device Webcam, or Simulation)
    for a specific junction approach.
    """
    key = f"{payload.junction_id}_{payload.approach.value}"
    stream_doc = {
        "junction_id": payload.junction_id,
        "approach": payload.approach.value,
        "stream_type": payload.stream_type,
        "stream_url": payload.stream_url or f"/api/videos/live/{payload.junction_id}_{payload.approach.value}",
        "is_active": payload.is_active,
        "sampling_fps": payload.sampling_fps,
        "status": "CONNECTED" if payload.is_active else "PAUSED"
    }
    _live_streams[key] = stream_doc
    return {
        "status": "success",
        "message": f"Live stream configured for {payload.junction_id} {payload.approach.value}",
        "config": stream_doc
    }


@router.get("/live-stream/{junction_id}", response_model=Dict[str, dict])
def get_junction_live_streams(junction_id: str):
    """Get all configured live stream feeds for a junction."""
    results = {}
    for app in ["NORTH", "SOUTH", "EAST", "WEST"]:
        key = f"{junction_id}_{app}"
        if key in _live_streams:
            results[app] = _live_streams[key]
        else:
            # Default placeholder/simulation config
            results[app] = {
                "junction_id": junction_id,
                "approach": app,
                "stream_type": "STANDBY",
                "stream_url": None,
                "is_active": False,
                "status": "NO_FEED"
            }
    return results


@router.delete("/live-stream/{junction_id}/{approach}", response_model=dict)
def delete_live_stream(junction_id: str, approach: ApproachEnum):
    key = f"{junction_id}_{approach.value}"
    if key in _live_streams:
        del _live_streams[key]
        return {"status": "success", "message": f"Live stream removed for {approach.value}"}
    return {"status": "not_found", "message": "Stream was not registered"}


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

