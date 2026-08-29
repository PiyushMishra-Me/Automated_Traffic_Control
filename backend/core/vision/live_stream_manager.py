import time
import threading
import cv2
import numpy as np
from typing import Dict, Optional, Generator
from datetime import datetime, timezone
from pathlib import Path

from backend.config import settings
from backend.models.traffic_schemas import (
    ApproachEnum,
    ApproachTrafficState,
    TrafficLevelEnum,
    MovementStateEnum,
)
from backend.core.vision.tracker import VehicleTracker
from backend.core.vision.video_processor import VideoProcessor
from backend.core.analytics.traffic_metrics import TrafficMetricsCalculator
from backend.db.repositories.traffic_repo import traffic_repo
from backend.core.vision.emergency_bridge import EmergencyVisionBridge


class LiveStreamWorker:
    """
    Dedicated worker thread that continuously consumes a live video feed (RTSP, HTTP MJPEG, or webcam),
    runs dual-model YOLO + custom ambulance inference on each frame,
    updates traffic_repo database state in real-time,
    and publishes annotated frames for live web streaming.
    """
    def __init__(
        self,
        junction_id: str,
        approach: ApproachEnum,
        stream_url: str,
        sampling_fps: float = 6.0
    ):
        self.junction_id = junction_id
        self.approach = approach if isinstance(approach, ApproachEnum) else ApproachEnum(approach)
        self.stream_url = str(stream_url).strip()
        self.sampling_fps = max(1.0, min(float(sampling_fps), 30.0))
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.latest_jpeg_frame: Optional[bytes] = None
        self.latest_state: Optional[ApproachTrafficState] = None
        self.lock = threading.Lock()
        
        # Dual-Model Tracker (Normal YOLOv8 + Custom Ambulance model)
        self.tracker = VehicleTracker()
        self.processor = VideoProcessor(tracker=self.tracker)
        self.metrics_calculator = TrafficMetricsCalculator(self.approach, None)
        self.emergency_bridge = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"LiveWorker_{self.junction_id}_{self.approach.value}"
        )
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def get_latest_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg_frame

    def get_latest_state(self) -> Optional[ApproachTrafficState]:
        with self.lock:
            return self.latest_state

    def _run(self):
        # Open video capture
        target_src = self.stream_url
        if target_src.isdigit():
            cap = cv2.VideoCapture(int(target_src))
        else:
            cap = cv2.VideoCapture(target_src)

        if not cap.isOpened():
            print(f"[LiveStreamWorker] Warning: Could not open {target_src} for {self.junction_id} {self.approach.value}")
            self.running = False
            return

        print(f"[LiveStreamWorker] Connected successfully to {target_src} for {self.junction_id} {self.approach.value}")
        
        frame_interval = 1.0 / self.sampling_fps
        frame_idx = 0
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        if fps <= 0 or fps > 120:
            fps = 25.0

        self.tracker.set_approach(approach=self.approach, fps=fps)
        self.tracker.reset()
        self.metrics_calculator.reset()
        self.processor._reset_video_state()

        last_db_save_time = 0.0

        try:
            while self.running and cap.isOpened():
                start_t = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.08)
                    continue

                frame_idx += 1

                # 1. Dual-Model Inference: Normal vehicles + Custom Ambulance Model
                try:
                    tracked_vehicles = self.tracker.track(frame, fps=fps)
                except Exception as e:
                    print(f"[LiveStreamWorker] Inference error: {e}")
                    tracked_vehicles = []

                # 2. Preserve Confirmed Ambulance Observations
                self.processor._register_ambulance_observations(tracked_vehicles)

                # 3. Calculate Traffic Metrics & Density
                state = self.metrics_calculator.calculate_metrics(
                    vehicles=tracked_vehicles,
                    frame_width=width,
                    frame_height=height,
                    processed_frames=frame_idx,
                    fps=fps,
                )

                # Check confirmed ambulance count
                ambulance_count = self.processor._get_confirmed_ambulance_count()
                if ambulance_count > 0:
                    state.ambulance_count = max(state.ambulance_count, ambulance_count)
                    state.emergency_detected = True
                    if "ambulance" not in state.class_counts:
                        state.class_counts["ambulance"] = 0
                    state.class_counts["ambulance"] = max(state.class_counts["ambulance"], ambulance_count)

                    # Trigger Interconnected Multi-Junction Green Wave Preemption
                    try:
                        from backend.core.control.emergency_orchestrator import emergency_orchestrator
                        emergency_orchestrator.notify_live_ambulance_detected(
                            junction_id=self.junction_id,
                            approach=self.approach,
                            ambulance_count=ambulance_count
                        )
                    except Exception as ev_err:
                        print(f"[LiveStreamWorker] Emergency orchestrator notification error: {ev_err}")

                # 5. Persist Observation into Traffic Database every ~1 second
                now = time.time()
                if now - last_db_save_time >= 1.0:
                    try:
                        traffic_repo.save_observation(self.junction_id, state)
                        last_db_save_time = now
                    except Exception as db_err:
                        print(f"[LiveStreamWorker] DB save error: {db_err}")

                # 6. Annotate Frame with Bounding Boxes, Labels & HUD
                try:
                    annotated_frame = self.processor._annotate_frame(
                        frame=frame,
                        tracked_vehicles=tracked_vehicles,
                        state=state,
                        approach=self.approach,
                        line_config=self.metrics_calculator.line_config,
                        width=width,
                        height=height,
                    )
                except Exception as e:
                    annotated_frame = frame

                # 7. Encode to JPEG for real-time web streaming
                success, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if success:
                    jpeg_bytes = buffer.tobytes()
                    with self.lock:
                        self.latest_jpeg_frame = jpeg_bytes
                        self.latest_state = state

                # Maintain Target Sampling Rate
                elapsed = time.time() - start_t
                sleep_time = max(0.01, frame_interval - elapsed)
                time.sleep(sleep_time)

        except Exception as err:
            print(f"[LiveStreamWorker] Exception in worker loop: {err}")
        finally:
            cap.release()
            self.running = False
            print(f"[LiveStreamWorker] Stopped worker for {self.junction_id} {self.approach.value}")


class LiveStreamManager:
    """
    Singleton manager for all active real-time CCTV / RTSP video streams across the city.
    """
    def __init__(self):
        self._workers: Dict[str, LiveStreamWorker] = {}
        self._lock = threading.Lock()

    def _get_key(self, junction_id: str, approach: ApproachEnum | str) -> str:
        app_str = approach.value if isinstance(approach, ApproachEnum) else str(approach).upper()
        return f"{junction_id}_{app_str}"

    def start_stream(
        self,
        junction_id: str,
        approach: ApproachEnum | str,
        stream_url: str,
        sampling_fps: float = 6.0
    ) -> LiveStreamWorker:
        key = self._get_key(junction_id, approach)
        app_enum = approach if isinstance(approach, ApproachEnum) else ApproachEnum(approach)
        
        with self._lock:
            if key in self._workers:
                existing = self._workers[key]
                if existing.running and existing.stream_url == stream_url:
                    return existing
                # Stop existing worker if URL changed
                existing.stop()

            worker = LiveStreamWorker(
                junction_id=junction_id,
                approach=app_enum,
                stream_url=stream_url,
                sampling_fps=sampling_fps
            )
            worker.start()
            self._workers[key] = worker
            return worker

    def stop_stream(self, junction_id: str, approach: ApproachEnum | str):
        key = self._get_key(junction_id, approach)
        with self._lock:
            if key in self._workers:
                self._workers[key].stop()
                del self._workers[key]

    def get_worker(self, junction_id: str, approach: ApproachEnum | str) -> Optional[LiveStreamWorker]:
        key = self._get_key(junction_id, approach)
        with self._lock:
            return self._workers.get(key)

    def get_latest_frame(self, junction_id: str, approach: ApproachEnum | str) -> Optional[bytes]:
        worker = self.get_worker(junction_id, approach)
        if worker:
            return worker.get_latest_frame()
        return None


# Global Live Stream Manager Instance
live_stream_manager = LiveStreamManager()
