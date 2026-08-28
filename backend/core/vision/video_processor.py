import os
import cv2
import numpy as np
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional
from backend.config import settings
from backend.models.traffic_schemas import ApproachEnum, ApproachTrafficState, CameraConfig, TrafficLevelEnum
from backend.core.vision.tracker import VehicleTracker
from backend.core.analytics.traffic_metrics import TrafficMetricsCalculator

# Colors (BGR)
COLOR_GREEN = (46, 204, 113)
COLOR_YELLOW = (0, 215, 255)
COLOR_ORANGE = (39, 127, 243)
COLOR_RED = (60, 76, 231)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_BG = (20, 24, 33)
COLOR_ACCENT = (235, 140, 52)

CLASS_COLORS = {
    "car": (245, 130, 49),
    "motorcycle": (60, 180, 75),
    "bus": (230, 25, 75),
    "truck": (145, 30, 180)
}

class VideoProcessor:
    def __init__(self, tracker: Optional[VehicleTracker] = None):
        self.tracker = tracker or VehicleTracker()

    def process_video(
        self,
        video_path: str | Path,
        approach: Optional[ApproachEnum] = None,
        output_path: str | Path = None,
        counting_line_config: Optional[dict] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        roi: Optional[list] = None,
        camera_config: Optional[CameraConfig] = None,
        emergency_bridge: Optional[Any] = None
    ) -> ApproachTrafficState:
        """
        Process an input traffic video for a specific approach or camera configuration,
        generating an annotated output video and returning the finalized ApproachTrafficState.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # OpenCV's mp4v output is not consistently supported by web browsers.
        # It is only an intermediate file; FFmpeg creates the final H.264 video.
        temporary_output_path = output_path.with_name(f"{output_path.stem}.working.mp4")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # FFmpeg performs the final browser-compatible H.264 encode below.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temporary_output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise ValueError("Could not create the temporary annotated video")

        # Determine approach and counting line
        active_approach = approach or (camera_config.approach if camera_config else ApproachEnum.NORTH)
        active_line_cfg = counting_line_config or (camera_config.counting_line.dict() if camera_config and camera_config.counting_line else None)

        metrics_calculator = TrafficMetricsCalculator(active_approach, active_line_cfg)
        if camera_config is not None:
            self.tracker.set_camera_config(camera_config)
        else:
            self.tracker.set_approach(approach=active_approach, fps=fps, roi=roi)
        self.tracker.reset()

        frame_idx = 0
        latest_state: Optional[ApproachTrafficState] = None
        peak_state: Optional[ApproachTrafficState] = None
        total_density = 0.0
        max_queue_length = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Run ByteTrack tracking with directional state
                tracked_vehicles = self.tracker.track(frame, fps=fps)

                # Compute traffic metrics for this frame
                state = metrics_calculator.calculate_metrics(
                    vehicles=tracked_vehicles,
                    frame_width=width,
                    frame_height=height,
                    processed_frames=frame_idx,
                    fps=fps
                )
                latest_state = state
                total_density += state.density
                max_queue_length = max(max_queue_length, state.estimated_queue_length)
                if peak_state is None or state.vehicle_count > peak_state.vehicle_count:
                    peak_state = state.model_copy(deep=True)

                # Process emergency vehicle detection/ETA/passage bridge if configured
                if emergency_bridge is not None:
                    emergency_bridge.process_frame(
                        vehicles=tracked_vehicles,
                        counting_line_config=metrics_calculator.line_config,
                        frame_width=width,
                        frame_height=height,
                        fps=fps
                    )

                # Draw Visual Annotations
                annotated_frame = self._annotate_frame(
                    frame=frame,
                    tracked_vehicles=tracked_vehicles,
                    state=state,
                    approach=approach,
                    line_config=metrics_calculator.line_config,
                    width=width,
                    height=height
                )

                out.write(annotated_frame)

                # Progress callback
                if progress_callback and frame_idx % 10 == 0:
                    prog = round((frame_idx / total_frames) * 100, 1)
                    progress_callback(min(99.0, prog), f"Processing frame {frame_idx}/{total_frames}")

        finally:
            cap.release()
            out.release()

        try:
            self._encode_for_browser(temporary_output_path, output_path)
        finally:
            temporary_output_path.unlink(missing_ok=True)

        # If video was processed, return final state
        if latest_state is None:
            latest_state = metrics_calculator.calculate_metrics(
                vehicles=[],
                frame_width=width,
                frame_height=height,
                processed_frames=frame_idx,
                fps=fps
            )
            peak_state = latest_state.model_copy(deep=True)

        # A completed upload represents an entire recorded video, not a live
        # frame. The final frame is often empty, so expose useful whole-video
        # values to the dashboard: peak traffic, average density, max queue,
        # and final cumulative flow.
        assert peak_state is not None
        latest_state = latest_state.model_copy(update={
            "vehicle_count": peak_state.vehicle_count,
            "class_counts": peak_state.class_counts,
            "density": round(total_density / max(1, frame_idx), 3),
            "estimated_queue_length": max_queue_length,
        })

        if progress_callback:
            progress_callback(100.0, "Processing complete")

        return latest_state

    @staticmethod
    def _encode_for_browser(source_path: Path, output_path: Path) -> None:
        """Encode an MP4 as H.264/yuv420p so it plays in standard browsers."""
        ffmpeg = settings.FFMPEG_BINARY or os.getenv("FFMPEG_BINARY") or shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "FFmpeg is required to create browser-playable annotated videos. "
                "Install FFmpeg and ensure its bin folder is on PATH."
            )

        command = [
            ffmpeg, "-y", "-i", str(source_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(output_path),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"FFmpeg could not encode the annotated video: {exc.stderr.strip()}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("FFmpeg timed out while encoding the annotated video") from exc

    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracked_vehicles: list,
        state: ApproachTrafficState,
        approach: ApproachEnum,
        line_config: dict,
        width: int,
        height: int
    ) -> np.ndarray:
        annotated = frame.copy()

        # 1. Draw Virtual Counting Line
        p1 = (int(line_config["p1"][0] * width), int(line_config["p1"][1] * height))
        p2 = (int(line_config["p2"][0] * width), int(line_config["p2"][1] * height))
        cv2.line(annotated, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            f"COUNT LINE ({approach.value if hasattr(approach, 'value') else approach})",
            (p1[0], max(20, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        # 2. Draw Bounding Boxes, Track IDs, and Directional Movement State
        for v in tracked_vehicles:
            x1, y1, x2, y2 = [int(c) for c in v.xyxy]
            color = CLASS_COLORS.get(v.class_name, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Direction state tag
            dir_str = v.direction.value if hasattr(v.direction, 'value') else str(v.direction)
            label = f"#{v.track_id} {v.class_name.upper()} | {dir_str}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), color, -1)
            cv2.putText(annotated, label, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw center dot
            cx, cy = int(v.center[0]), int(v.center[1])
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

        # 3. Draw HUD Statistics Overlay Panel
        hud_w, hud_h = 355, 215
        hud_x, hud_y = 15, 15

        overlay = annotated.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)
        cv2.rectangle(annotated, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (80, 90, 110), 1)

        # Traffic Level Color
        level_color = COLOR_GREEN
        if state.traffic_level == TrafficLevelEnum.MEDIUM:
            level_color = COLOR_YELLOW
        elif state.traffic_level == TrafficLevelEnum.HIGH:
            level_color = COLOR_ORANGE
        elif state.traffic_level == TrafficLevelEnum.VERY_HIGH:
            level_color = COLOR_RED

        app_name = approach.value if hasattr(approach, 'value') else str(approach)
        # Approach Badge & Level Header
        cv2.putText(annotated, f"APPROACH: {app_name}", (hud_x + 12, hud_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"LEVEL: {state.traffic_level.value}", (hud_x + 185, hud_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, level_color, 2, cv2.LINE_AA)
        cv2.line(annotated, (hud_x + 10, hud_y + 34), (hud_x + hud_w - 10, hud_y + 34), (70, 80, 100), 1)

        # Active & Directional Rows
        cv2.putText(annotated, f"Active Vehicles: {state.vehicle_count} (In: {state.incoming_count} | Out: {state.outgoing_count})", (hud_x + 12, hud_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Stopped: In {state.stopped_incoming_count} | Out {state.stopped_outgoing_count} | Parked: {state.parked_count}", (hud_x + 12, hud_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Est. Queue Length: ~{state.estimated_queue_length} veh", (hud_x + 12, hud_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Traffic Flow: {int(state.flow)} (In: {int(state.incoming_flow)} | Out: {int(state.outgoing_flow)})", (hud_x + 12, hud_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Traffic Density: {state.density:.2f}", (hud_x + 12, hud_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

        # Class breakdown pills
        classes_str = f"Cars: {state.class_counts.get('car', 0)} | Bikes: {state.class_counts.get('motorcycle', 0)} | Bus: {state.class_counts.get('bus', 0)} | Trk: {state.class_counts.get('truck', 0)}"
        cv2.putText(annotated, classes_str, (hud_x + 12, hud_y + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Unique Tracked: {state.total_unique_vehicles} | Frame: {state.processed_frames}", (hud_x + 12, hud_y + 192), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 150, 170), 1, cv2.LINE_AA)

        return annotated
