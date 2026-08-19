import os
import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Optional
from backend.config import settings
from backend.models.traffic_schemas import ApproachEnum, ApproachTrafficState, TrafficLevelEnum
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
        approach: ApproachEnum,
        output_path: str | Path,
        counting_line_config: Optional[dict] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> ApproachTrafficState:
        """
        Process an input traffic video for a specific approach, generating an annotated output video
        and returning the finalized ApproachTrafficState.
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open input video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup Video Writer (Use avc1 / mp4v codec for browser compatibility)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        metrics_calculator = TrafficMetricsCalculator(approach, counting_line_config)
        self.tracker.reset()

        frame_idx = 0
        latest_state: Optional[ApproachTrafficState] = None

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Run ByteTrack tracking
                tracked_vehicles = self.tracker.track(frame)

                # Compute traffic metrics for this frame
                state = metrics_calculator.calculate_metrics(
                    vehicles=tracked_vehicles,
                    frame_width=width,
                    frame_height=height,
                    processed_frames=frame_idx,
                    fps=fps
                )
                latest_state = state

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

        # If video was processed, return final state
        if latest_state is None:
            latest_state = metrics_calculator.calculate_metrics(
                vehicles=[],
                frame_width=width,
                frame_height=height,
                processed_frames=frame_idx,
                fps=fps
            )

        if progress_callback:
            progress_callback(100.0, "Processing complete")

        return latest_state

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
            f"COUNT LINE ({approach.value})",
            (p1[0], max(20, p1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        # 2. Draw Bounding Boxes and Track IDs
        for v in tracked_vehicles:
            x1, y1, x2, y2 = [int(c) for c in v.xyxy]
            color = CLASS_COLORS.get(v.class_name, (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Header label: ID + Class + Speed/Queue status
            queue_tag = " [QUEUED]" if (v.stationary_frames >= 5) else ""
            label = f"#{v.track_id} {v.class_name.upper()} ({v.confidence:.2f}){queue_tag}"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, max(20, y1)), color, -1)
            cv2.putText(annotated, label, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # Draw center dot
            cx, cy = int(v.center[0]), int(v.center[1])
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)

        # 3. Draw HUD Statistics Overlay Panel
        hud_w, hud_h = 340, 185
        hud_x, hud_y = 15, 15

        # Background rectangle with transparency
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

        # Approach Badge & Level Header
        cv2.putText(annotated, f"APPROACH: {approach.value}", (hud_x + 12, hud_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"LEVEL: {state.traffic_level.value}", (hud_x + 175, hud_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, level_color, 2, cv2.LINE_AA)
        cv2.line(annotated, (hud_x + 10, hud_y + 34), (hud_x + hud_w - 10, hud_y + 34), (70, 80, 100), 1)

        # Metrics rows
        cv2.putText(annotated, f"Active Vehicles: {state.vehicle_count}", (hud_x + 12, hud_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Est. Queue Length: ~{state.estimated_queue_length} veh", (hud_x + 12, hud_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Traffic Flow Count: {int(state.flow)}", (hud_x + 12, hud_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Traffic Density: {state.density:.2f}", (hud_x + 12, hud_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Class breakdown pills
        classes_str = f"Cars: {state.class_counts.get('car', 0)} | Bikes: {state.class_counts.get('motorcycle', 0)} | Bus: {state.class_counts.get('bus', 0)} | Trk: {state.class_counts.get('truck', 0)}"
        cv2.putText(annotated, classes_str, (hud_x + 12, hud_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"Unique Tracked: {state.total_unique_vehicles} | Frame: {state.processed_frames}", (hud_x + 12, hud_y + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 150, 170), 1, cv2.LINE_AA)

        return annotated
