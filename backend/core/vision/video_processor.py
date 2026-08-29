import os
import cv2
import numpy as np
import shutil
import subprocess

from pathlib import Path
from typing import Any, Callable, Optional

from backend.config import settings
from backend.models.traffic_schemas import (
    ApproachEnum,
    ApproachTrafficState,
    CameraConfig,
    MovementStateEnum,
    TrafficLevelEnum,
)
from backend.core.vision.tracker import (
    VehicleTracker,
    TrackedVehicle,
)
from backend.core.analytics.traffic_metrics import (
    TrafficMetricsCalculator,
)


# =========================================================
# COLORS (BGR)
# =========================================================

COLOR_GREEN = (46, 204, 113)
COLOR_YELLOW = (0, 215, 255)
COLOR_ORANGE = (39, 127, 243)
COLOR_RED = (60, 76, 231)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_BG = (20, 24, 33)
COLOR_ACCENT = (235, 140, 52)


# =========================================================
# CLASS COLORS
# =========================================================

CLASS_COLORS = {
    "car": (245, 130, 49),
    "motorcycle": (60, 180, 75),
    "bus": (230, 25, 75),
    "truck": (145, 30, 180),
    "ambulance": (0, 0, 255),
}


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class VideoProcessor:

    def __init__(
            self,
            tracker: Optional[VehicleTracker] = None,
    ):
        self.tracker = tracker or VehicleTracker()

        # Ambulance IDs confirmed during the current video.
        #
        # The VehicleTracker already performs ambulance
        # confidence filtering and temporal association.
        # We simply preserve the confirmed IDs at the
        # video-processing level so they are not lost when
        # selecting the peak traffic frame.
        self.confirmed_ambulance_ids: set[int] = set()

        # Highest ambulance confidence seen for each
        # confirmed ambulance ID.
        self.ambulance_max_confidence: dict[int, float] = {}


    # =====================================================
    # RESET VIDEO STATE
    # =====================================================

    def _reset_video_state(self) -> None:
        self.confirmed_ambulance_ids.clear()
        self.ambulance_max_confidence.clear()


    # =====================================================
    # REGISTER AMBULANCE OBSERVATIONS
    # =====================================================

    def _register_ambulance_observations(
            self,
            vehicles: list[TrackedVehicle],
    ) -> None:
        """
        Preserve confirmed ambulance detections across the
        entire video.

        The VehicleTracker is responsible for deciding when
        a raw ambulance prediction is confirmed. At this
        point we only collect the confirmed ambulance IDs.

        This prevents the final result from depending on
        whether the peak-traffic frame happened to contain
        the ambulance.
        """

        for vehicle in vehicles:

            if vehicle.class_name != "ambulance":
                continue

            track_id = int(
                vehicle.track_id
            )

            confidence = float(
                vehicle.confidence
            )

            self.confirmed_ambulance_ids.add(
                track_id
            )

            previous_confidence = (
                self.ambulance_max_confidence.get(
                    track_id,
                    0.0,
                )
            )

            self.ambulance_max_confidence[
                track_id
            ] = max(
                previous_confidence,
                confidence,
            )


    # =====================================================
    # GET AMBULANCE COUNT
    # =====================================================

    def _get_confirmed_ambulance_count(
            self,
    ) -> int:

        return len(
            self.confirmed_ambulance_ids
        )


    # =====================================================
    # PROCESS VIDEO
    # =====================================================

    def process_video(
            self,
            video_path: str | Path,
            approach: Optional[ApproachEnum] = None,
            output_path: Optional[str | Path] = None,
            counting_line_config: Optional[dict] = None,
            progress_callback: Optional[
                Callable[[float, str], None]
            ] = None,
            roi: Optional[list] = None,
            camera_config: Optional[
                CameraConfig
            ] = None,
            emergency_bridge: Optional[Any] = None,
    ) -> ApproachTrafficState:

        """
        Process a traffic video.

        The complete video is processed frame by frame.

        Normal vehicles are detected/tracked by VehicleTracker.

        Ambulances are detected by the custom ambulance model
        inside VehicleTracker. Confirmed ambulance IDs are
        preserved across the whole video so the final result
        does not lose an ambulance merely because the highest
        traffic-count frame did not contain it.
        """

        # =================================================
        # VALIDATE OUTPUT
        # =================================================

        if output_path is None:
            raise ValueError(
                "output_path is required"
            )

        video_path = Path(
            video_path
        )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =================================================
        # TEMPORARY OUTPUT
        # =================================================

        temporary_output_path = (
            output_path.with_name(
                f"{output_path.stem}.working.mp4"
            )
        )

        # =================================================
        # OPEN INPUT VIDEO
        # =================================================

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            raise ValueError(
                f"Could not open input video: "
                f"{video_path}"
            )

        total_frames = (
                int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )
                or 1
        )

        fps = (
                cap.get(
                    cv2.CAP_PROP_FPS
                )
                or 25.0
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        # =================================================
        # OUTPUT WRITER
        # =================================================

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            str(
                temporary_output_path
            ),
            fourcc,
            fps,
            (
                width,
                height,
            ),
        )

        if not out.isOpened():

            cap.release()

            raise ValueError(
                "Could not create the temporary "
                "annotated video"
            )

        # =================================================
        # ACTIVE APPROACH
        # =================================================

        active_approach = (
            approach
            if approach is not None
            else (
                camera_config.approach
                if camera_config is not None
                else ApproachEnum.NORTH
            )
        )

        # Explicit fallback for static type checking.
        if active_approach is None:
            active_approach = (
                ApproachEnum.NORTH
            )

        # =================================================
        # COUNTING LINE
        # =================================================

        if counting_line_config is not None:

            active_line_cfg = (
                counting_line_config
            )

        elif (
                camera_config is not None
                and camera_config.counting_line is not None
        ):

            active_line_cfg = (
                camera_config
                .counting_line
                .model_dump()
            )

        else:

            active_line_cfg = None

        # =================================================
        # METRICS CALCULATOR
        # =================================================

        metrics_calculator = (
            TrafficMetricsCalculator(
                active_approach,
                active_line_cfg,
            )
        )

        # =================================================
        # CONFIGURE TRACKER
        # =================================================

        if camera_config is not None:

            self.tracker.set_camera_config(
                camera_config
            )

        else:

            self.tracker.set_approach(
                approach=active_approach,
                fps=fps,
                roi=roi,
            )

        # =================================================
        # RESET
        # =================================================

        self.tracker.reset()

        metrics_calculator.reset()

        self._reset_video_state()

        # =================================================
        # VIDEO STATE
        # =================================================

        frame_idx = 0

        latest_state: Optional[
            ApproachTrafficState
        ] = None

        peak_state: Optional[
            ApproachTrafficState
        ] = None

        total_density = 0.0

        max_queue_length = 0

        # =================================================
        # MAIN LOOP
        # =================================================

        try:

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                frame_idx += 1

                # -----------------------------------------
                # TRACK VEHICLES
                # -----------------------------------------

                tracked_vehicles = (
                    self.tracker.track(
                        frame,
                        fps=fps,
                    )
                )

                # -----------------------------------------
                # PRESERVE CONFIRMED AMBULANCES
                # -----------------------------------------

                self._register_ambulance_observations(
                    tracked_vehicles
                )

                # -----------------------------------------
                # CALCULATE METRICS
                # -----------------------------------------

                state = (
                    metrics_calculator.calculate_metrics(
                        vehicles=tracked_vehicles,
                        frame_width=width,
                        frame_height=height,
                        processed_frames=frame_idx,
                        fps=fps,
                    )
                )

                latest_state = state

                total_density += (
                    float(state.density)
                )

                max_queue_length = max(
                    max_queue_length,
                    int(
                        state.estimated_queue_length
                    ),
                )

                # -----------------------------------------
                # PEAK STATE
                # -----------------------------------------

                if (
                        peak_state is None
                        or state.vehicle_count
                        > peak_state.vehicle_count
                ):

                    peak_state = (
                        state.model_copy(
                            deep=True
                        )
                    )

                # -----------------------------------------
                # EMERGENCY VEHICLE BRIDGE
                # -----------------------------------------

                if emergency_bridge is not None:
                    emergency_bridge.process_frame(
                        vehicles=tracked_vehicles,
                        counting_line_config=metrics_calculator.line_config,
                        frame_width=width,
                        frame_height=height,
                        fps=fps,
                    )

                # -----------------------------------------
                # ANNOTATE
                # -----------------------------------------

                annotated_frame = (
                    self._annotate_frame(
                        frame=frame,
                        tracked_vehicles=(
                            tracked_vehicles
                        ),
                        state=state,
                        approach=active_approach,
                        line_config=(
                            metrics_calculator.line_config
                        ),
                        width=width,
                        height=height,
                    )
                )

                out.write(
                    annotated_frame
                )

                # -----------------------------------------
                # PROGRESS
                # -----------------------------------------

                if (
                        progress_callback
                        and frame_idx % 10 == 0
                ):

                    progress = round(
                        (
                                frame_idx
                                / total_frames
                        )
                        * 100.0,
                        1,
                        )

                    progress_callback(
                        min(
                            99.0,
                            progress,
                        ),
                        (
                            f"Processing frame "
                            f"{frame_idx}/"
                            f"{total_frames}"
                        ),
                    )

        finally:

            cap.release()
            out.release()

        # =================================================
        # ENCODE VIDEO
        # =================================================

        try:

            self._encode_for_browser(
                temporary_output_path,
                output_path,
            )

        finally:

            temporary_output_path.unlink(
                missing_ok=True
            )

        # =================================================
        # EMPTY VIDEO FALLBACK
        # =================================================

        if latest_state is None:

            latest_state = (
                metrics_calculator.calculate_metrics(
                    vehicles=[],
                    frame_width=width,
                    frame_height=height,
                    processed_frames=frame_idx,
                    fps=fps,
                )
            )

            peak_state = (
                latest_state.model_copy(
                    deep=True
                )
            )

        # Make static type checking explicit.
        assert latest_state is not None
        assert peak_state is not None

        # =================================================
        # CONFIRMED AMBULANCES
        # =================================================

        confirmed_ambulance_count = (
            self._get_confirmed_ambulance_count()
        )

        # =================================================
        # FINAL CLASS COUNTS
        # =================================================

        final_class_counts = {
            "car": int(
                peak_state.class_counts.get(
                    "car",
                    0,
                )
            ),
            "motorcycle": int(
                peak_state.class_counts.get(
                    "motorcycle",
                    0,
                )
            ),
            "bus": int(
                peak_state.class_counts.get(
                    "bus",
                    0,
                )
            ),
            "truck": int(
                peak_state.class_counts.get(
                    "truck",
                    0,
                )
            ),
            "ambulance": confirmed_ambulance_count,
        }

        # =================================================
        # FINAL VEHICLE COUNT
        # =================================================
        #
        # peak_state.vehicle_count is normally the count
        # from the busiest frame.
        #
        # If the confirmed ambulance wasn't present in that
        # exact frame, add it once so:
        #
        #   16 cars + 1 truck + 1 ambulance = 18
        #
        # rather than:
        #
        #   16 cars + 1 truck = 17
        #
        # We do NOT add the ambulance repeatedly.
        # =================================================

        peak_ambulance_count = int(
            peak_state.class_counts.get(
                "ambulance",
                0,
            )
        )

        ambulance_difference = max(
            0,
            confirmed_ambulance_count
            - peak_ambulance_count,
            )

        final_vehicle_count = (
                int(
                    peak_state.vehicle_count
                )
                + ambulance_difference
        )

        # =================================================
        # FINAL STATE
        # =================================================

        latest_state = (
            latest_state.model_copy(
                update={
                    "vehicle_count":
                        final_vehicle_count,

                    "class_counts":
                        final_class_counts,

                    "density":
                        round(
                            total_density
                            / max(
                                1,
                                frame_idx,
                            ),
                            3,
                            ),

                    "estimated_queue_length":
                        max_queue_length,
                }
            )
        )

        # =================================================
        # COMPLETE
        # =================================================

        if progress_callback:

            progress_callback(
                100.0,
                "Processing complete",
            )

        return latest_state


    # =====================================================
    # ENCODE FOR BROWSER
    # =====================================================

    @staticmethod
    def _encode_for_browser(
            source_path: Path,
            output_path: Path,
    ) -> None:

        """
        Encode MP4 as H.264/yuv420p for browser playback.
        """

        ffmpeg = (
                settings.FFMPEG_BINARY
                or os.getenv(
            "FFMPEG_BINARY"
        )
                or shutil.which(
            "ffmpeg"
        )
        )

        if not ffmpeg:

            raise RuntimeError(
                "FFmpeg is required to create "
                "browser-playable annotated videos. "
                "Install FFmpeg and ensure its bin "
                "folder is on PATH."
            )

        command = [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ]

        try:

            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=1800,
            )

        except subprocess.CalledProcessError as exc:

            raise RuntimeError(
                "FFmpeg could not encode "
                f"the annotated video: "
                f"{exc.stderr.strip()}"
            ) from exc

        except subprocess.TimeoutExpired as exc:

            raise RuntimeError(
                "FFmpeg timed out while "
                "encoding the annotated video"
            ) from exc


    # =====================================================
    # ANNOTATE FRAME
    # =====================================================

    def _annotate_frame(
            self,
            frame: np.ndarray,
            tracked_vehicles: list,
            state: ApproachTrafficState,
            approach: ApproachEnum,
            line_config: dict,
            width: int,
            height: int,
    ) -> np.ndarray:

        annotated = frame.copy()

        # =================================================
        # COUNTING LINE
        # =================================================

        p1 = (
            int(
                line_config["p1"][0]
                * width
            ),
            int(
                line_config["p1"][1]
                * height
            ),
        )

        p2 = (
            int(
                line_config["p2"][0]
                * width
            ),
            int(
                line_config["p2"][1]
                * height
            ),
        )

        cv2.line(
            annotated,
            p1,
            p2,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            (
                f"COUNT LINE "
                f"({approach.value if hasattr(approach, 'value') else approach})"
            ),
            (
                p1[0],
                max(
                    20,
                    p1[1] - 8,
                    ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # =================================================
        # VEHICLE BOXES
        # =================================================

        for vehicle in tracked_vehicles:

            x1, y1, x2, y2 = [
                int(c)
                for c in vehicle.xyxy
            ]

            color = CLASS_COLORS.get(
                vehicle.class_name,
                (0, 255, 0),
            )

            cv2.rectangle(
                annotated,
                (
                    x1,
                    y1,
                ),
                (
                    x2,
                    y2,
                ),
                color,
                2,
            )

            direction = (
                vehicle.direction.value
                if hasattr(
                    vehicle.direction,
                    "value",
                )
                else str(
                    vehicle.direction
                )
            )

            if (
                    vehicle.class_name
                    == "ambulance"
            ):

                label = (
                    f"AMBULANCE "
                    f"{vehicle.confidence:.2f}"
                )

            else:

                label = (
                    f"#{vehicle.track_id} "
                    f"{vehicle.class_name.upper()} "
                    f"| {direction}"
                )

            (
                text_width,
                text_height,
            ), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                1,
            )

            label_y1 = max(
                0,
                y1 - 20,
                )

            label_y2 = max(
                20,
                y1,
            )

            cv2.rectangle(
                annotated,
                (
                    x1,
                    label_y1,
                ),
                (
                    x1
                    + text_width
                    + 6,
                    label_y2,
                ),
                color,
                -1,
            )

            cv2.putText(
                annotated,
                label,
                (
                    x1 + 3,
                    max(
                        15,
                        y1 - 5,
                        ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                COLOR_WHITE,
                1,
                cv2.LINE_AA,
            )

            # Center point.
            cx = int(
                vehicle.center[0]
            )

            cy = int(
                vehicle.center[1]
            )

            cv2.circle(
                annotated,
                (
                    cx,
                    cy,
                ),
                3,
                (0, 0, 255),
                -1,
            )

        # =================================================
        # HUD PANEL
        # =================================================

        hud_w = 355
        hud_h = 235

        hud_x = 15
        hud_y = 15

        overlay = annotated.copy()

        cv2.rectangle(
            overlay,
            (
                hud_x,
                hud_y,
            ),
            (
                hud_x + hud_w,
                hud_y + hud_h,
            ),
            COLOR_DARK_BG,
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.85,
            annotated,
            0.15,
            0,
            annotated,
        )

        cv2.rectangle(
            annotated,
            (
                hud_x,
                hud_y,
            ),
            (
                hud_x + hud_w,
                hud_y + hud_h,
            ),
            (
                80,
                90,
                110,
            ),
            1,
        )

        # =================================================
        # TRAFFIC LEVEL COLOR
        # =================================================

        level_color = COLOR_GREEN

        if (
                state.traffic_level
                == TrafficLevelEnum.MEDIUM
        ):

            level_color = COLOR_YELLOW

        elif (
                state.traffic_level
                == TrafficLevelEnum.HIGH
        ):

            level_color = COLOR_ORANGE

        elif (
                state.traffic_level
                == TrafficLevelEnum.VERY_HIGH
        ):

            level_color = COLOR_RED

        app_name = (
            approach.value
            if hasattr(
                approach,
                "value",
            )
            else str(
                approach
            )
        )

        # =================================================
        # HEADER
        # =================================================

        cv2.putText(
            annotated,
            f"APPROACH: {app_name}",
            (
                hud_x + 12,
                hud_y + 24,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            (
                f"LEVEL: "
                f"{state.traffic_level.value}"
            ),
            (
                hud_x + 185,
                hud_y + 24,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            level_color,
            2,
            cv2.LINE_AA,
        )

        cv2.line(
            annotated,
            (
                hud_x + 10,
                hud_y + 34,
            ),
            (
                hud_x + hud_w - 10,
                hud_y + 34,
            ),
            (
                70,
                80,
                100,
            ),
            1,
        )

        # =================================================
        # METRICS
        # =================================================

        cv2.putText(
            annotated,
            (
                f"Active Vehicles: "
                f"{state.vehicle_count} "
                f"(In: "
                f"{state.incoming_count} "
                f"| Out: "
                f"{state.outgoing_count})"
            ),
            (
                hud_x + 12,
                hud_y + 55,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (
                220,
                220,
                220,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            (
                f"Stopped: In "
                f"{state.stopped_incoming_count} "
                f"| Out "
                f"{state.stopped_outgoing_count} "
                f"| Parked: "
                f"{state.parked_count}"
            ),
            (
                hud_x + 12,
                hud_y + 75,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (
                220,
                220,
                220,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            (
                f"Est. Queue Length: "
                f"~{state.estimated_queue_length} veh"
            ),
            (
                hud_x + 12,
                hud_y + 95,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (
                220,
                220,
                220,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            (
                f"Traffic Flow: "
                f"{int(state.flow)} "
                f"(In: "
                f"{int(state.incoming_flow)} "
                f"| Out: "
                f"{int(state.outgoing_flow)})"
            ),
            (
                hud_x + 12,
                hud_y + 115,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (
                220,
                220,
                220,
            ),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            annotated,
            (
                f"Traffic Density: "
                f"{state.density:.2f}"
            ),
            (
                hud_x + 12,
                hud_y + 135,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (
                220,
                220,
                220,
            ),
            1,
            cv2.LINE_AA,
        )

        # =================================================
        # CLASS COUNTS
        # =================================================

        classes_str = (
            f"Cars: "
            f"{state.class_counts.get('car', 0)} "
            f"| Bikes: "
            f"{state.class_counts.get('motorcycle', 0)} "
            f"| Bus: "
            f"{state.class_counts.get('bus', 0)} "
            f"| Trk: "
            f"{state.class_counts.get('truck', 0)}"
        )

        cv2.putText(
            annotated,
            classes_str,
            (
                hud_x + 12,
                hud_y + 165,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (
                180,
                200,
                220,
            ),
            1,
            cv2.LINE_AA,
        )

        # =================================================
        # CONFIRMED AMBULANCES
        # =================================================

        ambulance_count = (
            self._get_confirmed_ambulance_count()
        )

        cv2.putText(
            annotated,
            (
                f"Confirmed Ambulances: "
                f"{ambulance_count}"
            ),
            (
                hud_x + 12,
                hud_y + 190,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (
                0,
                0,
                255,
            )
            if ambulance_count > 0
            else (
                180,
                180,
                180,
            ),
            2
            if ambulance_count > 0
            else 1,
            cv2.LINE_AA,
        )

        # =================================================
        # UNIQUE TRACKED
        # =================================================

        cv2.putText(
            annotated,
            (
                f"Unique Tracked: "
                f"{state.total_unique_vehicles} "
                f"| Frame: "
                f"{state.processed_frames}"
            ),
            (
                hud_x + 12,
                hud_y + 215,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (
                140,
                150,
                170,
            ),
            1,
            cv2.LINE_AA,
        )

        return annotated
