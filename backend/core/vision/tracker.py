from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ultralytics import YOLO
from backend.config import settings
from backend.models.traffic_schemas import ApproachEnum, CameraConfig, MovementStateEnum

@dataclass
class TrackedVehicle:
    track_id: int
    xyxy: list[float]
    confidence: float
    class_id: int
    class_name: str
    center: tuple[float, float]
    previous_center: Optional[tuple[float, float]] = None
    speed_px: float = 0.0  # approximate speed in pixels per frame
    stationary_frames: int = 0
    crossed_counting_line: bool = False
    raw_class_id: Optional[int] = None
    raw_class_name: Optional[str] = None
    # Directional movement & parking state fields
    direction: MovementStateEnum = MovementStateEnum.UNKNOWN
    stopped_duration_seconds: float = 0.0
    last_moving_direction: Optional[MovementStateEnum] = None
    is_parked: bool = False

_DEFAULT_ROI = object()

class VehicleTracker:
    def __init__(
        self,
        model_path: Optional[str] = None,
        roi: Optional[Union[List[float], List[int]]] = _DEFAULT_ROI,
        approach: Optional[ApproachEnum] = None,
        junction_vector: Optional[List[float]] = None,
        fps: float = 25.0,
        camera_config: Optional[CameraConfig] = None
    ):
        self.model_path = model_path or settings.MODEL_PATH
        self.model = YOLO(self.model_path)
        self.target_classes = settings.TARGET_CLASSES
        self.class_names = settings.CLASS_NAMES
        self.conf_threshold = settings.CONFIDENCE_THRESHOLD
        self.iou_threshold = settings.IOU_THRESHOLD
        
        # Camera Configuration
        self.camera_config = camera_config
        if camera_config is not None:
            self.approach = camera_config.approach
            j_vec = camera_config.junction_vector or [0.0, 1.0]
            self.fps = camera_config.fps or fps
            self.roi = camera_config.roi
        else:
            self.approach = approach or ApproachEnum.NORTH
            approach_key = self.approach.value if hasattr(self.approach, 'value') else str(self.approach)
            j_vec = junction_vector or settings.DEFAULT_JUNCTION_VECTORS.get(approach_key, [0.0, 1.0])
            self.fps = fps or 25.0
            self.roi = getattr(settings, "DETECTION_ROI", None) if roi is _DEFAULT_ROI else roi

        self.junction_vector = np.array(j_vec, dtype=np.float32)
        norm = np.linalg.norm(self.junction_vector)
        if norm > 0:
            self.junction_vector = self.junction_vector / norm

        # Historical memory for tracks
        self.track_history: Dict[int, List[Tuple[float, float]]] = {}
        self.stationary_counts: Dict[int, int] = {}
        self.crossed_ids: set[int] = set()
        
        # Temporal class voting memory: track_id -> {class_id: vote_count}
        self.class_votes: Dict[int, Dict[int, int]] = {}
        
        # Directional & Parking memory
        self.last_moving_direction: Dict[int, MovementStateEnum] = {}
        self.stopped_frames_count: Dict[int, int] = {}
        self.parked_status: Dict[int, bool] = {}
        self.stationary_ref_center: Dict[int, Tuple[float, float]] = {}

    def set_roi(self, roi: Optional[Union[List[float], List[int]]]):
        """Set or clear the detection region of interest (ROI). Set to None for full-frame."""
        self.roi = roi

    def set_camera_config(self, camera_config: CameraConfig):
        """Configure camera topology and geometry from CameraConfig."""
        self.camera_config = camera_config
        self.approach = camera_config.approach
        self.fps = camera_config.fps or self.fps
        self.roi = camera_config.roi
        j_vec = camera_config.junction_vector or [0.0, 1.0]
        self.junction_vector = np.array(j_vec, dtype=np.float32)
        norm = np.linalg.norm(self.junction_vector)
        if norm > 0:
            self.junction_vector = self.junction_vector / norm

    def set_approach(
        self,
        approach: ApproachEnum,
        junction_vector: Optional[List[float]] = None,
        fps: Optional[float] = None,
        roi: Optional[Union[List[float], List[int]]] = None
    ):
        """Set approach topology, junction direction vector, camera fps, and optional ROI."""
        self.approach = approach
        approach_key = approach.value if hasattr(approach, 'value') else str(approach)
        j_vec = junction_vector or settings.DEFAULT_JUNCTION_VECTORS.get(approach_key, [0.0, 1.0])
        self.junction_vector = np.array(j_vec, dtype=np.float32)
        norm = np.linalg.norm(self.junction_vector)
        if norm > 0:
            self.junction_vector = self.junction_vector / norm
        if fps is not None:
            self.fps = fps
        if roi is not None:
            self.roi = roi

    def reset(self):
        """Reset internal tracker history between video sessions."""
        self.track_history.clear()
        self.stationary_counts.clear()
        self.crossed_ids.clear()
        self.class_votes.clear()
        self.last_moving_direction.clear()
        self.stopped_frames_count.clear()
        self.parked_status.clear()
        self.stationary_ref_center.clear()

    def track(self, frame: np.ndarray, fps: Optional[float] = None) -> List[TrackedVehicle]:
        """
        Track vehicles across the given frame using ByteTrack with camera-specific ROI detection,
        temporal class stabilization, and per-vehicle directional-state determination.
        Translates detected bounding boxes back to the original full-frame coordinate space.
        Returns persistent TrackedVehicle objects in full-frame coordinates.
        """
        h, w = frame.shape[:2]
        x_offset, y_offset = 0, 0
        source_img = frame
        current_fps = fps or self.fps or 25.0
        edge_margin = getattr(settings, "EDGE_MARGIN_PIXELS", 25.0)

        # Crop ROI if defined and not spanning the entire frame
        if self.roi and len(self.roi) == 4:
            rx1, ry1, rx2, ry2 = self.roi
            # Support normalized coordinates (0.0 to 1.0)
            if all(0.0 <= v <= 1.0 for v in self.roi) and any(0.0 < v < 1.0 for v in self.roi):
                x1 = max(0, min(int(rx1 * w), w))
                y1 = max(0, min(int(ry1 * h), h))
                x2 = max(x1, min(int(rx2 * w), w))
                y2 = max(y1, min(int(ry2 * h), h))
            else:
                x1 = max(0, min(int(rx1), w))
                y1 = max(0, min(int(ry1), h))
                x2 = max(x1, min(int(rx2), w))
                y2 = max(y1, min(int(ry2), h))

            if (x2 - x1) > 0 and (y2 - y1) > 0 and not (x1 == 0 and y1 == 0 and x2 >= w and y2 >= h):
                source_img = frame[y1:y2, x1:x2]
                x_offset, y_offset = x1, y1

        results = self.model.track(
            source=source_img,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=settings.INFERENCE_IMAGE_SIZE,
            classes=self.target_classes,
            imgsz=960,
            device=0,
            verbose=False
        )

        tracked_vehicles: List[TrackedVehicle] = []
        if not results or len(results) == 0:
            return tracked_vehicles

        r = results[0]
        if r.boxes is None or r.boxes.id is None:
            # If no tracks assigned yet in early frames, boxes might still exist without track IDs
            return tracked_vehicles

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.int().cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.int().cpu().numpy()

        for box, track_id, conf, cls_id in zip(boxes, ids, confs, clss):
            cls_id = int(cls_id)
            if cls_id not in self.class_names:
                continue

            track_id = int(track_id)

            # Temporal class stabilization: accumulate votes for this track_id
            if track_id not in self.class_votes:
                self.class_votes[track_id] = {}
            self.class_votes[track_id][cls_id] = self.class_votes[track_id].get(cls_id, 0) + 1

            # Determine dominant / stable class for this track_id
            stable_cls_id = max(self.class_votes[track_id], key=self.class_votes[track_id].get)
            stable_cls_name = self.class_names[stable_cls_id]

            # Translate ROI bounding box back to full-frame coordinate space
            xyxy = [
                float(box[0] + x_offset),
                float(box[1] + y_offset),
                float(box[2] + x_offset),
                float(box[3] + y_offset)
            ]
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            curr_center = (cx, cy)

            # Compute displacement / speed in full-frame coordinate space
            prev_center = None
            speed_px = 0.0
            if track_id in self.track_history and len(self.track_history[track_id]) > 0:
                prev_center = self.track_history[track_id][-1]
                speed_px = float(np.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1]))
            
            # Update history (keep last 30 positions)
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(curr_center)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id].pop(0)

            history = self.track_history[track_id]

            # Update stationary frame count for queue length
            if speed_px < settings.QUEUE_SPEED_THRESHOLD:
                self.stationary_counts[track_id] = self.stationary_counts.get(track_id, 0) + 1
            else:
                self.stationary_counts[track_id] = max(0, self.stationary_counts.get(track_id, 0) - 1)

            # -------------------------------------------------------------
            # Directional Movement & Parking Determination
            # -------------------------------------------------------------
            is_stationary = (speed_px < settings.MOVEMENT_SPEED_THRESHOLD)

            # Check noise tolerance while stopped
            if is_stationary:
                if track_id not in self.stationary_ref_center:
                    self.stationary_ref_center[track_id] = curr_center
                ref_dist = float(np.hypot(curr_center[0] - self.stationary_ref_center[track_id][0],
                                          curr_center[1] - self.stationary_ref_center[track_id][1]))
                if ref_dist > settings.NOISE_DISPLACEMENT_THRESHOLD and speed_px >= settings.MOVEMENT_SPEED_THRESHOLD:
                    is_stationary = False
                    self.stationary_ref_center.pop(track_id, None)

            if is_stationary:
                self.stopped_frames_count[track_id] = self.stopped_frames_count.get(track_id, 0) + 1
                stopped_duration = self.stopped_frames_count[track_id] / current_fps

                if stopped_duration > settings.PARKED_DURATION_SECONDS:
                    self.parked_status[track_id] = True
                    movement_state = MovementStateEnum.PARKED
                else:
                    last_dir = self.last_moving_direction.get(track_id)
                    if last_dir == MovementStateEnum.INCOMING:
                        movement_state = MovementStateEnum.STOPPED_INCOMING
                    elif last_dir == MovementStateEnum.OUTGOING:
                        movement_state = MovementStateEnum.STOPPED_OUTGOING
                    else:
                        movement_state = MovementStateEnum.UNKNOWN
            else:
                # Vehicle is moving
                self.stopped_frames_count[track_id] = 0
                self.stationary_ref_center.pop(track_id, None)

                # If previously parked, unmark and recalculate direction from new movement
                if self.parked_status.get(track_id, False):
                    self.parked_status[track_id] = False
                    self.last_moving_direction.pop(track_id, None)

                if len(history) < settings.MIN_TRAJECTORY_POINTS:
                    movement_state = self.last_moving_direction.get(track_id, MovementStateEnum.UNKNOWN)
                else:
                    # Calculate smoothed movement vector over recent trajectory points
                    k = min(len(history), 8)
                    dx = history[-1][0] - history[-k][0]
                    dy = history[-1][1] - history[-k][1]
                    mag = float(np.hypot(dx, dy))

                    is_near_edge = (
                        cx < edge_margin or cx > (w - edge_margin) or
                        cy < edge_margin or cy > (h - edge_margin)
                    )

                    if mag < 2.0:
                        movement_state = self.last_moving_direction.get(track_id, MovementStateEnum.UNKNOWN)
                    else:
                        u_vec = np.array([dx / mag, dy / mag], dtype=np.float32)
                        dot_prod = float(np.dot(u_vec, self.junction_vector))
                        established_dir = self.last_moving_direction.get(track_id)

                        if established_dir is not None:
                            # Hysteresis: prevent edge-clipping shifts or tiny lateral deviations from flipping direction
                            if is_near_edge:
                                movement_state = established_dir
                            elif established_dir == MovementStateEnum.INCOMING:
                                if dot_prod > -0.15:
                                    movement_state = MovementStateEnum.INCOMING
                                else:
                                    # Contradictory evidence: require meaningful displacement to flip
                                    if mag >= settings.DIRECTION_FLIP_MIN_DISPLACEMENT and len(history) >= 5:
                                        movement_state = MovementStateEnum.OUTGOING
                                        self.last_moving_direction[track_id] = MovementStateEnum.OUTGOING
                                    else:
                                        movement_state = MovementStateEnum.INCOMING
                            elif established_dir == MovementStateEnum.OUTGOING:
                                if dot_prod < 0.15:
                                    movement_state = MovementStateEnum.OUTGOING
                                else:
                                    if mag >= settings.DIRECTION_FLIP_MIN_DISPLACEMENT and len(history) >= 5:
                                        movement_state = MovementStateEnum.INCOMING
                                        self.last_moving_direction[track_id] = MovementStateEnum.INCOMING
                                    else:
                                        movement_state = MovementStateEnum.OUTGOING
                        else:
                            # First time establishing direction
                            if dot_prod > 0.15:
                                movement_state = MovementStateEnum.INCOMING
                                self.last_moving_direction[track_id] = MovementStateEnum.INCOMING
                            elif dot_prod < -0.15:
                                movement_state = MovementStateEnum.OUTGOING
                                self.last_moving_direction[track_id] = MovementStateEnum.OUTGOING
                            else:
                                movement_state = MovementStateEnum.UNKNOWN

            is_parked = (movement_state == MovementStateEnum.PARKED)
            stopped_duration_sec = self.stopped_frames_count.get(track_id, 0) / current_fps

            tracked_vehicles.append(
                TrackedVehicle(
                    track_id=track_id,
                    xyxy=xyxy,
                    confidence=float(conf),
                    class_id=stable_cls_id,
                    class_name=stable_cls_name,
                    center=curr_center,
                    previous_center=prev_center,
                    speed_px=speed_px,
                    stationary_frames=self.stationary_counts.get(track_id, 0),
                    crossed_counting_line=(track_id in self.crossed_ids),
                    raw_class_id=cls_id,
                    raw_class_name=self.class_names[cls_id],
                    direction=movement_state,
                    stopped_duration_seconds=stopped_duration_sec,
                    last_moving_direction=self.last_moving_direction.get(track_id),
                    is_parked=is_parked
                )
            )

        return tracked_vehicles
