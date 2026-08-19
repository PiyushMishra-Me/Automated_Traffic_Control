from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralytics import YOLO
from backend.config import settings

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

class VehicleTracker:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model = YOLO(self.model_path)
        self.target_classes = settings.TARGET_CLASSES
        self.class_names = settings.CLASS_NAMES
        self.conf_threshold = settings.CONFIDENCE_THRESHOLD
        self.iou_threshold = settings.IOU_THRESHOLD
        # Historical memory for tracks
        self.track_history: Dict[int, List[Tuple[float, float]]] = {}
        self.stationary_counts: Dict[int, int] = {}
        self.crossed_ids: set[int] = set()

    def reset(self):
        """Reset internal tracker history between video sessions."""
        self.track_history.clear()
        self.stationary_counts.clear()
        self.crossed_ids.clear()

    def track(self, frame: np.ndarray) -> List[TrackedVehicle]:
        """
        Track vehicles across the given frame using ByteTrack.
        Returns persistent TrackedVehicle objects.
        """
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
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
            xyxy = [float(c) for c in box]
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            curr_center = (cx, cy)

            # Compute displacement / speed
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

            # Update stationary frame count
            if speed_px < settings.QUEUE_SPEED_THRESHOLD:
                self.stationary_counts[track_id] = self.stationary_counts.get(track_id, 0) + 1
            else:
                self.stationary_counts[track_id] = max(0, self.stationary_counts.get(track_id, 0) - 1)

            tracked_vehicles.append(
                TrackedVehicle(
                    track_id=track_id,
                    xyxy=xyxy,
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=self.class_names[cls_id],
                    center=curr_center,
                    previous_center=prev_center,
                    speed_px=speed_px,
                    stationary_frames=self.stationary_counts.get(track_id, 0),
                    crossed_counting_line=(track_id in self.crossed_ids)
                )
            )

        return tracked_vehicles
