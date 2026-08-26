from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ultralytics import YOLO
from backend.config import settings

@dataclass
class Detection:
    xyxy: list[float]      # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    center: tuple[float, float]

class VehicleDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.model = YOLO(self.model_path)
        self.target_classes = settings.TARGET_CLASSES
        self.class_names = settings.CLASS_NAMES
        self.conf_threshold = settings.CONFIDENCE_THRESHOLD
        self.iou_threshold = settings.IOU_THRESHOLD

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single frame and return filtered vehicle detections.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=settings.INFERENCE_IMAGE_SIZE,
            classes=self.target_classes,
            verbose=False
        )
        
        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        if r.boxes is None:
            return detections

        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self.class_names:
                continue
            
            conf = float(box.conf[0].item())
            xyxy = [float(coord) for coord in box.xyxy[0].tolist()]
            cx = (xyxy[0] + xyxy[2]) / 2.0
            cy = (xyxy[1] + xyxy[3]) / 2.0
            
            detections.append(
                Detection(
                    xyxy=xyxy,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self.class_names[cls_id],
                    center=(cx, cy)
                )
            )

        return detections
