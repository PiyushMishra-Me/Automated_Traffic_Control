from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from ultralytics import YOLO

from backend.config import settings


@dataclass
class Detection:
    xyxy: list[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    center: tuple[float, float]


class VehicleDetector:
    def __init__(self, model_path: Optional[str] = None):
        # Main YOLO model for normal vehicles
        self.model_path = model_path or settings.MODEL_PATH
        self.model = YOLO(self.model_path)

        # Custom trained YOLO model for ambulances
        self.ambulance_model = YOLO(settings.AMBULANCE_MODEL_PATH)

        self.target_classes = settings.TARGET_CLASSES
        self.class_names = settings.CLASS_NAMES
        self.conf_threshold = settings.CONFIDENCE_THRESHOLD
        self.iou_threshold = settings.IOU_THRESHOLD

        # If a normal vehicle overlaps an ambulance by this amount,
        # treat it as the same vehicle and remove the normal detection.
        self.ambulance_overlap_threshold = 0.40

    @staticmethod
    def calculate_iou(box1: list[float], box2: list[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        Box format: [x1, y1, x2, y2]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_width = max(0.0, x2 - x1)
        intersection_height = max(0.0, y2 - y1)
        intersection_area = intersection_width * intersection_height

        if intersection_area == 0:
            return 0.0

        box1_area = max(0.0, box1[2] - box1[0]) * max(
            0.0, box1[3] - box1[1]
        )
        box2_area = max(0.0, box2[2] - box2[0]) * max(
            0.0, box2[3] - box2[1]
        )

        union_area = box1_area + box2_area - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run both models and return:
        car, motorcycle, bus, truck, and ambulance.

        If an ambulance overlaps a normal vehicle detection,
        the normal vehicle detection is removed to prevent
        double counting.
        """

        vehicle_detections: List[Detection] = []
        ambulance_detections: List[Detection] = []

        # -----------------------------------------
        # 1. Detect normal vehicles
        # -----------------------------------------
        vehicle_results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=settings.INFERENCE_IMAGE_SIZE,
            classes=self.target_classes,
            verbose=False
        )

        if vehicle_results and vehicle_results[0].boxes is not None:
            for box in vehicle_results[0].boxes:
                cls_id = int(box.cls[0].item())

                if cls_id not in self.class_names:
                    continue

                conf = float(box.conf[0].item())
                xyxy = [float(coord) for coord in box.xyxy[0].tolist()]

                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0

                vehicle_detections.append(
                    Detection(
                        xyxy=xyxy,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=self.class_names[cls_id],
                        center=(cx, cy)
                    )
                )

        # -----------------------------------------
        # 2. Detect ambulances
        # -----------------------------------------
        ambulance_results = self.ambulance_model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=settings.INFERENCE_IMAGE_SIZE,
            verbose=False
        )

        if ambulance_results and ambulance_results[0].boxes is not None:
            for box in ambulance_results[0].boxes:
                conf = float(box.conf[0].item())
                xyxy = [float(coord) for coord in box.xyxy[0].tolist()]

                cx = (xyxy[0] + xyxy[2]) / 2.0
                cy = (xyxy[1] + xyxy[3]) / 2.0

                ambulance_detections.append(
                    Detection(
                        xyxy=xyxy,
                        confidence=conf,
                        class_id=settings.AMBULANCE_CLASS_ID,
                        class_name=settings.AMBULANCE_CLASS_NAME,
                        center=(cx, cy)
                    )
                )

        # -----------------------------------------
        # 3. Remove normal vehicles overlapping ambulances
        # -----------------------------------------
        filtered_vehicle_detections: List[Detection] = []

        for vehicle in vehicle_detections:
            overlaps_ambulance = False

            for ambulance in ambulance_detections:
                iou = self.calculate_iou(
                    vehicle.xyxy,
                    ambulance.xyxy
                )

                if iou >= self.ambulance_overlap_threshold:
                    overlaps_ambulance = True
                    break

            if not overlaps_ambulance:
                filtered_vehicle_detections.append(vehicle)

        # Combine remaining vehicles with ambulances
        return filtered_vehicle_detections + ambulance_detections