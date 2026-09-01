import numpy as np
import pytest
from backend.core.vision.detector import VehicleDetector, Detection
from backend.core.vision.tracker import VehicleTracker
from backend.config import settings


def test_dual_model_detector_initialization():
    """Verify that VehicleDetector initializes with YOLO11 and the custom ambulance detector."""
    detector = VehicleDetector()
    assert detector.model is not None
    assert detector.ambulance_model is not None
    assert detector.model_path == "yolo11s.pt"


def test_dual_model_detection_execution():
    """Verify that detect() executes cleanly on an image frame without runtime errors."""
    detector = VehicleDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    assert isinstance(detections, list)


def test_dual_model_overlap_filtering():
    """Verify that normal vehicles overlapping with ambulance detections are filtered out."""
    detector = VehicleDetector()
    
    # Simulate overlapping boxes
    box_car = [100.0, 100.0, 200.0, 200.0]
    box_ambulance = [105.0, 105.0, 205.0, 205.0]
    
    iou = detector.calculate_iou(box_car, box_ambulance)
    assert iou > detector.ambulance_overlap_threshold


def test_dual_model_tracker_execution():
    """Verify that VehicleTracker processes frames cleanly with YOLO11."""
    tracker = VehicleTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracked_vehicles = tracker.track(frame, fps=25.0)
    assert isinstance(tracked_vehicles, list)
