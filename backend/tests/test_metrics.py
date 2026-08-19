import pytest
from backend.models.traffic_schemas import ApproachEnum, TrafficLevelEnum
from backend.core.vision.tracker import TrackedVehicle
from backend.core.analytics.traffic_metrics import TrafficMetricsCalculator, intersect

def test_line_intersection():
    # Crossing horizontal line at y=50 from (20, 40) to (20, 60)
    p1 = (0.0, 50.0)
    p2 = (100.0, 50.0)
    a = (20.0, 40.0)
    b = (20.0, 60.0)
    assert intersect(a, b, p1, p2) is True

    # Not crossing line
    c = (20.0, 10.0)
    d = (20.0, 30.0)
    assert intersect(c, d, p1, p2) is False

def test_traffic_metrics_calculation():
    calc = TrafficMetricsCalculator(ApproachEnum.NORTH)
    
    vehicles = [
        TrackedVehicle(
            track_id=1,
            xyxy=[100.0, 100.0, 200.0, 200.0],
            confidence=0.85,
            class_id=2,
            class_name="car",
            center=(150.0, 150.0),
            previous_center=(150.0, 140.0),
            speed_px=10.0,
            stationary_frames=0
        ),
        TrackedVehicle(
            track_id=2,
            xyxy=[250.0, 200.0, 320.0, 280.0],
            confidence=0.90,
            class_id=3,
            class_name="motorcycle",
            center=(285.0, 240.0),
            previous_center=(285.0, 240.0),
            speed_px=0.0,
            stationary_frames=8 # Stationary / Queued
        )
    ]

    state = calc.calculate_metrics(
        vehicles=vehicles,
        frame_width=640,
        frame_height=480,
        processed_frames=1
    )

    assert state.approach == ApproachEnum.NORTH
    assert state.vehicle_count == 2
    assert state.class_counts["car"] == 1
    assert state.class_counts["motorcycle"] == 1
    assert state.class_counts["bus"] == 0
    assert state.estimated_queue_length == 1  # Vehicle 2 is stationary
    assert state.traffic_level == TrafficLevelEnum.LOW
