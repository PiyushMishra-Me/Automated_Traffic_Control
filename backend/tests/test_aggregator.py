import pytest
from backend.models.traffic_schemas import ApproachEnum, ApproachTrafficState, TrafficLevelEnum
from backend.core.analytics.junction_aggregator import JunctionAggregator

def test_junction_aggregation():
    north = ApproachTrafficState(
        approach=ApproachEnum.NORTH,
        vehicle_count=5,
        class_counts={"car": 4, "motorcycle": 1, "bus": 0, "truck": 0},
        density=0.3,
        estimated_queue_length=2,
        flow=10.0,
        traffic_level=TrafficLevelEnum.MEDIUM
    )
    south = ApproachTrafficState(
        approach=ApproachEnum.SOUTH,
        vehicle_count=12,
        class_counts={"car": 8, "motorcycle": 2, "bus": 1, "truck": 1},
        density=0.7,
        estimated_queue_length=6,
        flow=20.0,
        traffic_level=TrafficLevelEnum.HIGH
    )
    
    states = {
        "NORTH": north,
        "SOUTH": south
    }

    junction_state = JunctionAggregator.aggregate("J-TEST-01", states)

    assert junction_state.junction_id == "J-TEST-01"
    assert junction_state.total_active_vehicles == 17
    assert junction_state.north.vehicle_count == 5
    assert junction_state.south.vehicle_count == 12
    assert junction_state.east is None
    assert junction_state.west is None
    assert junction_state.aggregate_level == TrafficLevelEnum.HIGH
