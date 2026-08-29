import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.db.repositories.traffic_repo import traffic_repo
from backend.models.traffic_schemas import ApproachEnum, ApproachTrafficState, TrafficLevelEnum

client = TestClient(app)

def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "system" in data

def test_list_junctions():
    res = client.get("/api/junctions")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 1

def test_create_junction():
    payload = {
        "junction_id": "J-TEST-API",
        "name": "Test Junction Crossing",
        "location": "North Ring"
    }
    res = client.post("/api/junctions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["junction_id"] == "J-TEST-API"

def test_get_junction_state():
    res = client.get("/api/junctions/J-01/state")
    assert res.status_code == 200
    data = res.json()
    assert data["junction_id"] == "J-01"
    assert "north" in data
    assert "south" in data
    assert "east" in data
    assert "west" in data

def test_counting_line_calibration_and_analytics_history():
    calibration = {"custom_counting_lines": {"NORTH": {"p1": [0.1, 0.5], "p2": [0.9, 0.5], "orientation": "horizontal"}}}
    res = client.put("/api/junctions/J-01/counting-lines", json=calibration)
    assert res.status_code == 200
    assert res.json()["custom_counting_lines"]["NORTH"]["p1"] == [0.1, 0.5]

    traffic_repo.save_observation("J-01", ApproachTrafficState(approach=ApproachEnum.NORTH, vehicle_count=6, density=0.3, estimated_queue_length=2, flow=9, traffic_level=TrafficLevelEnum.MEDIUM))
    history = client.get("/api/analytics/junction/J-01/history?approach=NORTH")
    assert history.status_code == 200
    assert history.json()[0]["vehicle_count"] == 6
    summary = client.get("/api/analytics/junction/J-01/summary?approach=NORTH")
    assert summary.status_code == 200
    assert summary.json()["observations"] >= 1
    assert summary.json()["peak_vehicle_count"] >= 6

def test_signal_recommendation_simulation():
    response = client.get("/api/junctions/J-01/signal-recommendation")
    assert response.status_code == 200
    recommendation = response.json()
    assert recommendation["is_simulation"] is True
    assert recommendation["recommended_phase"] in {"NORTH_SOUTH_GREEN", "EAST_WEST_GREEN", "ALL_RED"}
    simulated = client.post("/api/junctions/J-01/signal-simulation", json={"current_phase": "ALL_RED"})
    assert simulated.status_code == 200
    assert simulated.json()["junction_id"] == "J-01"

def test_batch_video_status_endpoint():
    res = client.get("/api/videos/batch-status?junction_id=J-01")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

def test_live_stream_registration():
    payload = {
        "junction_id": "J-01",
        "approach": "NORTH",
        "stream_type": "RTSP",
        "stream_url": "rtsp://192.168.1.101:554/live",
        "is_active": True,
        "sampling_fps": 5.0
    }
    res = client.post("/api/videos/live-stream", json=payload)
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    streams = client.get("/api/videos/live-stream/J-01")
    assert streams.status_code == 200
    assert "NORTH" in streams.json()
    assert streams.json()["NORTH"]["stream_type"] == "RTSP"

    del_res = client.delete("/api/videos/live-stream/J-01/NORTH")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "success"

