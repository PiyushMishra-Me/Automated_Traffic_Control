import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.models.navigation_schemas import NavigationRequest, VehicleProfileEnum
from backend.models.ambulance_schemas import AmbulanceMissionCreate, AmbulanceCriticalityEnum
from backend.db.repositories.ambulance_repo import ambulance_repo
from backend.db.repositories.incident_repo import incident_repo

client = TestClient(app)

def test_basic_navigation_shortest_path():
    req = {
        "origin_junction_id": "J-04",
        "destination_junction_id": "J-02",
        "vehicle_type": "CAR",
        "avoid_high_congestion": True
    }
    resp = client.post("/api/navigation/route", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["origin_junction_id"] == "J-04"
    assert data["destination_junction_id"] == "J-02"
    assert len(data["optimal_route_junctions"]) >= 2
    assert data["total_distance_km"] > 0
    assert data["estimated_travel_time_seconds"] > 0
    assert len(data["steps"]) >= 1

def test_navigation_corridors_status_endpoint():
    resp = client.get("/api/navigation/corridors")
    assert resp.status_code == 200
    corridors = resp.json()
    assert len(corridors) >= 10
    first = corridors[0]
    assert "road_name" in first
    assert "estimated_transit_seconds" in first
    assert "live_traffic_multiplier" in first

def test_navigation_emergency_priority_warning():
    # Register an active Critical Ambulance along J-04 -> J-01
    amb_payload = AmbulanceMissionCreate(
        hospital_name="Apollo Emergency Hub",
        ambulance_vehicle_id="DL-01-AMB-7777",
        criticality=AmbulanceCriticalityEnum.CRITICAL_LIFE_THREATENING,
        patient_condition="Cardiac critical ICU transfer",
        victim_location="Central Ring Road",
        origin_junction_id="J-04",
        destination_junction_id="J-01"
    )
    amb_res = ambulance_repo.register_mission(amb_payload)
    assert amb_res.mission_id is not None

    # Compute commuter route along J-04 -> J-01
    req = {
        "origin_junction_id": "J-04",
        "destination_junction_id": "J-01",
        "vehicle_type": "CAR",
        "avoid_high_congestion": True
    }
    resp = client.post("/api/navigation/route", json=req)
    assert resp.status_code == 200
    data = resp.json()

    # Verify that the emergency clearance warning is captured
    assert len(data["emergency_corridor_warnings"]) > 0
    step_with_emergency = [s for s in data["steps"] if s["emergency_active"]]
    assert len(step_with_emergency) > 0
    assert step_with_emergency[0]["emergency_priority"] == 4
