import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.models.incident_schemas import IncidentTypeEnum, IncidentSeverityEnum, IncidentStatusEnum

client = TestClient(app)

def test_weather_endpoint_and_override():
    # 1. Fetch weather for default junction J-01
    res = client.get("/api/weather/junction/J-01")
    assert res.status_code == 200
    data = res.json()
    assert data["junction_id"] == "J-01"
    assert "temperature_c" in data
    assert "road_surface" in data
    assert "adjustments" in data

    # 2. Simulate Heavy Rain / Flooded weather override
    override_payload = {
        "condition": "HEAVY_RAIN",
        "precipitation_mm": 15.0,
        "visibility_km": 1.8,
        "road_surface": "WET"
    }
    res_override = client.post("/api/weather/junction/J-01/override", json=override_payload)
    assert res_override.status_code == 200
    ov_data = res_override.json()
    assert ov_data["condition"] == "HEAVY_RAIN"
    assert ov_data["adjustments"]["extra_yellow_seconds"] >= 1.5
    assert ov_data["adjustments"]["extra_all_red_seconds"] >= 2.0
    assert ov_data["adjustments"]["speed_advisory_kmh"] <= 35

    # 3. Clear override
    res_clear = client.delete("/api/weather/junction/J-01/override")
    assert res_clear.status_code == 200

def test_incident_reporting_and_diversion_lifecycle():
    # 1. Report an accident on J-01 NORTH
    incident_payload = {
        "junction_id": "J-01",
        "approach": "NORTH",
        "road_name": "North Boulevard",
        "incident_type": "ACCIDENT",
        "severity": "CRITICAL_ROAD_BLOCKED",
        "description": "Multi-vehicle collision blocking two lanes.",
        "estimated_clearance_minutes": 45,
        "reported_by": "Highway Patrol Unit 4"
    }
    res_create = client.post("/api/incidents", json=incident_payload)
    assert res_create.status_code == 200
    inc_data = res_create.json()
    inc_id = inc_data["incident_id"]
    assert inc_data["status"] == "ACTIVE"
    assert inc_data["diversion_plan"] is not None
    assert inc_data["diversion_plan"]["active"] is True
    assert len(inc_data["diversion_plan"]["steps"]) >= 2

    # 2. Query active diversions for J-01
    res_div = client.get("/api/incidents/junction/J-01/active-diversions")
    assert res_div.status_code == 200
    div_list = res_div.json()
    assert len(div_list) >= 1
    assert any(d["affected_approach"] == "NORTH" for d in div_list)

    # 3. Signal recommendation should now reflect incident alert and penalty
    res_sig = client.get("/api/junctions/J-01/signal-recommendation")
    assert res_sig.status_code == 200
    sig_data = res_sig.json()
    assert any("ACCIDENT" in a["message"] for a in sig_data["alerts"])

    # 4. Resolve incident
    res_resolve = client.patch(f"/api/incidents/{inc_id}/status", json={"status": "RESOLVED"})
    assert res_resolve.status_code == 200
    assert res_resolve.json()["status"] == "RESOLVED"

def test_traffic_police_base_dispatch_reporting_without_camera():
    """Traffic police at base receiving a call can report an accident towards a junction without live camera photo."""
    police_payload = {
        "junction_id": "J-02",
        "approach": "SOUTH",
        "road_name": "South Expressway",
        "incident_type": "ACCIDENT",
        "severity": "SEVERE",
        "description": "Base received PCR emergency call reporting multiple car pileup.",
        "estimated_clearance_minutes": 35,
        "reported_by": "Traffic Police Control Base Room",
        "reporter_role": "TRAFFIC_POLICE",
        "dispatch_call_ref": "POL-CALL-8842",
        "photo_base64": None,
        "is_live_captured": False
    }
    res = client.post("/api/incidents", json=police_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["reporter_role"] == "TRAFFIC_POLICE"
    assert data["dispatch_call_ref"] == "POL-CALL-8842"
    assert data["photo_base64"] is None
    assert data["is_live_captured"] is False
    assert data["status"] == "ACTIVE"
    assert data["diversion_plan"] is not None
    assert data["diversion_plan"]["active"] is True

def test_public_citizen_reporting_with_live_camera():
    """Public citizen on-scene reporting requires live captured photo evidence."""
    public_payload = {
        "junction_id": "J-03",
        "approach": "EAST",
        "road_name": "East Ring Road",
        "incident_type": "ROAD_HAZARD",
        "severity": "MODERATE",
        "description": "Fallen tree blocking eastbound lane, snapped live on camera.",
        "estimated_clearance_minutes": 20,
        "reported_by": "Citizen Commuter",
        "reporter_role": "PUBLIC_CITIZEN",
        "photo_base64": "data:image/jpeg;base64,samplephotoevidence123",
        "is_live_captured": True
    }
    res = client.post("/api/incidents", json=public_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["reporter_role"] == "PUBLIC_CITIZEN"
    assert data["photo_base64"] == "data:image/jpeg;base64,samplephotoevidence123"
    assert data["is_live_captured"] is True
    assert data["diversion_plan"] is not None
