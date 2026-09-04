import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.models.traffic_schemas import SignalPhaseEnum, ApproachEnum

client = TestClient(app)

def test_manual_signal_override_lifecycle():
    junction_id = "J-01"

    # 1. Check initial override status (should be inactive)
    res = client.get(f"/api/junctions/{junction_id}/signal-override")
    assert res.status_code == 200
    data = res.json()
    assert data["active"] is False

    # 2. Set EMERGENCY_ALL_RED override
    req_body = {
        "override_mode": "EMERGENCY_ALL_RED",
        "reason": "Vehicle fire blocking intersection",
        "duration_seconds": 120,
        "authorized_by": "traffic_command"
    }
    res = client.post(f"/api/junctions/{junction_id}/signal-override", json=req_body)
    assert res.status_code == 200
    override_data = res.json()
    assert override_data["active"] is True
    assert override_data["override_mode"] == "EMERGENCY_ALL_RED"
    assert override_data["phase"] == SignalPhaseEnum.ALL_RED.value
    assert len(override_data["forced_red_approaches"]) == 4

    # 3. Verify signal recommendation reflects the manual override
    rec_res = client.get(f"/api/junctions/{junction_id}/signal-recommendation")
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert rec_data["recommended_phase"] == SignalPhaseEnum.ALL_RED.value
    assert rec_data["manual_override_active"] is True
    assert "Vehicle fire blocking intersection" in rec_data["rationale"]

    # 4. Check active overrides list
    list_res = client.get("/api/junctions/active-signal-overrides")
    assert list_res.status_code == 200
    all_overrides = list_res.json()
    assert junction_id in all_overrides

    # 5. Clear override
    del_res = client.delete(f"/api/junctions/{junction_id}/signal-override")
    assert del_res.status_code == 200
    assert del_res.json()["cleared"] is True

    # 6. Verify cleared
    res_after = client.get(f"/api/junctions/{junction_id}/signal-override")
    assert res_after.json()["active"] is False

def test_manual_signal_override_directional_red():
    junction_id = "J-02"

    # Set directional HOLD_RED on NORTH approach
    req_body = {
        "override_mode": "HOLD_RED_APPROACH",
        "forced_red_approaches": ["NORTH"],
        "reason": "Collision on North bridge ramp",
        "duration_seconds": 90,
        "authorized_by": "police_ops"
    }
    res = client.post(f"/api/junctions/{junction_id}/signal-override", json=req_body)
    assert res.status_code == 200
    data = res.json()
    assert data["active"] is True
    assert data["forced_red_approaches"] == ["NORTH"]

    # Clean up
    client.delete(f"/api/junctions/{junction_id}/signal-override")
