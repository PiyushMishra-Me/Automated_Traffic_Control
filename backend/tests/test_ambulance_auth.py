import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.models.ambulance_schemas import (
    AmbulanceCriticalityEnum,
    AmbulanceStatusEnum,
    AmbulanceMissionCreate
)
from backend.models.auth_schemas import UserRoleEnum
from backend.core.control.ambulance_engine import ambulance_engine
from backend.db.repositories.ambulance_repo import ambulance_repo

client = TestClient(app)

def test_auth_profiles_and_login():
    # 1. Test profiles list
    res = client.get("/api/auth/profiles")
    assert res.status_code == 200
    profiles = res.json()
    assert len(profiles) == 3
    roles = [p["role"] for p in profiles]
    assert "PUBLIC_USER" in roles
    assert "HOSPITAL_DISPATCH" in roles
    assert "GOVERNMENT_OFFICIAL" in roles

    # 2. Test Public user login
    res_pub = client.post("/api/auth/login", json={
        "role": "PUBLIC_USER",
        "username": "Rahul Citizen",
        "password": ""
    })
    assert res_pub.status_code == 200
    assert res_pub.json()["role"] == "PUBLIC_USER"

    # 3. Test Hospital auth failure with bad password
    res_fail = client.post("/api/auth/login", json={
        "role": "HOSPITAL_DISPATCH",
        "username": "hospital_admin",
        "password": "wrongpassword999"
    })
    assert res_fail.status_code == 401

    # 4. Test Hospital login success
    res_hosp = client.post("/api/auth/login", json={
        "role": "HOSPITAL_DISPATCH",
        "username": "hospital_admin",
        "password": "hospital123",
        "organization_name": "Apollo Trauma Network"
    })
    assert res_hosp.status_code == 200
    assert res_hosp.json()["organization_name"] == "Apollo Trauma Network"

def test_ambulance_registration_and_green_corridor():
    payload = {
        "hospital_name": "AIIMS Apex Trauma Center",
        "ambulance_vehicle_id": "DL-01-EM-9999",
        "driver_contact": "+91 99887 76655",
        "criticality": "CRITICAL_LIFE_THREATENING",
        "patient_condition": "Severe Cardiac Arrest - Level 1 Priority",
        "victim_location": "Ring Road J-01 Crossing",
        "origin_junction_id": "J-04",
        "destination_junction_id": "J-02"
    }

    res = client.post("/api/ambulances/register", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["mission_id"].startswith("AMB-")
    assert data["priority_level"] == 4
    assert data["criticality"] == "CRITICAL_LIFE_THREATENING"
    assert len(data["route_corridor"]) >= 2

    mission_id = data["mission_id"]

    # Test status progression
    res_step1 = client.patch(f"/api/ambulances/{mission_id}/status", params={"new_status": "ON_SCENE_PICKUP"})
    assert res_step1.status_code == 200
    assert res_step1.json()["status"] == "ON_SCENE_PICKUP"

    res_step2 = client.patch(f"/api/ambulances/{mission_id}/status", params={"new_status": "TRANSIT_TO_HOSPITAL"})
    assert res_step2.status_code == 200
    assert res_step2.json()["status"] == "TRANSIT_TO_HOSPITAL"

    # Test junction preemption on route junction
    target_junction = data["route_corridor"][0]["junction_id"]
    res_preempt = client.get(f"/api/ambulances/junction/{target_junction}/preemption")
    assert res_preempt.status_code == 200
    preempt_data = res_preempt.json()
    assert preempt_data["is_preempted"] is True

def test_multi_ambulance_priority_conflict_resolution():
    # Register Mission 1: CRITICAL (Priority 4)
    m1 = ambulance_repo.register_mission(AmbulanceMissionCreate(
        hospital_name="Fortis ICU",
        ambulance_vehicle_id="DL-02-ICU-11",
        criticality=AmbulanceCriticalityEnum.CRITICAL_LIFE_THREATENING,
        patient_condition="Massive myocardial infarction",
        victim_location="Sector 4",
        origin_junction_id="J-04",
        destination_junction_id="J-01"
    ))

    # Register Mission 2: MEDIUM (Priority 2)
    m2 = ambulance_repo.register_mission(AmbulanceMissionCreate(
        hospital_name="Max Healthcare",
        ambulance_vehicle_id="DL-03-MED-22",
        criticality=AmbulanceCriticalityEnum.MEDIUM,
        patient_condition="Stable fracture transport",
        victim_location="Sector 2",
        origin_junction_id="J-02",
        destination_junction_id="J-01"
    ))

    # Test conflict resolver at junction J-01
    active = [m1, m2]
    conflict = ambulance_engine.resolve_conflicts(active, "J-01")

    assert conflict.has_conflict is True
    assert conflict.winning_mission_id == m1.mission_id
    assert conflict.winning_criticality == AmbulanceCriticalityEnum.CRITICAL_LIFE_THREATENING
    assert conflict.secondary_mission_id == m2.mission_id
    assert "PRIORITY OVERRIDE" in conflict.strategy
