import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["phase"] == 1
    assert data["status"] == "online"

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
