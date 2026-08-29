from datetime import datetime, timezone
from typing import List, Optional
from backend.db.mongo_client import db_manager
from backend.models.traffic_schemas import JunctionCreate, JunctionInfo

_memory_junctions: dict[str, dict] = {
    "J-01": {
        "junction_id": "J-01",
        "name": "Central Plaza Intersection",
        "location": "Connaught Outer Circle & Barakhamba",
        "latitude": 28.6315,
        "longitude": 77.2167,
        "road_names": {
            "NORTH": "North Boulevard",
            "SOUTH": "South Radial Expressway",
            "EAST": "East Arterial Corridor B",
            "WEST": "West Commercial Linkway"
        },
        "connected_junctions": ["J-02", "J-03", "J-04", "J-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-02": {
        "junction_id": "J-02",
        "name": "Tech City Interchange",
        "location": "East Ring Arterial & IT Hubway",
        "latitude": 28.6385,
        "longitude": 77.2310,
        "road_names": {
            "NORTH": "Tech Park Loop",
            "SOUTH": "Subway Bypass",
            "EAST": "Industrial Outer Way",
            "WEST": "Central Connection Road"
        },
        "connected_junctions": ["J-01", "J-03", "J-04"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-03": {
        "junction_id": "J-03",
        "name": "North Ring Crossing",
        "location": "North Ring Road & University Ave",
        "latitude": 28.6460,
        "longitude": 77.2110,
        "road_names": {
            "NORTH": "Grand Trunk Extension",
            "SOUTH": "North Boulevard",
            "EAST": "Campus Flyover",
            "WEST": "Civil Lines Passage"
        },
        "connected_junctions": ["J-01", "J-02", "J-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-04": {
        "junction_id": "J-04",
        "name": "Metro Transit Interchange",
        "location": "South Radial & Metro Line 3",
        "latitude": 28.6180,
        "longitude": 77.2190,
        "road_names": {
            "NORTH": "South Radial Expressway",
            "SOUTH": "Ring Road South",
            "EAST": "Station Terminal Road",
            "WEST": "Hospital Access Way"
        },
        "connected_junctions": ["J-01", "J-02", "J-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-05": {
        "junction_id": "J-05",
        "name": "West Commercial Gateway",
        "location": "West Linkway & Financial District",
        "latitude": 28.6250,
        "longitude": 77.1990,
        "road_names": {
            "NORTH": "Diplomatic Enclave Lane",
            "SOUTH": "Market Central Access",
            "EAST": "West Commercial Linkway",
            "WEST": "Airport Express Link"
        },
        "connected_junctions": ["J-01", "J-03", "J-04"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    }
}

class JunctionRepository:
    def __init__(self):
        pass

    @property
    def collection(self):
        db = db_manager.get_database()
        if db is not None:
            return db["junctions"]
        return None

    def create_junction(self, junction: JunctionCreate) -> dict:
        doc = {
            "junction_id": junction.junction_id,
            "name": junction.name,
            "location": junction.location or "",
            "latitude": junction.latitude,
            "longitude": junction.longitude,
            "road_names": junction.road_names or {},
            "connected_junctions": junction.connected_junctions or [],
            "created_at": datetime.now(timezone.utc),
            "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"],
            "custom_counting_lines": junction.custom_counting_lines.dict() if junction.custom_counting_lines else {}
        }
        col = self.collection
        if col is not None:
            try:
                col.update_one({"junction_id": junction.junction_id}, {"$set": doc}, upsert=True)
            except Exception:
                pass
        _memory_junctions[junction.junction_id] = doc
        return doc

    def get_junction(self, junction_id: str) -> Optional[dict]:
        col = self.collection
        if col is not None:
            try:
                res = col.find_one({"junction_id": junction_id})
                if res:
                    res["_id"] = str(res["_id"])
                    return res
            except Exception:
                pass
        return _memory_junctions.get(junction_id)

    def list_junctions(self) -> List[dict]:
        col = self.collection
        if col is not None:
            try:
                items = list(col.find({}))
                if items:
                    for item in items:
                        item["_id"] = str(item["_id"])
                    return items
            except Exception:
                pass
        return list(_memory_junctions.values())

    def update_counting_lines(self, junction_id: str, counting_lines: dict) -> Optional[dict]:
        """Persist calibrated normalized counting lines without overwriting junction details."""
        existing = self.get_junction(junction_id)
        if not existing:
            return None

        serializable_lines = {
            str(approach.value if hasattr(approach, "value") else approach): (
                config.model_dump() if hasattr(config, "model_dump") else config
            )
            for approach, config in counting_lines.items()
        }
        col = self.collection
        if col is not None:
            try:
                col.update_one(
                    {"junction_id": junction_id},
                    {"$set": {"custom_counting_lines": serializable_lines}},
                )
            except Exception:
                pass

        memory_doc = _memory_junctions.get(junction_id, existing)
        memory_doc["custom_counting_lines"] = serializable_lines
        _memory_junctions[junction_id] = memory_doc
        return memory_doc

junction_repo = JunctionRepository()
