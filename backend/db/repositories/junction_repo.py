from datetime import datetime, timezone
from typing import List, Optional
from backend.db.mongo_client import db_manager
from backend.models.traffic_schemas import JunctionCreate, JunctionInfo

_memory_junctions: dict[str, dict] = {
    "J-01": {
        "junction_id": "J-01",
        "name": "Central Plaza Intersection",
        "location": "Main St & 4th Avenue",
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

junction_repo = JunctionRepository()
