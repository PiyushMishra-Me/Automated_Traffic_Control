from datetime import datetime, timezone
from typing import Dict, List, Optional
from backend.db.mongo_client import db_manager
from backend.models.traffic_schemas import ApproachEnum, ApproachTrafficState

# In-memory storage cache as fallback/buffer
_memory_observations: List[dict] = []

class TrafficRepository:
    def __init__(self):
        pass

    @property
    def collection(self):
        db = db_manager.get_database()
        if db is not None:
            return db["traffic_observations"]
        return None

    def save_observation(self, junction_id: str, state: ApproachTrafficState) -> dict:
        doc = {
            "junction_id": junction_id,
            "approach": state.approach.value if isinstance(state.approach, ApproachEnum) else str(state.approach),
            "timestamp": state.timestamp or datetime.now(timezone.utc),
            "vehicle_count": state.vehicle_count,
            "class_counts": state.class_counts,
            "density": state.density,
            "estimated_queue_length": state.estimated_queue_length,
            "flow": state.flow,
            "traffic_level": state.traffic_level.value if hasattr(state.traffic_level, "value") else str(state.traffic_level),
            "processed_frames": state.processed_frames,
            "total_unique_vehicles": state.total_unique_vehicles,
            "annotated_video_url": state.annotated_video_url
        }

        col = self.collection
        if col is not None:
            try:
                res = col.insert_one(doc)
                doc["_id"] = str(res.inserted_id)
            except Exception as e:
                _memory_observations.append(doc)
        else:
            _memory_observations.append(doc)

        return doc

    def get_latest_approach_observation(self, junction_id: str, approach: str) -> Optional[dict]:
        col = self.collection
        if col is not None:
            try:
                res = col.find_one(
                    {"junction_id": junction_id, "approach": approach.upper()},
                    sort=[("timestamp", -1)]
                )
                if res:
                    res["_id"] = str(res["_id"])
                    return res
            except Exception:
                pass

        # Fallback to memory
        filtered = [
            o for o in _memory_observations
            if o.get("junction_id") == junction_id and o.get("approach") == approach.upper()
        ]
        return filtered[-1] if filtered else None

    def get_all_latest_for_junction(self, junction_id: str) -> Dict[str, dict]:
        states = {}
        for app in ["NORTH", "SOUTH", "EAST", "WEST"]:
            latest = self.get_latest_approach_observation(junction_id, app)
            if latest:
                states[app] = latest
        return states

traffic_repo = TrafficRepository()
