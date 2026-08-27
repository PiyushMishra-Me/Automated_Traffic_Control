import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict
from backend.db.mongo_client import db_manager
from backend.models.incident_schemas import (
    IncidentCreate,
    IncidentResponse,
    IncidentStatusEnum,
    IncidentSeverityEnum,
)
from backend.core.control.diversion_engine import diversion_engine
from backend.db.repositories.junction_repo import junction_repo

# In-memory store
_memory_incidents: Dict[str, dict] = {}

class IncidentRepository:
    @property
    def collection(self):
        db = db_manager.get_database()
        if db is not None:
            return db["incidents"]
        return None

    def create_incident(self, payload: IncidentCreate) -> IncidentResponse:
        inc_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(timezone.utc)
        
        # Determine coordinates if not supplied
        lat = payload.lat
        lng = payload.lng
        if lat is None or lng is None:
            j = junction_repo.get_junction(payload.junction_id)
            lat = float(j.get("latitude", 28.6139)) if j else 28.6139
            lng = float(j.get("longitude", 77.2090)) if j else 77.2090
            # Offset slightly by approach for map precision
            offsets = {
                "NORTH": (0.003, 0.0),
                "SOUTH": (-0.003, 0.0),
                "EAST": (0.0, 0.003),
                "WEST": (0.0, -0.003),
            }
            d_lat, d_lng = offsets.get(payload.approach.value, (0.0, 0.0))
            lat += d_lat
            lng += d_lng

        diversion = diversion_engine.calculate_diversion(
            junction_id=payload.junction_id,
            approach=payload.approach,
            severity=payload.severity
        )

        doc = {
            "incident_id": inc_id,
            "junction_id": payload.junction_id,
            "approach": payload.approach.value,
            "road_name": payload.road_name or "Main Arterial Avenue",
            "incident_type": payload.incident_type.value,
            "severity": payload.severity.value,
            "status": IncidentStatusEnum.ACTIVE.value,
            "description": payload.description,
            "estimated_clearance_minutes": payload.estimated_clearance_minutes,
            "reported_by": payload.reported_by or "Traffic Operations Center",
            "lat": lat,
            "lng": lng,
            "photo_base64": payload.photo_base64,
            "is_live_captured": payload.is_live_captured,
            "capture_timestamp": payload.capture_timestamp or now.isoformat(),
            "diversion_plan": diversion.model_dump(),
            "reported_at": now.isoformat(),
            "updated_at": now.isoformat()
        }

        col = self.collection
        if col is not None:
            try:
                col.insert_one(dict(doc))
            except Exception:
                pass

        _memory_incidents[inc_id] = doc
        return IncidentResponse(**doc)

    def get_incident(self, incident_id: str) -> Optional[IncidentResponse]:
        col = self.collection
        if col is not None:
            try:
                res = col.find_one({"incident_id": incident_id})
                if res:
                    res["_id"] = str(res["_id"])
                    return IncidentResponse(**res)
            except Exception:
                pass
        raw = _memory_incidents.get(incident_id)
        return IncidentResponse(**raw) if raw else None

    def list_incidents(self, junction_id: Optional[str] = None, status: Optional[str] = None) -> List[IncidentResponse]:
        col = self.collection
        results = []
        if col is not None:
            try:
                query = {}
                if junction_id:
                    query["junction_id"] = junction_id
                if status:
                    query["status"] = status
                items = list(col.find(query).sort("reported_at", -1))
                if items:
                    for item in items:
                        item["_id"] = str(item["_id"])
                        results.append(IncidentResponse(**item))
                    return results
            except Exception:
                pass

        for doc in sorted(_memory_incidents.values(), key=lambda x: x["reported_at"], reverse=True):
            if junction_id and doc["junction_id"] != junction_id:
                continue
            if status and doc["status"] != status:
                continue
            results.append(IncidentResponse(**doc))
        return results

    def get_active_for_junction(self, junction_id: str) -> List[IncidentResponse]:
        return self.list_incidents(junction_id=junction_id, status=IncidentStatusEnum.ACTIVE.value)

    def update_status(self, incident_id: str, new_status: IncidentStatusEnum) -> Optional[IncidentResponse]:
        now = datetime.now(timezone.utc)
        col = self.collection
        if col is not None:
            try:
                col.update_one(
                    {"incident_id": incident_id},
                    {"$set": {"status": new_status.value, "updated_at": now.isoformat()}}
                )
            except Exception:
                pass

        if incident_id in _memory_incidents:
            _memory_incidents[incident_id]["status"] = new_status.value
            _memory_incidents[incident_id]["updated_at"] = now.isoformat()
            if new_status == IncidentStatusEnum.RESOLVED and _memory_incidents[incident_id].get("diversion_plan"):
                _memory_incidents[incident_id]["diversion_plan"]["active"] = False
            return IncidentResponse(**_memory_incidents[incident_id])
        return None

incident_repo = IncidentRepository()
