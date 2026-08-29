import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict
from backend.db.mongo_client import db_manager
from backend.models.ambulance_schemas import (
    AmbulanceMissionCreate,
    AmbulanceMissionResponse,
    AmbulanceStatusEnum,
    AmbulanceCriticalityEnum,
    CRITICALITY_PRIORITY_MAP
)
from backend.core.control.ambulance_engine import ambulance_engine, CORRIDOR_MAP
from backend.db.repositories.junction_repo import junction_repo

_memory_missions: Dict[str, dict] = {}

class AmbulanceRepository:
    @property
    def collection(self):
        db = db_manager.get_database()
        if db is not None:
            return db["ambulance_missions"]
        return None

    def register_mission(self, payload: AmbulanceMissionCreate) -> AmbulanceMissionResponse:
        mission_id = f"AMB-{uuid.uuid4().hex[:6].upper()}"
        now = datetime.now(timezone.utc)

        route_nodes = ambulance_engine.plan_emergency_route(
            payload.origin_junction_id,
            payload.destination_junction_id
        )

        priority_lvl = CRITICALITY_PRIORITY_MAP.get(payload.criticality, 3)

        # Initial coordinates based on origin junction
        j = junction_repo.get_junction(payload.origin_junction_id)
        curr_lat = float(j.get("latitude", 28.6139)) if j else 28.6139
        curr_lng = float(j.get("longitude", 77.2090)) if j else 77.2090

        total_eta = sum(node.eta_seconds for node in route_nodes) if route_nodes else 180

        doc = {
            "mission_id": mission_id,
            "agency_type": payload.agency_type.value if hasattr(payload.agency_type, 'value') else payload.agency_type,
            "hospital_name": payload.hospital_name,
            "ambulance_vehicle_id": payload.ambulance_vehicle_id,
            "driver_contact": payload.driver_contact or "+91 98765 43210",
            "criticality": payload.criticality.value,
            "priority_level": priority_lvl,
            "patient_condition": payload.patient_condition,
            "victim_location": payload.victim_location,
            "origin_junction_id": payload.origin_junction_id,
            "destination_junction_id": payload.destination_junction_id,
            "route_corridor": [n.model_dump() for n in route_nodes],
            "active_node_index": 0,
            "status": AmbulanceStatusEnum.DISPATCHED_TO_VICTIM.value,
            "current_lat": curr_lat,
            "current_lng": curr_lng,
            "estimated_total_eta_seconds": total_eta,
            "conflict_resolution": None,
            "dispatched_at": now.isoformat(),
            "updated_at": now.isoformat()
        }

        col = self.collection
        if col is not None:
            try:
                col.insert_one(dict(doc))
            except Exception:
                pass

        _memory_missions[mission_id] = doc
        return self._format_response(doc)

    def list_missions(
        self,
        status: Optional[str] = None,
        hospital_name: Optional[str] = None
    ) -> List[AmbulanceMissionResponse]:
        col = self.collection
        results = []
        if col is not None:
            try:
                q = {}
                if status:
                    q["status"] = status
                if hospital_name:
                    q["hospital_name"] = hospital_name
                for d in col.find(q).sort("dispatched_at", -1):
                    d["_id"] = str(d["_id"])
                    results.append(self._format_response(d))
                return results
            except Exception:
                pass

        # In-memory fallback
        for d in sorted(_memory_missions.values(), key=lambda x: x["dispatched_at"], reverse=True):
            if status and d["status"] != status:
                continue
            if hospital_name and d["hospital_name"] != hospital_name:
                continue
            results.append(self._format_response(d))
        return results

    def get_mission(self, mission_id: str) -> Optional[AmbulanceMissionResponse]:
        col = self.collection
        if col is not None:
            try:
                d = col.find_one({"mission_id": mission_id})
                if d:
                    d["_id"] = str(d["_id"])
                    return self._format_response(d)
            except Exception:
                pass
        d = _memory_missions.get(mission_id)
        return self._format_response(d) if d else None

    def update_status(self, mission_id: str, new_status: AmbulanceStatusEnum) -> Optional[AmbulanceMissionResponse]:
        now = datetime.now(timezone.utc)
        col = self.collection
        if col is not None:
            try:
                col.update_one(
                    {"mission_id": mission_id},
                    {"$set": {"status": new_status.value, "updated_at": now.isoformat()}}
                )
            except Exception:
                pass

        if mission_id in _memory_missions:
            _memory_missions[mission_id]["status"] = new_status.value
            _memory_missions[mission_id]["updated_at"] = now.isoformat()

            # Advance coordinates along route on status change
            doc = _memory_missions[mission_id]
            if new_status == AmbulanceStatusEnum.ON_SCENE_PICKUP:
                doc["active_node_index"] = min(1, len(doc["route_corridor"]) - 1)
            elif new_status == AmbulanceStatusEnum.TRANSIT_TO_HOSPITAL:
                doc["active_node_index"] = min(2, len(doc["route_corridor"]) - 1)

            return self._format_response(doc)
        return None

    def _format_response(self, doc: dict, resolve_conflict: bool = True) -> AmbulanceMissionResponse:
        data = dict(doc)
        if resolve_conflict:
            active_list = [
                self._format_response(m, resolve_conflict=False) 
                for m in _memory_missions.values() 
                if m["status"] in ["DISPATCHED_TO_VICTIM", "TRANSIT_TO_HOSPITAL"]
            ]
            target_j = doc["route_corridor"][doc["active_node_index"]]["junction_id"] if doc.get("route_corridor") else "J-01"
            conflict = ambulance_engine.resolve_conflicts(active_list, target_j) if len(active_list) > 1 else None
            data["conflict_resolution"] = conflict
        return AmbulanceMissionResponse(**data)

ambulance_repo = AmbulanceRepository()
