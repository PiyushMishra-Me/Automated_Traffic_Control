from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from backend.models.traffic_schemas import (
    ManualSignalOverrideRequest, 
    ManualSignalOverrideResponse, 
    SignalPhaseEnum, 
    ApproachEnum
)

class ManualOverrideManager:
    """In-memory manager tracking active manual signal light overrides across junctions."""
    
    def __init__(self):
        self._overrides: Dict[str, dict] = {}

    def set_override(self, junction_id: str, req: ManualSignalOverrideRequest) -> ManualSignalOverrideResponse:
        if req.override_mode == "RESTORE_ADAPTIVE":
            self.clear_override(junction_id)
            return ManualSignalOverrideResponse(
                junction_id=junction_id,
                active=False,
                override_mode="RESTORE_ADAPTIVE",
                phase=None,
                forced_red_approaches=[],
                incident_id=req.incident_id,
                reason="Restored to automated AI adaptive signal control",
                expires_at=None,
                authorized_by=req.authorized_by,
                created_at=datetime.now(timezone.utc)
            )

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=req.duration_seconds) if req.duration_seconds > 0 else None

        # Resolve phase & forced approaches based on mode
        phase = req.phase
        forced_apps = list(req.forced_red_approaches)

        if req.override_mode == "EMERGENCY_ALL_RED":
            phase = SignalPhaseEnum.ALL_RED
            forced_apps = [ApproachEnum.NORTH, ApproachEnum.SOUTH, ApproachEnum.EAST, ApproachEnum.WEST]
        elif req.override_mode == "HOLD_RED_APPROACH":
            if not forced_apps and req.phase:
                # If specific phase was chosen
                phase = req.phase

        override_data = {
            "junction_id": junction_id,
            "active": True,
            "override_mode": req.override_mode,
            "phase": phase,
            "forced_red_approaches": forced_apps,
            "incident_id": req.incident_id,
            "reason": req.reason,
            "expires_at": expires_at,
            "authorized_by": req.authorized_by,
            "created_at": now
        }
        self._overrides[junction_id] = override_data
        return ManualSignalOverrideResponse(**override_data)

    def get_override(self, junction_id: str) -> Optional[ManualSignalOverrideResponse]:
        data = self._overrides.get(junction_id)
        if not data:
            return None

        # Check expiration
        if data.get("expires_at") and datetime.now(timezone.utc) > data["expires_at"]:
            del self._overrides[junction_id]
            return None

        return ManualSignalOverrideResponse(**data)

    def clear_override(self, junction_id: str) -> bool:
        if junction_id in self._overrides:
            del self._overrides[junction_id]
            return True
        return False

    def list_active_overrides(self) -> Dict[str, ManualSignalOverrideResponse]:
        active = {}
        to_delete = []
        now = datetime.now(timezone.utc)

        for j_id, data in self._overrides.items():
            if data.get("expires_at") and now > data["expires_at"]:
                to_delete.append(j_id)
            else:
                active[j_id] = ManualSignalOverrideResponse(**data)

        for j_id in to_delete:
            del self._overrides[j_id]

        return active

manual_override_manager = ManualOverrideManager()
