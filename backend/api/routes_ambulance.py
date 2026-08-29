"""
routes_ambulance.py
FastAPI router for hospital ambulance emergency dispatch and multi-junction orchestration.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from backend.models.ambulance_schemas import (
    AmbulanceMissionCreate,
    AmbulanceMissionResponse,
    AmbulanceStatusEnum,
    AmbulancePreemptionStatus
)
from backend.db.repositories.ambulance_repo import ambulance_repo
from backend.core.control.ambulance_engine import ambulance_engine
from backend.core.control.emergency_orchestrator import emergency_orchestrator

router = APIRouter(prefix="/api/ambulances", tags=["Hospital Ambulance Emergency Dispatch"])

@router.post("/register", response_model=AmbulanceMissionResponse)
def register_ambulance_mission(payload: AmbulanceMissionCreate):
    """
    Register and launch an emergency ambulance mission with criticality level
    (LOW, MEDIUM, HIGH, CRITICAL_LIFE_THREATENING). Calculates shortest route
    and triggers dynamic Green Wave corridor preemption.
    """
    try:
        mission = ambulance_repo.register_mission(payload)
        # Register into multi-junction Emergency Orchestrator
        emergency_orchestrator.register_mission(
            mission_id=mission.mission_id,
            vehicle_id=mission.ambulance_vehicle_id,
            origin_junction_id=mission.origin_junction_id,
            destination_junction_id=mission.destination_junction_id,
            criticality=mission.criticality,
            route_nodes=mission.route_corridor
        )
        return mission
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register ambulance mission: {str(e)}")

@router.get("", response_model=List[AmbulanceMissionResponse])
def list_ambulance_missions(
    status: Optional[str] = Query(None),
    hospital_name: Optional[str] = Query(None)
):
    """List all registered ambulance emergency missions."""
    return ambulance_repo.list_missions(status=status, hospital_name=hospital_name)

@router.get("/{mission_id}", response_model=AmbulanceMissionResponse)
def get_ambulance_mission(mission_id: str):
    """Get real-time details, route telemetry, and conflict status of a specific ambulance mission."""
    mission = ambulance_repo.get_mission(mission_id)
    if not mission:
        raise HTTPException(status_code=404, detail="Ambulance mission not found")
    return mission

@router.patch("/{mission_id}/status", response_model=AmbulanceMissionResponse)
def update_mission_status(mission_id: str, new_status: AmbulanceStatusEnum):
    """Progress an ambulance mission status (DISPATCHED -> ON_SCENE_PICKUP -> TRANSIT_TO_HOSPITAL -> MISSION_ACCOMPLISHED)."""
    mission = ambulance_repo.update_status(mission_id, new_status)
    if not mission:
        raise HTTPException(status_code=404, detail="Ambulance mission not found")

    context = emergency_orchestrator.get_mission_context(mission_id)
    if context:
        context.status = new_status
        if new_status == AmbulanceStatusEnum.MISSION_ACCOMPLISHED:
            context.is_completed = True

    return mission

@router.get("/junction/{junction_id}/preemption", response_model=AmbulancePreemptionStatus)
def get_junction_preemption(junction_id: str):
    """Get active emergency preemption status and multi-ambulance conflict resolution for a junction."""
    active_missions = ambulance_repo.list_missions(status="DISPATCHED_TO_VICTIM") + ambulance_repo.list_missions(status="TRANSIT_TO_HOSPITAL")
    return ambulance_engine.get_preemption_for_junction(junction_id, active_missions)
