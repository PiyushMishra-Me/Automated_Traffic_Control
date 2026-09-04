from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.models.traffic_schemas import (
    CountingLinesUpdate,
    JunctionCreate,
    JunctionTrafficState,
    ApproachTrafficState,
    SignalRecommendation,
    SignalSimulationRequest,
    SignalSimulationResult,
    CorridorSimulationRequest,
    CorridorSimulationResult,
    ManualSignalOverrideRequest,
    ManualSignalOverrideResponse,
)
from backend.db.repositories.junction_repo import junction_repo
from backend.db.repositories.traffic_repo import traffic_repo
from backend.core.analytics.junction_aggregator import JunctionAggregator
from backend.core.control.adaptive_signal import AdaptiveSignalController
from backend.core.control.signal_simulation import TrafficSimulator, CorridorTrafficSimulator
from backend.core.control.manual_override_manager import manual_override_manager

router = APIRouter(prefix="/api/junctions", tags=["Junctions"])

@router.get("/active-signal-overrides", response_model=dict)
def get_all_active_signal_overrides():
    """Retrieve all active police manual signal light overrides across the metropolitan network."""
    overrides = manual_override_manager.list_active_overrides()
    return {k: v.model_dump(mode="json") for k, v in overrides.items()}

@router.get("", response_model=list[dict])
def list_junctions(city: Optional[str] = Query(None, description="Filter junctions by city")):
    return junction_repo.list_junctions(city=city)

@router.post("", response_model=dict)
def create_junction(payload: JunctionCreate):
    return junction_repo.create_junction(payload)

@router.get("/{junction_id}", response_model=dict)
def get_junction(junction_id: str):
    j = junction_repo.get_junction(junction_id)
    if not j:
        raise HTTPException(status_code=404, detail="Junction not found")
    return j

@router.put("/{junction_id}/counting-lines", response_model=dict)
def update_counting_lines(junction_id: str, payload: CountingLinesUpdate):
    updated = junction_repo.update_counting_lines(junction_id, payload.custom_counting_lines)
    if not updated:
        raise HTTPException(status_code=404, detail="Junction not found")
    return {"message": "Counting lines updated", "custom_counting_lines": payload.custom_counting_lines}

@router.get("/{junction_id}/state", response_model=JunctionTrafficState)
def get_junction_state(junction_id: str):
    j = junction_repo.get_junction(junction_id)
    if not j:
        raise HTTPException(status_code=404, detail="Junction not found")
    return _current_junction_state(junction_id)

def _current_junction_state(junction_id: str) -> JunctionTrafficState:
    raw_states = traffic_repo.get_all_latest_for_junction(junction_id)
    states = {}
    for approach, data in raw_states.items():
        try:
            states[approach] = ApproachTrafficState(**data)
        except Exception:
            pass
    return JunctionAggregator.aggregate(junction_id, states)

@router.get("/{junction_id}/signal-recommendation", response_model=SignalRecommendation)
def get_signal_recommendation(junction_id: str):
    if not junction_repo.get_junction(junction_id):
        raise HTTPException(status_code=404, detail="Junction not found")
    return AdaptiveSignalController.recommend(_current_junction_state(junction_id))

@router.post("/{junction_id}/signal-simulation", response_model=SignalSimulationResult)
def simulate_signal_recommendation(junction_id: str, payload: SignalSimulationRequest):
    if not junction_repo.get_junction(junction_id):
        raise HTTPException(status_code=404, detail="Junction not found")
    # Runs a deterministic, time-stepped simulation of the junction over the
    # requested horizon, enforcing any directional manual RED overrides.
    forced_apps = [a.value for a in payload.forced_red_approaches] if payload.forced_red_approaches else None
    return TrafficSimulator.run(
        _current_junction_state(junction_id),
        horizon=payload.horizon_seconds,
        forced_red_approaches=forced_apps,
    )

@router.post("/corridor-simulation", response_model=CorridorSimulationResult)
def simulate_corridor_recommendation(payload: CorridorSimulationRequest):
    if not payload.junction_ids:
        raise HTTPException(status_code=400, detail="Must specify at least one junction ID")
    
    # Build states for all requested junctions
    junction_states = {}
    for j_id in payload.junction_ids:
        if not junction_repo.get_junction(j_id):
            raise HTTPException(status_code=404, detail=f"Junction {j_id} not found")
        junction_states[j_id] = _current_junction_state(j_id)
    
    forced_red_dict = {}
    for j_id, apps in payload.forced_red.items():
        forced_red_dict[j_id] = [a.value if hasattr(a, 'value') else str(a) for a in apps]

    return CorridorTrafficSimulator.run_corridor(
        junction_states=junction_states,
        junction_ids=payload.junction_ids,
        links=payload.links if payload.links else None,
        forced_red=forced_red_dict,
        horizon=payload.horizon_seconds,
    )

@router.post("/{junction_id}/signal-override", response_model=ManualSignalOverrideResponse)
def set_manual_signal_override(junction_id: str, payload: ManualSignalOverrideRequest):
    """
    Apply a manual police emergency signal light override on a junction.
    Enforces EMERGENCY_ALL_RED, HOLD_RED_APPROACH, or FORCED_PHASE for emergency containment.
    """
    if not junction_repo.get_junction(junction_id):
        raise HTTPException(status_code=404, detail=f"Junction {junction_id} not found")
    return manual_override_manager.set_override(junction_id, payload)

@router.get("/{junction_id}/signal-override")
def get_manual_signal_override(junction_id: str):
    """Get active manual police signal override for a junction, if any."""
    override = manual_override_manager.get_override(junction_id)
    if not override:
        return {"junction_id": junction_id, "active": False, "override_mode": "ADAPTIVE_AI"}
    return override.model_dump(mode="json")

@router.delete("/{junction_id}/signal-override")
def clear_manual_signal_override(junction_id: str):
    """Clear manual police signal override and restore automated AI adaptive control."""
    cleared = manual_override_manager.clear_override(junction_id)
    return {
        "junction_id": junction_id,
        "active": False,
        "cleared": cleared,
        "message": "Manual override cleared. Restored to automated AI adaptive signal control."
    }
