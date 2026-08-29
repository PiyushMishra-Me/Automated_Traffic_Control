from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.models.traffic_schemas import CountingLinesUpdate, JunctionCreate, JunctionTrafficState, ApproachTrafficState, SignalRecommendation, SignalSimulationRequest, SignalSimulationResult
from backend.db.repositories.junction_repo import junction_repo
from backend.db.repositories.traffic_repo import traffic_repo
from backend.core.analytics.junction_aggregator import JunctionAggregator
from backend.core.control.adaptive_signal import AdaptiveSignalController
from backend.core.control.signal_simulation import TrafficSimulator

router = APIRouter(prefix="/api/junctions", tags=["Junctions"])

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
    return updated

@router.get("/{junction_id}/state", response_model=JunctionTrafficState)
def get_junction_traffic_state(junction_id: str):
    """
    Returns the aggregated traffic state of the junction across all four approaches:
    NORTH, SOUTH, EAST, WEST.
    """
    raw_states = traffic_repo.get_all_latest_for_junction(junction_id)
    
    # Parse into ApproachTrafficState objects
    approach_states = {}
    for app_name, data in raw_states.items():
        try:
            approach_states[app_name] = ApproachTrafficState(**data)
        except Exception:
            pass

    return JunctionAggregator.aggregate(junction_id, approach_states)

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
    # requested horizon. No hardware state is changed; this is analysis only.
    return TrafficSimulator.run(_current_junction_state(junction_id), horizon=payload.horizon_seconds)
