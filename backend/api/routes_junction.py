from fastapi import APIRouter, HTTPException
from backend.models.traffic_schemas import JunctionCreate, JunctionInfo, JunctionTrafficState, ApproachTrafficState
from backend.db.repositories.junction_repo import junction_repo
from backend.db.repositories.traffic_repo import traffic_repo
from backend.core.analytics.junction_aggregator import JunctionAggregator

router = APIRouter(prefix="/api/junctions", tags=["Junctions"])

@router.get("", response_model=list[dict])
def list_junctions():
    return junction_repo.list_junctions()

@router.post("", response_model=dict)
def create_junction(payload: JunctionCreate):
    return junction_repo.create_junction(payload)

@router.get("/{junction_id}", response_model=dict)
def get_junction(junction_id: str):
    j = junction_repo.get_junction(junction_id)
    if not j:
        raise HTTPException(status_code=404, detail="Junction not found")
    return j

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
