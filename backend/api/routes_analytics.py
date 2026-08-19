from fastapi import APIRouter, HTTPException
from backend.models.traffic_schemas import ApproachTrafficState
from backend.db.repositories.traffic_repo import traffic_repo

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])

@router.get("/junction/{junction_id}/approach/{approach}")
def get_latest_approach_state(junction_id: str, approach: str):
    data = traffic_repo.get_latest_approach_observation(junction_id, approach)
    if not data:
        raise HTTPException(status_code=404, detail=f"No traffic observation found for approach {approach}")
    return data

@router.get("/junction/{junction_id}/all")
def get_all_approaches_state(junction_id: str):
    return traffic_repo.get_all_latest_for_junction(junction_id)
