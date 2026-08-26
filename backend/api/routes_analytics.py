from fastapi import APIRouter, HTTPException
from backend.models.traffic_schemas import AnalyticsSummary, ApproachEnum
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

@router.get("/junction/{junction_id}/history")
def get_observation_history(junction_id: str, approach: ApproachEnum | None = None, limit: int = 50):
    return traffic_repo.get_observation_history(junction_id, approach.value if approach else None, limit)

@router.get("/junction/{junction_id}/summary", response_model=AnalyticsSummary)
def get_analytics_summary(junction_id: str, approach: ApproachEnum | None = None, limit: int = 100):
    observations = traffic_repo.get_observation_history(junction_id, approach.value if approach else None, limit)
    if not observations:
        return AnalyticsSummary(junction_id=junction_id, approach=approach)

    count = len(observations)
    return AnalyticsSummary(
        junction_id=junction_id,
        approach=approach,
        observations=count,
        average_vehicle_count=round(sum(item.get("vehicle_count", 0) for item in observations) / count, 2),
        average_density=round(sum(item.get("density", 0) for item in observations) / count, 3),
        average_queue_length=round(sum(item.get("estimated_queue_length", 0) for item in observations) / count, 2),
        latest_flow=float(observations[0].get("flow", 0)),
        peak_vehicle_count=max(item.get("vehicle_count", 0) for item in observations),
    )
