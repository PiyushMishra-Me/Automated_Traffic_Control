from typing import List
from fastapi import APIRouter, HTTPException, status
from backend.models.navigation_schemas import (
    NavigationRequest,
    NavigationResponse,
    CorridorCostDetail
)
from backend.core.control.navigation_engine import navigation_engine

router = APIRouter(prefix="/api/navigation", tags=["Public Navigation & Route Optimization"])

@router.post("/route", response_model=NavigationResponse)
def compute_route(request: NavigationRequest):
    """
    Computes the shortest and fastest optimal path for public commuters,
    taking into account live traffic congestion, active incidents,
    weather surface friction, and active emergency vehicle Green Wave preemption.
    """
    try:
        return navigation_engine.compute_optimal_route(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Navigation routing failed: {str(e)}"
        )

@router.get("/corridors", response_model=List[CorridorCostDetail])
def get_all_corridor_statuses():
    """
    Returns the real-time weight, congestion multiplier, and emergency status
    for all road corridors in the metropolitan grid.
    """
    try:
        return navigation_engine.list_all_corridor_statuses()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch corridor statuses: {str(e)}"
        )
