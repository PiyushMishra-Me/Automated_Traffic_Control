from typing import List, Optional
from fastapi import APIRouter, HTTPException
from backend.models.incident_schemas import (
    IncidentCreate,
    IncidentResponse,
    IncidentStatusUpdate,
    DiversionPlan,
)
from backend.db.repositories.incident_repo import incident_repo
from backend.db.repositories.junction_repo import junction_repo

router = APIRouter(prefix="/api/incidents", tags=["Incidents & Diversions"])

@router.get("", response_model=List[IncidentResponse])
def list_incidents(junction_id: Optional[str] = None, status: Optional[str] = None):
    return incident_repo.list_incidents(junction_id=junction_id, status=status)

@router.post("", response_model=IncidentResponse)
def report_incident(payload: IncidentCreate):
    j = junction_repo.get_junction(payload.junction_id)
    if not j:
        raise HTTPException(status_code=404, detail="Junction not found")
    return incident_repo.create_incident(payload)

@router.get("/{incident_id}", response_model=IncidentResponse)
def get_incident(incident_id: str):
    inc = incident_repo.get_incident(incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    return inc

@router.patch("/{incident_id}/status", response_model=IncidentResponse)
def update_incident_status(incident_id: str, payload: IncidentStatusUpdate):
    updated = incident_repo.update_status(incident_id, payload.status)
    if not updated:
        raise HTTPException(status_code=404, detail="Incident not found")
    return updated

@router.get("/junction/{junction_id}/active-diversions", response_model=List[DiversionPlan])
def get_active_junction_diversions(junction_id: str):
    active_incidents = incident_repo.get_active_for_junction(junction_id)
    plans = []
    for inc in active_incidents:
        if inc.diversion_plan and inc.diversion_plan.active:
            plans.append(inc.diversion_plan)
    return plans
