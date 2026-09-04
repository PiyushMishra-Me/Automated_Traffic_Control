from enum import Enum
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from backend.models.traffic_schemas import ApproachEnum

class IncidentTypeEnum(str, Enum):
    ACCIDENT = "ACCIDENT"
    VEHICLE_BREAKDOWN = "VEHICLE_BREAKDOWN"
    ROAD_HAZARD = "ROAD_HAZARD"
    WATERLOGGING = "WATERLOGGING"
    ROAD_WORK = "ROAD_WORK"

class IncidentSeverityEnum(str, Enum):
    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    CRITICAL_ROAD_BLOCKED = "CRITICAL_ROAD_BLOCKED"

class IncidentStatusEnum(str, Enum):
    REPORTED = "REPORTED"
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"

class DiversionStep(BaseModel):
    step_number: int
    instruction: str
    corridor: str
    upstream_junction_id: Optional[str] = None
    signal_action: str

class DiversionPlan(BaseModel):
    affected_junction_id: str
    affected_approach: ApproachEnum
    severity: IncidentSeverityEnum
    bypass_junction_id: Optional[str] = None
    recommended_reroute_corridor: str
    signal_timing_strategy: str
    steps: List[DiversionStep] = Field(default_factory=list)
    active: bool = True

class IncidentCreate(BaseModel):
    junction_id: str
    approach: ApproachEnum
    road_name: Optional[str] = "Main Arterial Avenue"
    incident_type: IncidentTypeEnum = IncidentTypeEnum.ACCIDENT
    severity: IncidentSeverityEnum = IncidentSeverityEnum.SEVERE
    description: str
    estimated_clearance_minutes: int = 30
    reported_by: Optional[str] = "Traffic Operations Center"
    reporter_role: Optional[str] = Field("TRAFFIC_POLICE", description="Role of reporter: TRAFFIC_POLICE, PUBLIC_CITIZEN, etc.")
    dispatch_call_ref: Optional[str] = Field(None, description="Official call log or dispatch reference for base reports")
    lat: Optional[float] = None
    lng: Optional[float] = None
    photo_base64: Optional[str] = Field(None, description="On-the-spot camera captured live photo base64 data URI")
    is_live_captured: bool = Field(False, description="Strict enforcement: photo was captured live from camera viewfinder")
    capture_timestamp: Optional[str] = None

class IncidentStatusUpdate(BaseModel):
    status: IncidentStatusEnum

class IncidentResponse(BaseModel):
    incident_id: str
    junction_id: str
    approach: ApproachEnum
    road_name: str
    incident_type: IncidentTypeEnum
    severity: IncidentSeverityEnum
    status: IncidentStatusEnum
    description: str
    estimated_clearance_minutes: int
    reported_by: str
    reporter_role: Optional[str] = "TRAFFIC_POLICE"
    dispatch_call_ref: Optional[str] = None
    lat: float
    lng: float
    photo_base64: Optional[str] = None
    is_live_captured: bool = False
    capture_timestamp: Optional[str] = None
    diversion_plan: Optional[DiversionPlan] = None
    reported_at: datetime
    updated_at: datetime
