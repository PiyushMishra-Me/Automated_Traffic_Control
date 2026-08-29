from enum import Enum
from typing import List, Optional, Dict
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from backend.models.traffic_schemas import ApproachEnum

class EmergencyAgencyEnum(str, Enum):
    HOSPITAL = "HOSPITAL"
    FIRE_RESCUE = "FIRE_RESCUE"
    POLICE_DISASTER = "POLICE_DISASTER"

class AmbulanceCriticalityEnum(str, Enum):
    LOW = "LOW"                                    # Priority 1: Stable transfer, routine medical
    MEDIUM = "MEDIUM"                              # Priority 2: Urgent, moderate trauma / fracture
    HIGH = "HIGH"                                  # Priority 3: Severe trauma, emergency surgery needed
    CRITICAL_LIFE_THREATENING = "CRITICAL_LIFE_THREATENING" # Priority 4: Cardiac arrest, stroke, Golden Hour ICU

CRITICALITY_PRIORITY_MAP: Dict[AmbulanceCriticalityEnum, int] = {
    AmbulanceCriticalityEnum.CRITICAL_LIFE_THREATENING: 4,
    AmbulanceCriticalityEnum.HIGH: 3,
    AmbulanceCriticalityEnum.MEDIUM: 2,
    AmbulanceCriticalityEnum.LOW: 1,
}

class AmbulanceStatusEnum(str, Enum):
    DISPATCHED_TO_VICTIM = "DISPATCHED_TO_VICTIM"
    ON_SCENE_PICKUP = "ON_SCENE_PICKUP"
    TRANSIT_TO_HOSPITAL = "TRANSIT_TO_HOSPITAL"
    MISSION_ACCOMPLISHED = "MISSION_ACCOMPLISHED"
    CANCELLED = "CANCELLED"

class RouteJunctionNode(BaseModel):
    junction_id: str
    approach: ApproachEnum
    corridor_name: str
    eta_seconds: int
    preemption_active: bool = False

class ConflictResolutionResult(BaseModel):
    has_conflict: bool = False
    junction_id: Optional[str] = None
    winning_mission_id: Optional[str] = None
    winning_criticality: Optional[AmbulanceCriticalityEnum] = None
    winning_approach: Optional[ApproachEnum] = None
    secondary_mission_id: Optional[str] = None
    secondary_approach: Optional[ApproachEnum] = None
    strategy: str = "No intersecting emergency vehicle conflicts detected."

class AmbulanceMissionCreate(BaseModel):
    agency_type: EmergencyAgencyEnum = EmergencyAgencyEnum.HOSPITAL
    hospital_name: str = "City Central Trauma Hospital"
    ambulance_vehicle_id: str = "DL-01-AMB-8899"
    driver_contact: Optional[str] = "+91 98765 43210"
    criticality: AmbulanceCriticalityEnum = AmbulanceCriticalityEnum.CRITICAL_LIFE_THREATENING
    patient_condition: str = "Cardiac arrest / acute respiratory distress"
    victim_location: str = "Central Plaza Crossing / Ring Road"
    origin_junction_id: str = "J-04"
    destination_junction_id: str = "J-02"

class AmbulanceMissionResponse(BaseModel):
    mission_id: str
    agency_type: EmergencyAgencyEnum = EmergencyAgencyEnum.HOSPITAL
    hospital_name: str
    ambulance_vehicle_id: str
    driver_contact: str
    criticality: AmbulanceCriticalityEnum
    priority_level: int # 1 to 4
    patient_condition: str
    victim_location: str
    origin_junction_id: str
    destination_junction_id: str
    route_corridor: List[RouteJunctionNode]
    active_node_index: int = 0
    status: AmbulanceStatusEnum
    current_lat: float
    current_lng: float
    estimated_total_eta_seconds: int
    conflict_resolution: Optional[ConflictResolutionResult] = None
    dispatched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AmbulancePreemptionStatus(BaseModel):
    junction_id: str
    is_preempted: bool
    active_mission_id: Optional[str] = None
    priority_level: int = 0
    preempted_approach: Optional[ApproachEnum] = None
    clearing_phase_duration_seconds: int = 0
    advisory: str = "Normal adaptive cycle."
