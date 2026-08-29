from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from backend.models.traffic_schemas import ApproachEnum

class VehicleProfileEnum(str, Enum):
    CAR = "CAR"
    TWO_WHEELER = "TWO_WHEELER"
    BUS_HEAVY = "BUS_HEAVY"
    EV = "EV"

class CorridorCostDetail(BaseModel):
    origin_junction_id: str
    destination_junction_id: str
    road_name: str
    approach: ApproachEnum
    base_distance_km: float
    free_flow_seconds: int
    live_traffic_multiplier: float
    weather_multiplier: float
    has_active_incident: bool = False
    has_active_emergency: bool = False
    active_emergency_mission_id: Optional[str] = None
    active_emergency_priority: int = 0
    estimated_transit_seconds: int
    congestion_level: str = "LOW"

class NavigationStep(BaseModel):
    step_number: int
    from_junction_id: str
    to_junction_id: str
    road_name: str
    approach: ApproachEnum
    instruction: str
    distance_km: float
    eta_seconds: int
    congestion_level: str
    emergency_active: bool = False
    emergency_priority: int = 0
    emergency_warning: Optional[str] = None
    advisory_notes: Optional[str] = None

class NavigationRequest(BaseModel):
    origin_junction_id: str = "J-04"
    destination_junction_id: str = "J-02"
    vehicle_type: VehicleProfileEnum = VehicleProfileEnum.CAR
    avoid_high_congestion: bool = True
    priority_mode: str = "FASTEST_TRAFFIC_AWARE" # "FASTEST_TRAFFIC_AWARE" | "SHORTEST_DISTANCE"

class NavigationResponse(BaseModel):
    origin_junction_id: str
    destination_junction_id: str
    origin_name: str
    destination_name: str
    vehicle_type: VehicleProfileEnum
    total_distance_km: float
    estimated_travel_time_seconds: int
    estimated_travel_time_formatted: str
    delay_seconds: int
    delay_saved_seconds: int
    optimal_route_junctions: List[str]
    alternative_route_junctions: List[str] = []
    steps: List[NavigationStep]
    emergency_corridor_warnings: List[str] = []
    load_balancing_advisory: Optional[str] = None
    weather_impact_advisory: Optional[str] = None
