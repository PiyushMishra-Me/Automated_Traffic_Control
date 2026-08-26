from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator

class ApproachEnum(str, Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

class TrafficLevelEnum(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY HIGH"


class SignalPhaseEnum(str, Enum):
    NORTH_SOUTH_GREEN = "NORTH_SOUTH_GREEN"
    EAST_WEST_GREEN = "EAST_WEST_GREEN"
    ALL_RED = "ALL_RED"


class AlertSeverityEnum(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class VehicleClassCounts(BaseModel):
    car: int = 0
    motorcycle: int = 0
    bus: int = 0
    truck: int = 0

class CountingLineConfig(BaseModel):
    p1: List[float] = Field(..., description="Normalized [x1, y1] coordinates (0.0 - 1.0)")
    p2: List[float] = Field(..., description="Normalized [x2, y2] coordinates (0.0 - 1.0)")
    orientation: str = "horizontal"  # "horizontal" or "vertical"

    @field_validator("p1", "p2")
    @classmethod
    def validate_normalized_point(cls, point: List[float]) -> List[float]:
        if len(point) != 2 or any(value < 0 or value > 1 for value in point):
            raise ValueError("Counting-line points must have two normalized coordinates between 0 and 1")
        return point


class CountingLinesUpdate(BaseModel):
    """Per-approach counting-line calibration for a junction camera."""
    custom_counting_lines: Dict[ApproachEnum, CountingLineConfig]


class AnalyticsSummary(BaseModel):
    junction_id: str
    approach: Optional[ApproachEnum] = None
    observations: int = 0
    average_vehicle_count: float = 0.0
    average_density: float = 0.0
    average_queue_length: float = 0.0
    latest_flow: float = 0.0
    peak_vehicle_count: int = 0

class ApproachTrafficState(BaseModel):
    approach: ApproachEnum
    vehicle_count: int = Field(0, description="Active vehicles in current frame / scene")
    class_counts: Dict[str, int] = Field(default_factory=lambda: {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0})
    density: float = Field(0.0, description="Density index (0.0 to 1.0)")
    estimated_queue_length: int = Field(0, description="Estimated count of stationary / queued vehicles")
    flow: float = Field(0.0, description="Traffic flow rate (cumulative vehicles counted crossing the line)")
    traffic_level: TrafficLevelEnum = TrafficLevelEnum.LOW
    processed_frames: int = 0
    total_unique_vehicles: int = 0
    annotated_video_url: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class JunctionTrafficState(BaseModel):
    junction_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    north: Optional[ApproachTrafficState] = None
    south: Optional[ApproachTrafficState] = None
    east: Optional[ApproachTrafficState] = None
    west: Optional[ApproachTrafficState] = None
    total_active_vehicles: int = 0
    aggregate_level: TrafficLevelEnum = TrafficLevelEnum.LOW


class TrafficAlert(BaseModel):
    severity: AlertSeverityEnum
    approach: Optional[ApproachEnum] = None
    message: str


class SignalRecommendation(BaseModel):
    junction_id: str
    recommended_phase: SignalPhaseEnum
    green_duration_seconds: int = 0
    yellow_duration_seconds: int = 4
    all_red_duration_seconds: int = 2
    north_south_score: float = 0.0
    east_west_score: float = 0.0
    rationale: str
    alerts: List[TrafficAlert] = Field(default_factory=list)
    is_simulation: bool = True
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SignalSimulationRequest(BaseModel):
    current_phase: SignalPhaseEnum = SignalPhaseEnum.ALL_RED

class JunctionCreate(BaseModel):
    junction_id: str = Field(..., json_schema_extra={"example": "J-MAIN-01"})
    name: str = Field(..., json_schema_extra={"example": "Central Crossing"})
    location: Optional[str] = "Main Avenue & 5th Street"
    custom_counting_lines: Optional[Dict[str, CountingLineConfig]] = None

class JunctionInfo(BaseModel):
    junction_id: str
    name: str
    location: Optional[str] = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    approaches_configured: List[str] = []

class ProcessingJobStatus(BaseModel):
    job_id: str
    junction_id: str
    approach: ApproachEnum
    status: str = "PENDING"  # PENDING, PROCESSING, COMPLETED, FAILED
    progress: float = 0.0    # 0.0 to 100.0
    message: Optional[str] = None
    result: Optional[ApproachTrafficState] = None
    video_filename: Optional[str] = None
    annotated_filename: Optional[str] = None
