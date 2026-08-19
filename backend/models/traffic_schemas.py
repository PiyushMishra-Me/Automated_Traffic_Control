from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

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

class VehicleClassCounts(BaseModel):
    car: int = 0
    motorcycle: int = 0
    bus: int = 0
    truck: int = 0

class CountingLineConfig(BaseModel):
    p1: List[float] = Field(..., description="Normalized [x1, y1] coordinates (0.0 - 1.0)")
    p2: List[float] = Field(..., description="Normalized [x2, y2] coordinates (0.0 - 1.0)")
    orientation: str = "horizontal"  # "horizontal" or "vertical"

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
