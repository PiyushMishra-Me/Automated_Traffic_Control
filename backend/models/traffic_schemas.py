from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class ApproachEnum(str, Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"


class MovementStateEnum(str, Enum):
    INCOMING = "INCOMING"
    OUTGOING = "OUTGOING"
    STOPPED_INCOMING = "STOPPED_INCOMING"
    STOPPED_OUTGOING = "STOPPED_OUTGOING"
    PARKED = "PARKED"
    UNKNOWN = "UNKNOWN"


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
    ambulance: int = 0


class CountingLineConfig(BaseModel):
    p1: List[float] = Field(
        ...,
        description="Normalized [x1, y1] coordinates (0.0 - 1.0)"
    )
    p2: List[float] = Field(
        ...,
        description="Normalized [x2, y2] coordinates (0.0 - 1.0)"
    )
    orientation: str = "horizontal"

    @field_validator("p1", "p2")
    @classmethod
    def validate_normalized_point(cls, point: List[float]) -> List[float]:
        if len(point) != 2 or any(value < 0 or value > 1 for value in point):
            raise ValueError(
                "Counting-line points must have two normalized coordinates between 0 and 1"
            )
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


class CameraConfig(BaseModel):
    camera_id: str = Field(
        ...,
        json_schema_extra={"example": "CAM-J1-NORTH"}
    )
    junction_id: str = Field(
        ...,
        json_schema_extra={"example": "J-MAIN-01"}
    )
    approach: ApproachEnum = ApproachEnum.NORTH

    roi: Optional[List[float]] = Field(
        None,
        description=(
            "Normalized ROI [x1, y1, x2, y2] or "
            "pixel [x1, y1, x2, y2], or None for full frame"
        ),
    )

    junction_vector: List[float] = Field(
        default_factory=lambda: [0.0, 1.0],
        description="Normalized [dx, dy] pointing toward junction",
    )

    counting_line: Optional[CountingLineConfig] = None

    incoming_corridor: Optional[List[List[float]]] = Field(
        None,
        description="Normalized polygon for incoming traffic corridor",
    )

    outgoing_corridor: Optional[List[List[float]]] = Field(
        None,
        description="Normalized polygon for outgoing traffic corridor",
    )

    fps: float = 25.0
    is_bidirectional: bool = False


class ApproachTrafficState(BaseModel):
    approach: ApproachEnum

    vehicle_count: int = Field(
        0,
        description="Active vehicles in current frame / scene (excluding parked)",
    )

    class_counts: Dict[str, int] = Field(
        default_factory=lambda: {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
            "ambulance": 0,
        }
    )

    # Emergency vehicle information
    ambulance_count: int = Field(
        0,
        description="Number of active ambulances detected",
    )

    emergency_detected: bool = Field(
        False,
        description="True when at least one ambulance is detected",
    )

    density: float = Field(
        0.0,
        description="Density index (0.0 to 1.0)",
    )

    estimated_queue_length: int = Field(
        0,
        description="Estimated count of stationary / queued vehicles",
    )

    flow: float = Field(
        0.0,
        description="Traffic flow rate (cumulative vehicles counted crossing the line)",
    )

    traffic_level: TrafficLevelEnum = TrafficLevelEnum.LOW
    processed_frames: int = 0
    total_unique_vehicles: int = 0

    # Directional movement metrics
    incoming_count: int = Field(
        0,
        description="Active incoming vehicles (moving + stopped)",
    )

    outgoing_count: int = Field(
        0,
        description="Active outgoing vehicles (moving + stopped)",
    )

    stopped_incoming_count: int = Field(
        0,
        description="Stopped vehicles that were incoming",
    )

    stopped_outgoing_count: int = Field(
        0,
        description="Stopped vehicles that were outgoing",
    )

    parked_count: int = Field(
        0,
        description="Parked vehicles (stationary > 5 mins)",
    )

    unknown_direction_count: int = Field(
        0,
        description="Vehicles with unknown movement state",
    )

    incoming_flow: float = Field(
        0.0,
        description="Cumulative incoming vehicles crossing counting line",
    )

    outgoing_flow: float = Field(
        0.0,
        description="Cumulative outgoing vehicles crossing counting line",
    )

    annotated_video_url: Optional[str] = None

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class JunctionTrafficState(BaseModel):
    junction_id: str

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

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

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SignalSimulationRequest(BaseModel):
    current_phase: SignalPhaseEnum = SignalPhaseEnum.ALL_RED

    horizon_seconds: int = Field(
        180,
        ge=30,
        le=600,
    )


class SimulationStep(BaseModel):
    """A single one-second snapshot of the adaptive simulation timeline."""

    t: int = Field(
        ...,
        description="Elapsed simulated seconds",
    )

    phase: SignalPhaseEnum
    phase_label: str

    phase_time_remaining: int = Field(
        0,
        description="Seconds left in the current phase",
    )

    lights: Dict[str, str] = Field(
        ...,
        description="Approach -> GREEN | YELLOW | RED",
    )

    queues: Dict[str, int] = Field(
        ...,
        description="Approach -> vehicles currently queued",
    )

    served_total: int = Field(
        0,
        description="Cumulative vehicles discharged so far",
    )


class ApproachSimSummary(BaseModel):
    approach: ApproachEnum
    arrivals: int = 0
    served: int = 0
    max_queue: int = 0
    final_queue: int = 0

    avg_wait: float = Field(
        0.0,
        description="Average delay per vehicle (seconds)",
    )


class SimulationComparison(BaseModel):
    """Adaptive controller vs a naive fixed-timer baseline."""

    adaptive_avg_wait: float = 0.0
    fixed_avg_wait: float = 0.0
    adaptive_served: int = 0
    fixed_served: int = 0

    improvement_pct: float = Field(
        0.0,
        description="Reduction in avg wait vs fixed-timer (%)",
    )


class SignalSimulationResult(BaseModel):
    junction_id: str
    total_seconds: int
    steps: List[SimulationStep] = Field(default_factory=list)
    per_approach: List[ApproachSimSummary] = Field(default_factory=list)
    comparison: SimulationComparison = Field(
        default_factory=SimulationComparison
    )
    recommendation: SignalRecommendation
    rationale: str = ""

    seeded_demo: bool = Field(
        False,
        description=(
            "True when no observations existed and a demo scenario was used"
        ),
    )

    is_simulation: bool = True

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class JunctionCreate(BaseModel):
    junction_id: str = Field(
        ...,
        json_schema_extra={"example": "J-MAIN-01"},
    )

    name: str = Field(
        ...,
        json_schema_extra={"example": "Central Crossing"},
    )

    location: Optional[str] = "Main Avenue & 5th Street"

    latitude: float = Field(
        28.6139,
        description="Geographic latitude coordinate",
    )
    longitude: float = Field(
        77.2090,
        description="Geographic longitude coordinate",
    )
    road_names: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {
            "NORTH": "North Boulevard",
            "SOUTH": "South Expressway",
            "EAST": "East Arterial Corridor",
            "WEST": "West Linkway",
        }
    )
    connected_junctions: Optional[List[str]] = Field(
        default_factory=list
    )

    custom_counting_lines: Optional[
        Dict[str, CountingLineConfig]
    ] = None


class JunctionInfo(BaseModel):
    junction_id: str
    name: str
    location: Optional[str] = ""

    latitude: float = 28.6139
    longitude: float = 77.2090
    road_names: Dict[str, str] = Field(
        default_factory=dict
    )
    connected_junctions: List[str] = Field(
        default_factory=list
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    approaches_configured: List[str] = []


class ProcessingJobStatus(BaseModel):
    job_id: str
    junction_id: str
    approach: ApproachEnum
    status: str = "PENDING"
    progress: float = 0.0
    message: Optional[str] = None
    result: Optional[ApproachTrafficState] = None
    video_filename: Optional[str] = None
    annotated_filename: Optional[str] = None


class BatchUploadResponse(BaseModel):
    junction_id: str
    jobs: List[ProcessingJobStatus] = Field(default_factory=list)
    message: str = "Batch upload initiated"


class LiveStreamConfigRequest(BaseModel):
    junction_id: str
    approach: ApproachEnum
    stream_type: str = "RTSP"  # "RTSP" | "HLS" | "WEBRTC" | "DEVICE_WEBCAM" | "SIMULATION"
    stream_url: Optional[str] = None
    is_active: bool = True
    sampling_fps: float = 5.0

