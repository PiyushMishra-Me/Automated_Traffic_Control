from enum import Enum
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class WeatherConditionEnum(str, Enum):
    CLEAR = "CLEAR"
    PARTLY_CLOUDY = "PARTLY_CLOUDY"
    CLOUDY = "CLOUDY"
    LIGHT_RAIN = "LIGHT_RAIN"
    HEAVY_RAIN = "HEAVY_RAIN"
    THUNDERSTORM = "THUNDERSTORM"
    FOG = "FOG"
    SNOW = "SNOW"

class RoadSurfaceEnum(str, Enum):
    DRY = "DRY"
    WET = "WET"
    SLIPPERY = "SLIPPERY"
    FLOODED = "FLOODED"
    ICY = "ICY"

class WeatherTimingAdjustments(BaseModel):
    extra_yellow_seconds: float = 0.0
    extra_all_red_seconds: float = 0.0
    saturation_flow_reduction_pct: float = 0.0
    speed_advisory_kmh: int = 50
    safety_advisory: str = "Normal driving conditions."

class WeatherTelemetry(BaseModel):
    junction_id: str
    temperature_c: float = 24.0
    condition: WeatherConditionEnum = WeatherConditionEnum.CLEAR
    precipitation_mm: float = 0.0
    humidity_pct: float = 45.0
    wind_speed_kmh: float = 12.0
    visibility_km: float = 10.0
    road_surface: RoadSurfaceEnum = RoadSurfaceEnum.DRY
    braking_distance_factor: float = 1.0
    adjustments: WeatherTimingAdjustments = Field(default_factory=WeatherTimingAdjustments)
    is_live: bool = True
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class WeatherOverrideRequest(BaseModel):
    condition: Optional[WeatherConditionEnum] = None
    precipitation_mm: Optional[float] = None
    visibility_km: Optional[float] = None
    road_surface: Optional[RoadSurfaceEnum] = None
