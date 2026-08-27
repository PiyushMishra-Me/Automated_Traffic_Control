import urllib.request
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from backend.models.weather_schemas import (
    WeatherConditionEnum,
    RoadSurfaceEnum,
    WeatherTimingAdjustments,
    WeatherTelemetry,
)

logger = logging.getLogger("traffic_weather")

class WeatherService:
    def __init__(self):
        self._cache: Dict[str, Tuple[datetime, WeatherTelemetry]] = {}
        self._overrides: Dict[str, dict] = {}

    def get_weather_for_junction(
        self,
        junction_id: str,
        lat: float = 28.6139,
        lng: float = 77.2090
    ) -> WeatherTelemetry:
        """
        Retrieves weather telemetry for a junction coordinate.
        Uses cached data if fetched within 5 minutes, or live queries Open-Meteo.
        """
        # Check override
        override = self._overrides.get(junction_id)
        if override:
            return self._build_telemetry_from_raw(
                junction_id=junction_id,
                temp=override.get("temperature_c", 22.0),
                precip=override.get("precipitation_mm", 12.0),
                humidity=override.get("humidity_pct", 85.0),
                wind=override.get("wind_speed_kmh", 25.0),
                visibility=override.get("visibility_km", 2.5),
                condition=override.get("condition", WeatherConditionEnum.HEAVY_RAIN),
                road_surface=override.get("road_surface", RoadSurfaceEnum.WET),
                is_live=False
            )

        now = datetime.now(timezone.utc)
        cached = self._cache.get(junction_id)
        if cached and (now - cached[0]).total_seconds() < 300:
            return cached[1]

        # Live fetch from Open-Meteo
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat:.4f}&longitude={lng:.4f}&"
                f"current=temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m,visibility"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "IntelligentTrafficControl/3.0"})
            with urllib.request.urlopen(req, timeout=3.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                current = data.get("current", {})
                
                temp = float(current.get("temperature_2m", 25.0))
                humidity = float(current.get("relative_humidity_2m", 50.0))
                precip = float(current.get("precipitation", 0.0))
                wind = float(current.get("wind_speed_10m", 10.0))
                vis_m = float(current.get("visibility", 10000.0))
                vis_km = round(vis_m / 1000.0, 1)
                w_code = int(current.get("weather_code", 0))

                condition = self._map_wmo_code(w_code, precip)
                road_surface = self._determine_road_surface(precip, condition)

                telemetry = self._build_telemetry_from_raw(
                    junction_id=junction_id,
                    temp=temp,
                    precip=precip,
                    humidity=humidity,
                    wind=wind,
                    visibility=vis_km,
                    condition=condition,
                    road_surface=road_surface,
                    is_live=True
                )
                self._cache[junction_id] = (now, telemetry)
                return telemetry

        except Exception as e:
            logger.warning(f"Failed to fetch live weather for {junction_id}: {e}. Using deterministic standard profile.")
            fallback = self._build_telemetry_from_raw(
                junction_id=junction_id,
                temp=26.5,
                precip=0.0,
                humidity=48.0,
                wind=11.2,
                visibility=9.5,
                condition=WeatherConditionEnum.CLEAR,
                road_surface=RoadSurfaceEnum.DRY,
                is_live=False
            )
            return fallback

    def set_override(self, junction_id: str, override_data: dict) -> WeatherTelemetry:
        self._overrides[junction_id] = override_data
        return self.get_weather_for_junction(junction_id)

    def clear_override(self, junction_id: str):
        if junction_id in self._overrides:
            del self._overrides[junction_id]
        if junction_id in self._cache:
            del self._cache[junction_id]

    def _map_wmo_code(self, code: int, precip: float) -> WeatherConditionEnum:
        if code in [95, 96, 99]:
            return WeatherConditionEnum.THUNDERSTORM
        if code in [65, 80, 81, 82] or precip > 5.0:
            return WeatherConditionEnum.HEAVY_RAIN
        if code in [51, 53, 55, 61, 63] or precip > 0.1:
            return WeatherConditionEnum.LIGHT_RAIN
        if code in [45, 48]:
            return WeatherConditionEnum.FOG
        if code in [71, 73, 75, 85, 86]:
            return WeatherConditionEnum.SNOW
        if code in [1, 2]:
            return WeatherConditionEnum.PARTLY_CLOUDY
        if code == 3:
            return WeatherConditionEnum.CLOUDY
        return WeatherConditionEnum.CLEAR

    def _determine_road_surface(self, precip: float, condition: WeatherConditionEnum) -> RoadSurfaceEnum:
        if condition == WeatherConditionEnum.SNOW:
            return RoadSurfaceEnum.ICY
        if condition in [WeatherConditionEnum.THUNDERSTORM, WeatherConditionEnum.HEAVY_RAIN] or precip >= 8.0:
            return RoadSurfaceEnum.FLOODED
        if condition == WeatherConditionEnum.LIGHT_RAIN or precip > 0.0:
            return RoadSurfaceEnum.WET
        if condition == WeatherConditionEnum.FOG:
            return RoadSurfaceEnum.SLIPPERY
        return RoadSurfaceEnum.DRY

    def _build_telemetry_from_raw(
        self,
        junction_id: str,
        temp: float,
        precip: float,
        humidity: float,
        wind: float,
        visibility: float,
        condition: WeatherConditionEnum,
        road_surface: RoadSurfaceEnum,
        is_live: bool
    ) -> WeatherTelemetry:
        # Dynamic Safety Adjustments
        if road_surface == RoadSurfaceEnum.FLOODED or condition == WeatherConditionEnum.THUNDERSTORM:
            adjustments = WeatherTimingAdjustments(
                extra_yellow_seconds=2.0,
                extra_all_red_seconds=2.5,
                saturation_flow_reduction_pct=30.0,
                speed_advisory_kmh=25,
                safety_advisory="Severe weather & flooded surface. Extended amber & all-red safety clearance active; max speed 25 km/h."
            )
            brake_factor = 2.0
        elif condition == WeatherConditionEnum.HEAVY_RAIN or road_surface == RoadSurfaceEnum.SLIPPERY:
            adjustments = WeatherTimingAdjustments(
                extra_yellow_seconds=1.5,
                extra_all_red_seconds=2.0,
                saturation_flow_reduction_pct=20.0,
                speed_advisory_kmh=35,
                safety_advisory="Heavy rain / wet asphalt. Extended clearance active to prevent braking skids; advisory 35 km/h."
            )
            brake_factor = 1.6
        elif condition == WeatherConditionEnum.FOG or visibility < 1.5:
            adjustments = WeatherTimingAdjustments(
                extra_yellow_seconds=2.0,
                extra_all_red_seconds=2.0,
                saturation_flow_reduction_pct=25.0,
                speed_advisory_kmh=30,
                safety_advisory="Dense fog & low visibility. Headway gap limits increased; advisory 30 km/h with low beams."
            )
            brake_factor = 1.4
        elif condition == WeatherConditionEnum.LIGHT_RAIN or road_surface == RoadSurfaceEnum.WET:
            adjustments = WeatherTimingAdjustments(
                extra_yellow_seconds=1.0,
                extra_all_red_seconds=1.0,
                saturation_flow_reduction_pct=10.0,
                speed_advisory_kmh=40,
                safety_advisory="Wet roads. Reduced traction; maintain safe following distance."
            )
            brake_factor = 1.3
        else:
            adjustments = WeatherTimingAdjustments(
                extra_yellow_seconds=0.0,
                extra_all_red_seconds=0.0,
                saturation_flow_reduction_pct=0.0,
                speed_advisory_kmh=50,
                safety_advisory="Normal traffic conditions and optimal road surface traction."
            )
            brake_factor = 1.0

        return WeatherTelemetry(
            junction_id=junction_id,
            temperature_c=temp,
            condition=condition,
            precipitation_mm=precip,
            humidity_pct=humidity,
            wind_speed_kmh=wind,
            visibility_km=visibility,
            road_surface=road_surface,
            braking_distance_factor=brake_factor,
            adjustments=adjustments,
            is_live=is_live,
            last_updated=datetime.now(timezone.utc)
        )

weather_service = WeatherService()
