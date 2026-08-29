from fastapi import APIRouter, HTTPException
from backend.models.weather_schemas import WeatherTelemetry, WeatherOverrideRequest
from backend.core.weather.weather_service import weather_service
from backend.db.repositories.junction_repo import junction_repo

router = APIRouter(prefix="/api/weather", tags=["Weather"])

@router.get("/junction/{junction_id}", response_model=WeatherTelemetry)
def get_junction_weather(junction_id: str):
    j = junction_repo.get_junction(junction_id)
    if not j:
        raise HTTPException(status_code=404, detail="Junction not found")
    
    lat = float(j.get("latitude", 28.6139))
    lng = float(j.get("longitude", 77.2090))
    return weather_service.get_weather_for_junction(junction_id, lat=lat, lng=lng)

@router.post("/junction/{junction_id}/override", response_model=WeatherTelemetry)
def override_junction_weather(junction_id: str, payload: WeatherOverrideRequest):
    j = junction_repo.get_junction(junction_id)
    if not j:
        raise HTTPException(status_code=404, detail="Junction not found")
    
    override_dict = {k: v for k, v in payload.model_dump().items() if v is not None}
    return weather_service.set_override(junction_id, override_dict)

@router.delete("/junction/{junction_id}/override", response_model=WeatherTelemetry)
def clear_junction_weather_override(junction_id: str):
    j = junction_repo.get_junction(junction_id)
    if not j:
        raise HTTPException(status_code=404, detail="Junction not found")
    
    weather_service.clear_override(junction_id)
    lat = float(j.get("latitude", 28.6139))
    lng = float(j.get("longitude", 77.2090))
    return weather_service.get_weather_for_junction(junction_id, lat=lat, lng=lng)
