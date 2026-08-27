from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.api.routes_junction import router as junction_router
from backend.api.routes_video import router as video_router
from backend.api.routes_analytics import router as analytics_router
from backend.api.routes_incident import router as incident_router
from backend.api.routes_weather import router as weather_router
from backend.api.routes_auth import router as auth_router
from backend.api.routes_ambulance import router as ambulance_router

app = FastAPI(
    title=settings.APP_NAME,
    description="Phase 3: Traffic monitoring, analytics, live map, upstream diversion, weather-adaptive control, ambulance green corridors, and role-based profiles",
    version="3.2.0"
)

# Enable CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(junction_router)
app.include_router(video_router)
app.include_router(analytics_router)
app.include_router(incident_router)
app.include_router(weather_router)
app.include_router(auth_router)
app.include_router(ambulance_router)

@app.get("/")
def root():
    return {
        "system": settings.APP_NAME,
        "phase": 3,
        "status": "online",
        "endpoints": {
            "docs": "/docs",
            "junctions": "/api/junctions",
            "video_upload": "/api/videos/upload",
            "analytics": "/api/analytics",
            "incidents": "/api/incidents",
            "weather": "/api/weather"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
