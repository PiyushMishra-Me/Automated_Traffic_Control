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
from backend.api.routes_navigation import router as navigation_router

app = FastAPI(
    title=settings.APP_NAME,
    description="Automated traffic monitoring, dual-model vision AI, real-time camera inference, geospatial map, dynamic routing, weather-adaptive control, and green wave emergency corridors",
    version="1.0.0"
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
app.include_router(navigation_router)

@app.get("/")
def root():
    return {
        "system": settings.APP_NAME,
        "status": "online",
        "endpoints": {
            "docs": "/docs",
            "junctions": "/api/junctions",
            "video_upload": "/api/videos/upload",
            "analytics": "/api/analytics",
            "incidents": "/api/incidents",
            "weather": "/api/weather",
            "navigation": "/api/navigation/route"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
