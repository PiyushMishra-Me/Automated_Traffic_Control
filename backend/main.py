from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.api.routes_junction import router as junction_router
from backend.api.routes_video import router as video_router
from backend.api.routes_analytics import router as analytics_router

app = FastAPI(
    title=settings.APP_NAME,
    description="Phase 1: Real-Time Video Traffic Monitoring Foundation with YOLOv8n and ByteTrack",
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

@app.get("/")
def root():
    return {
        "system": settings.APP_NAME,
        "phase": 1,
        "status": "online",
        "endpoints": {
            "docs": "/docs",
            "junctions": "/api/junctions",
            "video_upload": "/api/videos/upload",
            "analytics": "/api/analytics"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
