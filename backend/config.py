import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    APP_NAME: str = "Intelligent Traffic Management System - Phase 1"
    DEBUG: bool = True
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    ANNOTATED_DIR: Path = DATA_DIR / "annotated"
    MODEL_PATH: str = "yolov8n.pt"
    
    # MongoDB
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DATABASE_NAME: str = "traffic_management_db"
    
    # Vision & Detection
    CONFIDENCE_THRESHOLD: float = 0.30
    IOU_THRESHOLD: float = 0.45
    TARGET_CLASSES: list[int] = [2, 3, 5, 7] # 2: car, 3: motorcycle, 5: bus, 7: truck
    CLASS_NAMES: dict[int, str] = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }
    
    # Approach-specific counting line relative coordinates (x1, y1, x2, y2)
    # Allows independent configuration for each camera viewpoint
    DEFAULT_COUNTING_LINES: dict = {
        "NORTH": {"p1": [0.1, 0.65], "p2": [0.9, 0.65], "orientation": "horizontal"},
        "SOUTH": {"p1": [0.1, 0.35], "p2": [0.9, 0.35], "orientation": "horizontal"},
        "EAST": {"p1": [0.35, 0.1], "p2": [0.35, 0.9], "orientation": "vertical"},
        "WEST": {"p1": [0.65, 0.1], "p2": [0.65, 0.9], "orientation": "vertical"},
    }
    
    # Traffic Level Thresholds (based on active vehicle count & density)
    THRESHOLD_LOW: int = 4
    THRESHOLD_MEDIUM: int = 9
    THRESHOLD_HIGH: int = 16

    # Queue Speed Threshold (pixels per frame below which vehicle is considered queued/stationary)
    QUEUE_SPEED_THRESHOLD: float = 2.5

settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
