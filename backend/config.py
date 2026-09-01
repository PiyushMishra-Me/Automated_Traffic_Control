import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    # =====================================================
    # App
    # =====================================================

    APP_NAME: str = "Automated Traffic Control & Intelligent Safety System"
    DEBUG: bool = True

    # =====================================================
    # Paths
    # =====================================================

    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    ANNOTATED_DIR: Path = DATA_DIR / "annotated"

    # =====================================================
    # Models
    # =====================================================

    # Main YOLO model for normal vehicles (YOLO11)
    MODEL_PATH: str = "yolo11s.pt"

    # Custom ambulance detection model
    AMBULANCE_MODEL_PATH: str = str(
        BASE_DIR
        / "runs"
        / "detect"
        / "runs"
        / "detect"
        / "ambulance_with_negatives-2"
        / "weights"
        / "best.pt"
    )

    # Optional full path to FFmpeg.
    # None means the backend searches the system PATH.
    FFMPEG_BINARY: str | None = None

    # =====================================================
    # MongoDB
    # =====================================================

    MONGODB_URI: str = os.getenv(
        "MONGODB_URI",
        "mongodb://localhost:27017"
    )

    DATABASE_NAME: str = "traffic_management_db"

    # =====================================================
    # Vision & Detection
    # =====================================================

    # Normal vehicle detection confidence threshold
    CONFIDENCE_THRESHOLD: float = 0.20

    # Ambulance detection confidence threshold.
    # Keep this lower because the custom model may produce
    # valid ambulance detections below 0.40 confidence.
    AMBULANCE_CONFIDENCE_THRESHOLD: float = 0.36

    # Non-Maximum Suppression IoU threshold
    IOU_THRESHOLD: float = 0.45

    # YOLO inference image size (640 is optimal for high FPS without loss of accuracy)
    INFERENCE_IMAGE_SIZE: int = 640

    # =====================================================
    # Normal Vehicle Classes
    # =====================================================

    # COCO class IDs
    TARGET_CLASSES: list[int] = [
        2,  # car
        3,  # motorcycle
        5,  # bus
        7,  # truck
    ]

    CLASS_NAMES: dict[int, str] = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    # =====================================================
    # Ambulance Class
    # =====================================================

    # Application-level ambulance class ID.
    # The custom YOLO model may internally use class 0,
    # but the application maps ambulance to class 4.
    AMBULANCE_CLASS_ID: int = 4
    AMBULANCE_CLASS_NAME: str = "ambulance"

    # =====================================================
    # Detection ROI
    # =====================================================

    # IMPORTANT:
    #
    # Default ROI is None, which means FULL FRAME.
    #
    # Do NOT put a fixed pixel ROI here because different
    # uploaded videos can have different resolutions.
    #
    # Each video/camera can set its own ROI independently
    # using CameraConfig or VehicleTracker.set_roi().
    #
    # Supported ROI format:
    #
    # Pixel coordinates:
    # [x1, y1, x2, y2]
    # Example:
    # [100, 200, 700, 900]
    #
    # Normalized coordinates:
    # [x1, y1, x2, y2]
    # where every value is between 0.0 and 1.0.
    # Example:
    # [0.10, 0.20, 0.90, 0.80]
    #
    # None = detect on complete video frame.
    DETECTION_ROI: list[float] | None = None

    # =====================================================
    # Counting Lines
    # =====================================================

    # These are normalized coordinates relative to the
    # current video's frame dimensions.
    DEFAULT_COUNTING_LINES: dict = {
        "NORTH": {
            "p1": [0.1, 0.65],
            "p2": [0.9, 0.65],
            "orientation": "horizontal"
        },
        "SOUTH": {
            "p1": [0.1, 0.35],
            "p2": [0.9, 0.35],
            "orientation": "horizontal"
        },
        "EAST": {
            "p1": [0.35, 0.1],
            "p2": [0.35, 0.9],
            "orientation": "vertical"
        },
        "WEST": {
            "p1": [0.65, 0.1],
            "p2": [0.65, 0.9],
            "orientation": "vertical"
        },
    }

    # =====================================================
    # Junction Movement Vectors
    # =====================================================

    DEFAULT_JUNCTION_VECTORS: dict = {
        "NORTH": [0.0, 1.0],
        "SOUTH": [0.0, -1.0],
        "EAST": [-1.0, 0.0],
        "WEST": [1.0, 0.0],
    }

    # =====================================================
    # Directional Movement / Parking
    # =====================================================

    # Vehicle must remain stationary for this duration
    # before being considered parked.
    PARKED_DURATION_SECONDS: float = 300.0

    # Pixel movement below this value is considered
    # stationary.
    MOVEMENT_SPEED_THRESHOLD: float = 2.5

    # Small displacement threshold used to reduce
    # detection/tracking noise.
    NOISE_DISPLACEMENT_THRESHOLD: float = 1.2

    # Minimum trajectory points before determining
    # movement direction.
    MIN_TRAJECTORY_POINTS: int = 3

    # Distance from frame edge used for direction logic.
    EDGE_MARGIN_PIXELS: float = 25.0

    # Minimum displacement required before allowing a
    # vehicle's established direction to flip.
    DIRECTION_FLIP_MIN_DISPLACEMENT: float = 10.0

    # =====================================================
    # Traffic Level Thresholds
    # =====================================================

    THRESHOLD_LOW: int = 4
    THRESHOLD_MEDIUM: int = 9
    THRESHOLD_HIGH: int = 16

    # =====================================================
    # Queue Detection
    # =====================================================

    # Vehicles moving slower than this threshold are
    # considered potentially stationary/queued.
    QUEUE_SPEED_THRESHOLD: float = 2.5


# =========================================================
# Settings Instance
# =========================================================

settings = Settings()


# =========================================================
# Ensure Required Directories Exist
# =========================================================

settings.UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

settings.ANNOTATED_DIR.mkdir(
    parents=True,
    exist_ok=True
)