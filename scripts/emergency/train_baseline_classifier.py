import os
import sys
import time
import json
import torch
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "processed"
RUNS_DIR = PROJECT_ROOT / "runs" / "emergency_classifier" / "baseline"
REPORTS_DIR = PROJECT_ROOT / "data" / "emergency_vehicle_dataset" / "reports"

def train_emergency_classifier():
    print("=" * 75)
    print("STEP 2: TRAINING ISOLATED BASELINE EMERGENCY CLASSIFIER")
    print("=" * 75)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Hardware Acceleration Device: {device} (CUDA Available: {torch.cuda.is_available()})")

    start_time = time.time()

    # Initialize YOLOv8 Nano classification model
    model = YOLO("yolov8n-cls.pt")

    train_config = {
        "data": str(DATASET_DIR),
        "epochs": 20,
        "imgsz": 128,
        "batch": 32,
        "device": device,
        "project": str(RUNS_DIR.parent),
        "name": "baseline",
        "exist_ok": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "patience": 8,
        "save": True,
        "verbose": True
    }

    print("\nTraining Configuration:")
    for k, v in train_config.items():
        print(f"  {k:<15}: {v}")

    # Start training
    results = model.train(**train_config)

    train_duration = time.time() - start_time
    print(f"\nTraining completed in {train_duration:.1f}s ({train_duration/60.0:.2f} mins)")

    best_checkpoint = RUNS_DIR / "weights" / "best.pt"
    last_checkpoint = RUNS_DIR / "weights" / "last.pt"

    print(f"Best Model Checkpoint: {best_checkpoint} (Exists: {best_checkpoint.exists()})")
    print(f"Last Model Checkpoint: {last_checkpoint} (Exists: {last_checkpoint.exists()})")

    # Save training metadata
    train_metadata = {
        "model_architecture": "YOLOv8n-cls",
        "input_size": 128,
        "epochs": 20,
        "batch_size": 32,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "device": device,
        "training_duration_seconds": train_duration,
        "best_checkpoint": str(best_checkpoint.relative_to(PROJECT_ROOT)) if best_checkpoint.exists() else None,
        "last_checkpoint": str(last_checkpoint.relative_to(PROJECT_ROOT)) if last_checkpoint.exists() else None
    }

    with open(REPORTS_DIR / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(train_metadata, f, indent=2)

    return train_metadata

if __name__ == "__main__":
    train_emergency_classifier()
