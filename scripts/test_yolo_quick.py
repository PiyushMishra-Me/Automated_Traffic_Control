import sys
import time
from pathlib import Path
import numpy as np

print("Step 1: Importing YOLO from ultralytics...", flush=True)
t0 = time.time()
from ultralytics import YOLO
print(f"Step 1 done in {time.time() - t0:.2f}s", flush=True)

print("Step 2: Loading model yolov8n.pt...", flush=True)
t0 = time.time()
model = YOLO("yolov8n.pt")
print(f"Step 2 done in {time.time() - t0:.2f}s", flush=True)

print("Step 3: Running test inference...", flush=True)
t0 = time.time()
img = np.zeros((360, 640, 3), dtype=np.uint8)
res = model.predict(img, conf=0.25, verbose=False)
print(f"Step 3 done in {time.time() - t0:.2f}s", flush=True)

print("Step 4: Running ByteTrack on blank frame...", flush=True)
t0 = time.time()
tracks = model.track(img, persist=True, tracker="bytetrack.yaml", verbose=False)
print(f"Step 4 done in {time.time() - t0:.2f}s", flush=True)

print("ALL STEPS COMPLETED SUCCESSFULLY!", flush=True)
