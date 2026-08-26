# Intelligent Traffic Management and Road Safety System
## Phase 1 — Real-Time Multi-Approach Traffic Monitoring Foundation

A college-level SIH prototype implementing approach-specific traffic monitoring with **YOLOv8n** vehicle detection, **ByteTrack** tracking, per-approach analytics, 4-way junction aggregation, historical observations, camera calibration, **MongoDB** persistence, and a **React** dashboard.

---

## Key Features through Phase 2

1. **Approach-Specific Architecture**:
   - Each uploaded video represents a dedicated camera feed for one road approach (**NORTH**, **SOUTH**, **EAST**, or **WEST**).
   - No quadrant splitting of a single video feed.
2. **YOLOv8n Vehicle Detection**:
   - Detects `car`, `motorcycle`, `bus`, and `truck` classes using a lightweight neural network.
3. **ByteTrack Vehicle Tracking**:
   - Maintains persistent track IDs across frames to prevent duplicate counting.
4. **Traffic Analytics Engine**:
   - **Current Vehicle Count**: Active vehicles in current frame/scene.
   - **Class Breakdown**: Counts for cars, bikes, buses, trucks.
   - **Traffic Flow**: Cumulative count of vehicles crossing the virtual counting line.
   - **Estimated Queue Length**: Approximate count of slow-moving/stationary vehicles.
   - **Traffic Density**: Area occupancy and spatial concentration index (0.0 to 1.0).
   - **Traffic Level**: Categorized as `LOW`, `MEDIUM`, `HIGH`, or `VERY HIGH`.
5. **Configurable Counting Lines**:
   - Independent counting line geometry for each directional approach (North, South, East, West) to match varying camera viewpoints.
6. **Junction Traffic State Aggregation**:
   - Unified 4-approach state matrix representing the entire intersection condition (ready for downstream adaptive signal controllers in future phases).
7. **MongoDB Storage**:
   - Persists granular traffic observations and junction configurations.
8. **React Monitoring Dashboard**:
   - Select or create junctions (e.g., `J-01`).
   - Upload traffic videos for specific directional approaches.
   - Real-time progress indicators.
   - Replay annotated videos with bounding boxes, track IDs, counting line, and HUD overlay.
   - View approach statistics and intersection-wide aggregated metrics.
9. **Historical Analytics (Phase 2)**:
   - Stores each completed-video observation and exposes per-junction or per-approach history.
   - Dashboard summary shows average/peak vehicle count, density, queue, flow, and a recent-observation chart.
10. **Camera Calibration (Phase 2)**:
   - Configure a normalized virtual counting line for each junction approach from the dashboard.
   - Saved calibration is used the next time a video is processed for that approach.

---

## Project Structure

```
traffic_management/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Configurable thresholds, paths, counting lines
│   ├── api/
│   │   ├── routes_junction.py   # Junction CRUD & aggregated state
│   │   ├── routes_video.py      # Video upload, processing & streaming
│   │   └── routes_analytics.py  # Traffic observations & queries
│   ├── core/
│   │   ├── vision/
│   │   │   ├── detector.py      # YOLOv8n vehicle detector
│   │   │   ├── tracker.py       # ByteTrack vehicle tracker
│   │   │   └── video_processor.py # Video inference, HUD overlay, annotation
│   │   └── analytics/
│   │       ├── traffic_metrics.py # Queue estimation, density, flow rate
│   │       └── junction_aggregator.py # 4-way junction state aggregation
│   ├── db/
│   │   ├── mongo_client.py      # MongoDB connection manager
│   │   └── repositories/        # Observation and junction repositories
│   ├── models/
│   │   └── traffic_schemas.py   # Pydantic schemas
│   └── tests/                   # Pytest test suite
├── frontend/                    # React + Vite dashboard
│   └── src/
│       ├── components/          # JunctionSelector, VideoUploader, ApproachFeedCard, JunctionOverview
│       ├── services/api.js      # REST API client
│       └── App.jsx
├── data/
│   ├── uploads/                 # Input traffic videos
│   └── annotated/               # Annotated output videos
├── scripts/
│   ├── verify_pipeline.py       # End-to-end verification script
│   └── download_sample_video.py # Sample video utility
└── requirements.txt             # Python dependencies
```

---

## How to Run Locally

### 1. Start the Backend

```bash
# In project root: traffic_management
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation: `http://localhost:8000/docs`

### 2. Start the Frontend Dashboard

```bash
# In frontend directory
cd frontend
npm install
npm run dev
```
Dashboard UI: `http://localhost:5173`

## Phase 2 API additions

- `PUT /api/junctions/{junction_id}/counting-lines` saves calibrated counting lines. Coordinates are normalized from `0` to `1`.
- `GET /api/analytics/junction/{junction_id}/history?approach=NORTH&limit=50` returns latest-first completed-video observations.
- `GET /api/analytics/junction/{junction_id}/summary?approach=NORTH` returns aggregate operational metrics.

## Scope and safety

This application processes uploaded, recorded traffic video. It does not yet connect to live cameras or operate physical traffic lights. Any future signal-control integration should use a simulator, manual override, fail-safe state, authorization, and a jurisdiction-approved controller interface.

---

## Running Tests

```bash
# Run unit & API test suite
python -m pytest backend/tests -v

# Run full end-to-end video pipeline verification
python scripts/verify_pipeline.py
```
