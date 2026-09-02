# Automated Traffic Control & Intelligent Road Safety System

An end-to-end intelligent traffic control and safety automation platform featuring **Dual-Model Real-Time Vision AI** (YOLO11 / YOLOv8 + Custom Ambulance Detector), **Continuous CCTV / RTSP Stream Analytics**, **Interactive Real-World Geospatial Map (Leaflet / OpenStreetMap)** across major metropolitan cities, **Dynamic Shortest Path Citizen Navigation**, **Open-Meteo Weather-Adaptive Signal Timing**, **Automated Upstream Accident Diversion**, and **Multi-Junction Green Wave Emergency Preemption**.

---

## 🌟 Core System Capabilities

### 1. Dual-Model Real-Time AI Vision & Live Stream Ingestion
- **Dual-Model Inference Pipeline**:
  - **YOLO11** for standard multi-class vehicle detection (`car`, `motorcycle`, `bus`, `truck`).
  - **Custom Ambulance Neural Network** trained for high-confidence emergency vehicle detection with temporal confirmation.
- **Continuous Live CCTV / RTSP Stream Ingestion**:
  - `LiveStreamWorker` background threads consume live RTSP feeds, IP webcams (e.g. Android IP Webcam), HTTP MJPEG, and device webcams.
  - Real-time vehicle counting, density estimation, speed calculation, and queue tracking.
  - Live annotated MJPEG streaming endpoint: `GET /api/videos/live/{junction_id}/{approach}/annotated-stream`.
- **4-Way Multi-Approach Ingest**:
  - Dedicated approach cameras for **NORTH**, **SOUTH**, **EAST**, and **WEST** with virtual tripwire counting lines.

### 2. Interactive Real-World Geospatial Map & Multi-City Grids
- **Real-World OpenStreetMap (OSM) / CartoDB Integration**:
  - Interactive Leaflet map projecting actual GPS coordinates (`latitude, longitude`) onto real street maps.
- **Metropolitan City Presets**:
  - 🏛️ **New Delhi / NCR**: Connaught Place, ITO, Civil Lines, AIIMS & Dhaula Kuan.
  - 🌊 **Mumbai Metropolitan**: BKC Financial Core, Dadar TT, Marine Drive, Andheri WEH & Vashi Toll.
  - 💎 **Hyderabad Cyber Hub**: Hitec City Cyber Towers, Gachibowli ORR, Jubilee Hills, Begumpet & Charminar.
  - 🌳 **Bengaluru Tech Grid**: Silk Board, Electronic City, Koramangala, Indiranagar & MG Road.
- **Interactive Map HUD**:
  - Live color-coded congestion markers, ambulance tracking halos, active accident hazard icons, and click-to-route shortcuts.

### 3. Dynamic Shortest Path Citizen Navigation Engine
- **Real-Time Traffic-Weighted Pathfinder**:
  - Computes optimal shortest route based on physical GPS distance, live vision AI queue lengths, and weather road friction.
- **Dynamic Incident & Roadblock Avoidance**:
  - Automatically identifies blocked corridors and reroutes commuters through clear bypass arterials.
- **Turn-by-Turn Guidance**:
  - Provides step-by-step instructions, estimated transit time, and distance breakdown.

### 4. Weather-Adaptive Signal Control
- **Live Open-Meteo Satellite API Telemetry**:
  - Real-time weather ingestion (precipitation, humidity, visibility, wind speed, surface friction factor).
- **Safety Interval Adjustments**:
  - Automatically extends yellow intervals ($+1.5$s) and all-red clearance intervals ($+2.0$s) during rain or fog to prevent skidding in dilemma zones.
- **Speed Limit Advisories**:
  - Calculates dynamic speed limit recommendations based on road surface traction.

### 5. Multi-Junction Green Wave & Emergency Preemption
- **Cascading Interconnected Preemption**:
  - When an ambulance is detected on any approach, the local junction immediately locks green.
  - Automatically pre-notifies the downstream interconnected junction along the corridor to flush standing queues 45 seconds ahead of arrival.
- **Safe Clearance Transitions**:
  - Applies strict $G_{min}$ safety checks, Yellow (3s), and All-Red (2s) safety transitions before emergency green activation.

### 6. Role-Based Portal Gateway
- **🏛️ Traffic Police & Government Command Portal**: Full junction telemetry, live CCTV feeds, manual signal overrides, incident management.
- **🚑 Emergency Hospital & Dispatch Portal**: Active mission tracking, corridor visualization, emergency preemption triggers.
- **👥 Public Citizen & Commuter Portal**: Dynamic shortest path navigation, weather road safety advisories, instant accident reporting.

---

## 🏗️ Project Architecture

```
Automated_Traffic_Control/
├── backend/
│   ├── main.py                  # FastAPI application entry point
│   ├── config.py                # System settings and model paths
│   ├── api/
│   │   ├── routes_junction.py   # Junction CRUD, multi-city filtering, aggregated state
│   │   ├── routes_video.py      # Video processing & real-time live MJPEG stream
│   │   ├── routes_navigation.py # Dynamic shortest path navigation API
│   │   ├── routes_ambulance.py  # Emergency dispatch & mission management
│   │   ├── routes_incident.py   # Incident reporting & upstream diversion API
│   │   ├── routes_weather.py    # Live weather telemetry API
│   │   └── routes_auth.py       # Role authentication
│   ├── core/
│   │   ├── vision/
│   │   │   ├── detector.py      # YOLOv8 detector
│   │   │   ├── tracker.py       # Dual-model tracker (YOLO + Custom Ambulance)
│   │   │   ├── video_processor.py # Video inference and annotation
│   │   │   ├── live_stream_manager.py # Real-time live camera thread worker
│   │   │   └── emergency_bridge.py # Vision-to-Emergency bridge
│   │   ├── control/
│   │   │   ├── emergency_orchestrator.py # Multi-junction green wave coordinator
│   │   │   ├── adaptive_signal.py # Weather & incident aware adaptive controller
│   │   │   ├── navigation_engine.py # Real-world Dijkstra shortest pathfinder
│   │   │   ├── diversion_engine.py# Upstream rerouting compensation
│   │   │   └── signal_simulation.py # Time-stepped adaptive simulation
│   │   └── weather/
│   │       └── weather_service.py # Open-Meteo live weather client
│   ├── db/
│   │   ├── mongo_client.py      # MongoDB client with in-memory fallback
│   │   └── repositories/        # Junction, traffic, incident, ambulance repos
│   └── tests/                   # Pytest test suite (49 test cases)
├── frontend/                    # React + Vite Dashboard
│   └── src/
│       ├── components/
│       │   ├── LiveJunctionMap.jsx # Interactive Leaflet real-world map
│       │   ├── PublicCitizenPortal.jsx # Dynamic shortest path & commuter hub
│       │   ├── GovernmentCommandPortal.jsx # Police command & live feeds
│       │   ├── AmbulanceDispatchPortal.jsx # Emergency dispatch portal
│       │   ├── ApproachFeedCard.jsx # Real-time AI annotated MJPEG stream player
│       │   ├── WeatherWidget.jsx   # Live weather telemetry & road friction
│       │   └── RoleAuthHeader.jsx  # Single-portal banner & gateway switcher
│       ├── services/api.js      # REST API client
│       └── App.jsx
└── requirements.txt             # Python dependencies
```

---

## 🚀 How to Run Locally

### 1. Start the Backend API Server

```bash
# In project root
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- **API Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Root API Status**: `http://localhost:8000/`

### 2. Start the Frontend Dashboard

```bash
# In frontend directory
cd frontend
npm install
npm run dev
```
- **Web Application UI**: `http://localhost:5173`

---

## 🧪 Running Tests

```bash
# Run backend test suite (49 test suites)
$env:PYTHONPATH='.'; pytest backend/tests -v
```

