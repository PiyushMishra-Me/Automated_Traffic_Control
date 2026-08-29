# Intelligent Traffic Management and Road Safety System
## Phase 3.1 — Live Geospatial Traffic Map, Upstream Diversion & Weather-Adaptive Control

A complete prototype implementing approach-specific traffic monitoring with **YOLOv8n** vehicle detection, **ByteTrack** tracking, per-approach analytics, 4-way junction aggregation, **Live Interactive Geospatial Map**, **Accident & Upstream Traffic Diversion Engine**, **Open-Meteo Weather-Adaptive Signal Control**, and a **React** dashboard.

---

## Key Features through Phase 3.1

1. **Approach-Specific Architecture**:
   - Each uploaded video represents a dedicated camera feed for one road approach (**NORTH**, **SOUTH**, **EAST**, or **WEST**).
   - Normalized coordinate system per camera perspective.
2. **YOLOv8 Vehicle Detection & ByteTrack**:
   - Detects `car`, `motorcycle`, `bus`, and `truck` classes.
   - Maintains persistent track IDs across frames.
3. **Traffic Analytics Engine**:
   - Vehicle counts, class breakdown, flow rates, queue estimation, spatial density index, and traffic level categorization (`LOW`, `MEDIUM`, `HIGH`, `VERY HIGH`).
4. **Live Urban Geospatial Map**:
   - Interactive vector map displaying interconnected junction nodes with live traffic status colors (Green, Amber, Red, Hazard Pulse).
   - Real-time road corridors with animated flow dashes, detour arcs, and interactive node tooltips with one-click junction selection.
5. **Accident & Incident Management with Upstream Traffic Diversions**:
   - Portal for reporting accidents, vehicle breakdowns, road blockages, and waterlogging.
   - **Automated Upstream Diversion Algorithm**: Computes alternate detour bypass routes (e.g., Northbound traffic diverted via East Arterial to J-02), cuts green time on blocked roads, and extends bypass green phases by $+15$s to $+25$s to clear bottlenecks.
6. **Weather Checking & Weather-Adaptive Signal Control**:
   - Integrates live meteorological telemetry from **Open-Meteo API** (temperature, rainfall, wind, visibility, road friction factor).
   - Dynamically extends yellow transition intervals ($+1.5$s) and all-red clearance intervals ($+2.0$s) during rain/fog to prevent skidding in dilemma zones.
   - Computes dynamic speed limit advisories (e.g. $35$ km/h on wet asphalt).
   - Includes on-demand scenario testing (Heavy Rain, Fog, Storm, Clear Skies).
7. **Adaptive Signal Simulator & Decision Backend**:
   - Rule-based multi-criteria priority engine (`G_MIN` safety, `G_MAX` enforcement, empty queue/gap-out termination, single green invariant).
   - Indian PCU weighting and continuous wait-time tracking.
8. **React Monitoring Dashboard**:
   - Live visual signal simulation, historical analytics trend charts, video upload & HUD replay, and virtual tripwire calibrator.

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
│   │   ├── routes_analytics.py  # Traffic observations & queries
│   │   ├── routes_incident.py   # Incident reporting & upstream diversion API
│   │   └── routes_weather.py    # Live weather telemetry & simulation override API
│   ├── core/
│   │   ├── vision/
│   │   │   ├── detector.py      # YOLOv8n vehicle detector
│   │   │   ├── tracker.py       # ByteTrack vehicle tracker
│   │   │   └── video_processor.py # Video inference, HUD overlay, annotation
│   │   ├── analytics/
│   │   │   ├── traffic_metrics.py # Queue estimation, density, flow rate
│   │   │   └── junction_aggregator.py # 4-way junction state aggregation
│   │   ├── control/
│   │   │   ├── adaptive_signal.py # Weather & incident aware adaptive controller
│   │   │   ├── diversion_engine.py# Upstream rerouting & signal compensation
│   │   │   └── signal_simulation.py # Deterministic time-stepped simulation
│   │   └── weather/
│   │       └── weather_service.py # Open-Meteo live weather client & road safety
│   ├── decisionbackend/         # 4-approach PCU decision engine & test suite
│   ├── db/
│   │   ├── mongo_client.py      # MongoDB connection manager
│   │   └── repositories/        # Observation, junction, & incident repositories
│   └── tests/                   # Pytest test suite
├── frontend/                    # React + Vite dashboard
│   └── src/
│       ├── components/
│       │   ├── LiveJunctionMap.jsx # Interactive live geospatial map
│       │   ├── WeatherWidget.jsx   # Live weather telemetry & road friction
│       │   ├── IncidentManager.jsx # Incident & diversion management hub
│       │   ├── IncidentReportingModal.jsx # Accident reporting dialog
│       │   ├── SignalSimulator.jsx # Adaptive signal simulator
│       │   └── ...
│       ├── services/api.js      # REST API client
│       └── App.jsx
└── requirements.txt             # Python dependencies
```

---

## How to Run Locally

### 1. Start the Backend

```bash
# In project root: Automated_Traffic_Control
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
API Docs: `http://localhost:8000/docs`

### 2. Start the Frontend Dashboard

```bash
# In frontend directory
cd frontend
npm install
npm run dev
```
Dashboard UI: `http://localhost:5173`

---

## Running Tests

```bash
# Run unit & API test suite
python -m pytest backend/tests -v

# Run decision engine test suite
python -m pytest backend/decisionbackend/tests.py -v
```
