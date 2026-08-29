# 4-Approach Traffic Signal Decision Engine

A deterministic, multi-criteria adaptive traffic signal controller designed for 4-way intersections.

## Core Modules
- **`models.py`**: Enums (`Approach`, `SignalColor`, `PhaseState`) and dataclasses (`DirectionTraffic`, `SignalDecision`, `PriorityScore`).
- **`junction_config.py`**: Field-measured PCU weights, normalization maximums, priority weights, and timing parameters (`G_MIN`, `G_MAX`, `YELLOW`, `ALL_RED`, `GAP_OUT`).
- **`pcu.py`**: Computes dynamic Passenger Car Units (PCU) from vehicle class counts.
- **`traffic_metrics.py`**: Normalizes Queue PCU, Wait Time, and Flow Rate into $[0.0, 1.0]$.
- **`priority.py`**: Computes multi-factor priority score:
  $$P(d) = 0.45 \times Q_{\text{norm}} + 0.35 \times W_{\text{norm}} + 0.20 \times F_{\text{norm}}$$
- **`signal_state.py`**: Maintains persistent junction state, active approach, and continuous approach wait-times.
- **`signal_controller.py`**: Executes deterministic state machine ensuring `G_MIN` safety, `G_MAX` enforcement, empty queue & gap-out termination, priority preemption, and yellow/all-red clearance intervals.
- **`decision_engine.py`**: High-level facade for seamless integration with vision and upstream API services.
- **`simulation.py`**: Deterministic simulation scenario runner.
- **`tests.py`**: Comprehensive 12-test validation suite.

## Running Tests & Simulation
```bash
# Run unit tests
python -m pytest backend/decisionbackend/tests.py -v

# Run deterministic simulation demo
python -m backend.decisionbackend
```
