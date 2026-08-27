# DecisionBackend: Normal Traffic Signal Decision-Making Backend

## 1. Purpose

`decisionbackend` is a modular, deterministic traffic signal decision-making engine designed for a **4-approach isolated junction** (`NORTH`, `SOUTH`, `EAST`, `WEST`).

The controller independently evaluates real-time traffic conditions on all four approaches at each decision tick and dynamically allocates the green signal to the approach with the highest demand while strictly guaranteeing single-green invariants, minimum/maximum green bounds, gap-out early termination, and yellow/all-red clearance intervals.

---

## 2. Architecture & Separation of Responsibilities

```
  CCTV Cameras (North, South, East, West)
               │
               ▼
  Vision & Detection Layer (YOLOv8 + ByteTrack)
               │
               ▼
  Traffic Metrics Ingestion (Queue PCU, Wait Time, Flow Rate)
               │
               ▼
  ┌────────────────────────────────────────────────────────┐
  │                 decisionbackend                        │
  │                                                        │
  │  1. PCU Conversion (pcu.py)                           │
  │  2. Metric Normalization (traffic_metrics.py)         │
  │  3. Priority Formula P(d) (priority.py)                │
  │  4. Deterministic State Machine (signal_controller.py) │
  │  5. Phase & Light State Manager (signal_state.py)      │
  └────────────────────────────────────────────────────────┘
               │
               ▼
  Signal Decisions (Light States, Next Green, Timings)
               │
               ▼
  Frontend / Signal Controllers / Hardware Interface
```

`decisionbackend` contains **zero computer vision, database, or UI code**. It purely acts as a mathematical, state-machine decision processor receiving traffic metrics and producing signal state commands.

---

## 3. Indian Field-Measured PCU Values

Vehicle counts are converted into Passenger Car Units (PCU) using field-measured Indian coefficients defined in `junction_config.py`:

$$\text{Queue\_PCU}(d) = \sum (\text{vehicle\_count}[\text{class}] \times \text{PCU}[\text{class}])$$

| Vehicle Category | PCU Factor |
| :--- | :--- |
| **Two-Wheeler** (`two_wheeler`, `bike`, `motorcycle`) | **0.13** |
| **Car** (`car`, `taxi`, `van`) | **1.00** |
| **Auto-Rickshaw** (`auto_rickshaw`, `auto`) | **0.75** |
| **Bus** (`bus`) | **5.40** |
| **Truck** (`truck`, `lorry`) | **3.70** |

*Example:* 10 Two-Wheelers ($10 \times 0.13 = 1.3$) + 8 Cars ($8 \times 1.0 = 8.0$) + 2 Buses ($2 \times 5.4 = 10.8$) = **20.1 PCU**.

---

## 4. Traffic Inputs & Normalization

At every decision tick, each approach $d \in \{\text{NORTH}, \text{SOUTH}, \text{EAST}, \text{WEST}\}$ provides three raw metrics:
1. **$\text{Queue\_PCU}(d)$:** Total PCU of stationary/queued vehicles waiting at the stop line.
2. **$\text{WaitTime}(d)$:** Continuous seconds elapsed since approach $d$ last held GREEN.
3. **$\text{FlowRate}(d)$:** PCU-weighted flow crossing the counting line in PCU/min (rolling 30s window).

Raw metrics are normalized into unitless values bounded in $[0.0, 1.0]$:

$$\text{Queue\_norm} = \min\left(\frac{\text{Queue\_PCU}}{\text{QUEUE\_PCU\_MAX}}, 1.0\right) \quad (\text{Default max} = 40.0\text{ PCU})$$

$$\text{WaitTime\_norm} = \min\left(\frac{\text{WaitTime}}{\text{WAIT\_TIME\_REF}}, 1.0\right) \quad (\text{Default ref} = 90.0\text{ s})$$

$$\text{FlowRate\_norm} = \min\left(\frac{\text{FlowRate}}{\text{FLOW\_RATE\_MAX}}, 1.0\right) \quad (\text{Default max} = 12.0\text{ PCU/min})$$

---

## 5. Priority Formula $P(d)$

For each approach $d$, the priority score $P(d)$ is calculated using fixed weights:

$$P(d) = 0.45 \times \text{Queue\_norm} + 0.35 \times \text{WaitTime\_norm} + 0.20 \times \text{FlowRate\_norm}$$

- **Range:** $0.0 \le P(d) \le 1.0$
- Priority weights remain constant across all phases.

---

## 6. Signal States & Single-Green Invariant

Supported light colors for each approach:
- `RED`
- `YELLOW`
- `GREEN`

Supported junction phase states:
- `GREEN`: Exactly ONE approach is GREEN; all three other approaches are RED.
- `YELLOW`: Active approach is YELLOW; all three other approaches are RED.
- `ALL_RED`: All four approaches are RED simultaneously for intersection clearance.

**Strict Safety Rule:** Never allow more than one approach to be GREEN simultaneously.

---

## 7. Configurable Signal Timing & Decision Rules

Defined in `SignalTimingConfig` (`junction_config.py`):

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `g_min` | **10.0 s** | Minimum green duration. Green phase cannot be interrupted before $G_{\min}$. |
| `g_max` | **45.0 s** | Maximum green duration. Green phase is forcefully terminated at $G_{\max}$. |
| `yellow_time` | **3.0 s** | Duration of yellow change interval. |
| `all_red_time` | **2.0 s** | Duration of all-red clearance interval before next green. |
| `gap_out_time` | **3.5 s** | Inactivity headway threshold triggering early green termination. |
| `decision_interval` | **1.0 s** | Discrete decision tick interval. |
| `priority_switch_margin` | **0.15** | Delta by which rival RED approach priority must exceed active green priority to switch before $G_{\max}$. |
| `empty_queue_threshold_pcu` | **0.5 PCU** | Threshold below which an approach is considered empty. |

---

## 8. Decision Logic Flow

1. **Phase $< G_{\min}$:** Keep current approach GREEN.
2. **Phase $\ge G_{\min}$ (Empty Approach):** If current approach queue is empty ($\le 0.5$ PCU) and no incoming traffic:
   - Terminate green immediately $\rightarrow$ Select $\operatorname{argmax} P(d)$ among RED approaches.
3. **Phase $\ge G_{\min}$ (Gap-Out):** If vehicle headway exceeds `gap_out_time`:
   - Terminate green $\rightarrow$ Select $\operatorname{argmax} P(d)$ among RED approaches.
4. **Phase $\ge G_{\min}$ (Priority Switch):** If a RED approach priority $P(\text{rival}) > P(\text{current}) + 0.15$:
   - Trigger phase switch to highest priority RED approach.
5. **Phase $\ge G_{\max}$:** Force switch to highest priority RED approach.
6. **Switch Execution:** $\text{GREEN} \xrightarrow{\text{yellow\_time}} \text{YELLOW} \xrightarrow{\text{all\_red\_time}} \text{ALL\_RED} \rightarrow \text{NEW GREEN}$.

---

## 9. Example Decision Output

```
 TICK #012 (Time:  12.0s) | Active Phase: GREEN   | Current Green: NORTH  | Phase Duration: 12.0s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NORTH  | Q=  0.0 PCU (norm=0.00) | W=  0.0s (norm=0.00) | F= 0.0 PCU/m (norm=0.00) | P=0.00 | [GREEN] 
  SOUTH  | Q=  8.0 PCU (norm=0.20) | W= 75.0s (norm=0.83) | F= 3.0 PCU/m (norm=0.25) | P=0.43 | [RED]   
  EAST   | Q= 32.0 PCU (norm=0.80) | W= 20.0s (norm=0.22) | F=10.0 PCU/m (norm=0.83) | P=0.61 | [RED]   
  WEST   | Q=  4.5 PCU (norm=0.11) | W= 10.0s (norm=0.11) | F= 2.0 PCU/m (norm=0.17) | P=0.12 | [RED]   
────────────────────────────────────────────────────────────────────────────────
 Decision Reason : EARLY TERMINATION: NORTH is effectively empty (Queue=0.0 PCU). Switching to EAST (P=0.61).
 Next Target     : EAST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 10. How to Run

### Run Unit Tests (12 Test Cases):
```bash
python -m unittest backend.decisionbackend.tests
```

### Run Deterministic Simulation Demo:
```bash
python -m backend.decisionbackend
```
or
```bash
python backend/decisionbackend/simulation.py
```
