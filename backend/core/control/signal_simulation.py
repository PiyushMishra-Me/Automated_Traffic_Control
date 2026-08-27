"""Deterministic discrete-time traffic-signal simulation.

Unlike :class:`AdaptiveSignalController` (which returns a single instantaneous
recommendation), this steps a junction second-by-second: queues grow from
per-approach arrival rates and drain while their signal is green. It runs two
scenarios over the *same* arrivals — an adaptive controller that lengthens the
green for the busier axis, and a naive fixed-timer baseline — so the two can be
compared fairly. It is fully deterministic (no randomness), so results are
reproducible and testable. Nothing here touches physical hardware.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from backend.core.control.adaptive_signal import AdaptiveSignalController
from backend.models.traffic_schemas import (
    ApproachEnum,
    ApproachSimSummary,
    JunctionTrafficState,
    SignalPhaseEnum,
    SignalSimulationResult,
    SimulationComparison,
    SimulationStep,
)

# Signal timing (seconds)
YELLOW_SECONDS = 4
ALL_RED_SECONDS = 2
FIXED_GREEN = 30          # baseline: same green every cycle, no adaptation
MIN_GREEN = 10            # adaptive green is clamped to this range for watchability
MAX_GREEN = 40

# Traffic-flow model
SATURATION_FLOW = 0.5     # vehicles discharged per second per approach while green
GREEN_PER_VEHICLE = 2.0   # adaptive: seconds of green added per queued vehicle

APPROACHES = ["NORTH", "SOUTH", "EAST", "WEST"]
NS_AXIS = ["NORTH", "SOUTH"]
EW_AXIS = ["EAST", "WEST"]

# Illustrative demand used only when a junction has no processed observations,
# so the button always produces a visible run.
DEMO_QUEUES = {"NORTH": 8, "SOUTH": 6, "EAST": 3, "WEST": 2}
DEMO_RATES = {"NORTH": 0.25, "SOUTH": 0.2, "EAST": 0.12, "WEST": 0.1}


@dataclass
class _ScenarioResult:
    steps: List[SimulationStep] = field(default_factory=list)
    arrivals: Dict[str, float] = field(default_factory=dict)
    served: Dict[str, float] = field(default_factory=dict)
    max_queue: Dict[str, float] = field(default_factory=dict)
    final_queue: Dict[str, float] = field(default_factory=dict)
    wait: Dict[str, float] = field(default_factory=dict)  # veh-seconds per approach


def _lights(active_axis: List[str], color: str) -> Dict[str, str]:
    """All approaches RED except those on ``active_axis``, which show ``color``."""
    return {a: (color if a in active_axis else "RED") for a in APPROACHES}


def _adaptive_green(queues: Dict[str, float], axis: List[str]) -> int:
    demand = queues[axis[0]] + queues[axis[1]]
    green = MIN_GREEN + demand * GREEN_PER_VEHICLE
    return int(max(MIN_GREEN, min(MAX_GREEN, round(green))))


class TrafficSimulator:
    """Steps a junction through repeated signal cycles for a fixed horizon."""

    @staticmethod
    def _extract(junction_state: JunctionTrafficState):
        """Build initial queues + arrival rates from the latest observations."""
        mapping = {
            "NORTH": junction_state.north,
            "SOUTH": junction_state.south,
            "EAST": junction_state.east,
            "WEST": junction_state.west,
        }
        queues: Dict[str, int] = {}
        rates: Dict[str, float] = {}
        any_data = False
        for approach, state in mapping.items():
            if state is not None:
                any_data = True
                queues[approach] = max(state.estimated_queue_length, state.vehicle_count)
                rates[approach] = round(0.05 + state.density * 0.35, 4)
            else:
                queues[approach] = 0
                rates[approach] = 0.05
        return queues, rates, any_data

    @staticmethod
    def _run_scenario(
        initial_queues: Dict[str, int],
        arrival_rates: Dict[str, float],
        horizon: int,
        adaptive: bool,
    ) -> _ScenarioResult:
        queue = {a: float(initial_queues[a]) for a in APPROACHES}
        arrivals = {a: float(initial_queues[a]) for a in APPROACHES}
        served = {a: 0.0 for a in APPROACHES}
        wait = {a: 0.0 for a in APPROACHES}
        max_queue = {a: float(initial_queues[a]) for a in APPROACHES}
        steps: List[SimulationStep] = []
        served_total = 0.0
        t = 0

        while t < horizon:
            green_ns = _adaptive_green(queue, NS_AXIS) if adaptive else FIXED_GREEN
            green_ew = _adaptive_green(queue, EW_AXIS) if adaptive else FIXED_GREEN

            # (label, phase enum, duration, lights) for one full cycle
            segments = [
                ("NORTH/SOUTH GREEN", SignalPhaseEnum.NORTH_SOUTH_GREEN, green_ns, _lights(NS_AXIS, "GREEN")),
                ("NORTH/SOUTH YELLOW", SignalPhaseEnum.NORTH_SOUTH_GREEN, YELLOW_SECONDS, _lights(NS_AXIS, "YELLOW")),
                ("ALL RED", SignalPhaseEnum.ALL_RED, ALL_RED_SECONDS, _lights([], "RED")),
                ("EAST/WEST GREEN", SignalPhaseEnum.EAST_WEST_GREEN, green_ew, _lights(EW_AXIS, "GREEN")),
                ("EAST/WEST YELLOW", SignalPhaseEnum.EAST_WEST_GREEN, YELLOW_SECONDS, _lights(EW_AXIS, "YELLOW")),
                ("ALL RED", SignalPhaseEnum.ALL_RED, ALL_RED_SECONDS, _lights([], "RED")),
            ]

            for label, phase, duration, lights in segments:
                green_set = [a for a in APPROACHES if lights[a] == "GREEN"]
                for second in range(duration):
                    if t >= horizon:
                        return _ScenarioResult(steps, arrivals, served, max_queue, queue, wait)

                    # 1. Vehicles arrive on every approach.
                    for a in APPROACHES:
                        queue[a] += arrival_rates[a]
                        arrivals[a] += arrival_rates[a]

                    # 2. Green approaches discharge at the saturation flow rate.
                    for a in green_set:
                        discharged = min(queue[a], SATURATION_FLOW)
                        queue[a] -= discharged
                        served[a] += discharged
                        served_total += discharged

                    # 3. Everything still queued waits one more second.
                    for a in APPROACHES:
                        wait[a] += queue[a]
                        if queue[a] > max_queue[a]:
                            max_queue[a] = queue[a]

                    steps.append(
                        SimulationStep(
                            t=t,
                            phase=phase,
                            phase_label=label,
                            phase_time_remaining=duration - second - 1,
                            lights=dict(lights),
                            queues={a: int(round(queue[a])) for a in APPROACHES},
                            served_total=int(round(served_total)),
                        )
                    )
                    t += 1

        return _ScenarioResult(steps, arrivals, served, max_queue, queue, wait)

    @classmethod
    def run(cls, junction_state: JunctionTrafficState, horizon: int = 180) -> SignalSimulationResult:
        initial_queues, arrival_rates, any_data = cls._extract(junction_state)

        seeded_demo = False
        if not any_data:
            initial_queues = dict(DEMO_QUEUES)
            arrival_rates = dict(DEMO_RATES)
            seeded_demo = True

        adaptive = cls._run_scenario(initial_queues, arrival_rates, horizon, adaptive=True)
        fixed = cls._run_scenario(initial_queues, arrival_rates, horizon, adaptive=False)

        per_approach: List[ApproachSimSummary] = []
        for a in APPROACHES:
            arr = adaptive.arrivals[a]
            per_approach.append(
                ApproachSimSummary(
                    approach=ApproachEnum(a),
                    arrivals=int(round(arr)),
                    served=int(round(adaptive.served[a])),
                    max_queue=int(round(adaptive.max_queue[a])),
                    final_queue=int(round(adaptive.final_queue[a])),
                    avg_wait=round(adaptive.wait[a] / arr, 1) if arr > 0 else 0.0,
                )
            )

        adaptive_arrivals = sum(adaptive.arrivals.values())
        fixed_arrivals = sum(fixed.arrivals.values())
        adaptive_avg_wait = round(sum(adaptive.wait.values()) / adaptive_arrivals, 1) if adaptive_arrivals else 0.0
        fixed_avg_wait = round(sum(fixed.wait.values()) / fixed_arrivals, 1) if fixed_arrivals else 0.0
        improvement_pct = round((fixed_avg_wait - adaptive_avg_wait) / fixed_avg_wait * 100, 1) if fixed_avg_wait > 0 else 0.0

        comparison = SimulationComparison(
            adaptive_avg_wait=adaptive_avg_wait,
            fixed_avg_wait=fixed_avg_wait,
            adaptive_served=int(round(sum(adaptive.served.values()))),
            fixed_served=int(round(sum(fixed.served.values()))),
            improvement_pct=improvement_pct,
        )

        recommendation = AdaptiveSignalController.recommend(junction_state)

        if improvement_pct >= 0:
            verdict = f"a {improvement_pct}% reduction in average wait"
        else:
            verdict = f"a {abs(improvement_pct)}% increase in average wait"
        rationale = (
            f"Simulated {horizon}s of traffic. Adaptive control cleared "
            f"{comparison.adaptive_served} vehicles at an average wait of "
            f"{adaptive_avg_wait}s, versus {fixed_avg_wait}s under a fixed {FIXED_GREEN}s "
            f"timer — {verdict}."
        )
        if seeded_demo:
            rationale = "No processed observations yet — running an illustrative demo scenario. " + rationale

        return SignalSimulationResult(
            junction_id=junction_state.junction_id,
            total_seconds=horizon,
            steps=adaptive.steps,
            per_approach=per_approach,
            comparison=comparison,
            recommendation=recommendation,
            rationale=rationale,
            seeded_demo=seeded_demo,
        )
