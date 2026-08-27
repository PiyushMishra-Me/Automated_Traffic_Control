"""
simulation.py
Deterministic demonstration and simulation runner for the 4-approach traffic signal controller.
Runs scenarios demonstrating:
- G_MIN protection
- Priority-based switching
- Empty approach early termination
- Gap-out early termination
- G_MAX forced switching
- Yellow and All-Red clearance transitions
"""

import sys
from typing import Dict
from pathlib import Path

from backend.decisionbackend.models import (
    Approach,
    SignalColor,
    PhaseState,
    DirectionTraffic,
    SignalDecision,
)
from backend.decisionbackend.junction_config import JunctionConfig, SignalTimingConfig
from backend.decisionbackend.signal_controller import SignalController


def format_approach_row(app: Approach, traffic: DirectionTraffic, dec: SignalDecision) -> str:
    p_obj = dec.priority_scores[app]
    color = dec.signal_states[app]
    color_tag = f"[{color.value}]"

    return (
        f"  {app.value:<6} | Q={traffic.queue_pcu:5.1f} PCU (norm={p_obj.queue_norm:.2f}) | "
        f"W={traffic.wait_time:5.1f}s (norm={p_obj.wait_norm:.2f}) | "
        f"F={traffic.flow_rate:4.1f} PCU/m (norm={p_obj.flow_norm:.2f}) | "
        f"P={p_obj.score:4.2f} | {color_tag:<8}"
    )


def print_decision_block(step_num: int, sim_time: float, traffic: Dict[Approach, DirectionTraffic], dec: SignalDecision):
    print("\n" + "=" * 80)
    print(f" TICK #{step_num:03d} (Time: {sim_time:5.1f}s) | Active Phase: {dec.active_phase.value:<7} | "
          f"Current Green: {dec.current_green.value if dec.current_green else 'NONE':<6} | "
          f"Phase Duration: {dec.phase_duration:4.1f}s")
    print("=" * 80)
    
    for app in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]:
        print(format_approach_row(app, traffic[app], dec))
        
    print("-" * 80)
    print(f" Decision Reason : {dec.reason}")
    if dec.is_switch_in_progress:
        print(f" Next Target     : {dec.next_green_candidate.value if dec.next_green_candidate else 'None'}")
    print("=" * 80)


def run_deterministic_simulation():
    """
    Simulates a 40-second timeline covering key transitions.
    """
    config = JunctionConfig(
        timing=SignalTimingConfig(
            g_min=10.0,
            g_max=30.0,
            yellow_time=3.0,
            all_red_time=2.0,
            gap_out_time=3.0,
            decision_interval=1.0,
            empty_queue_threshold_pcu=0.5
        )
    )

    controller = SignalController(config=config, initial_green=Approach.NORTH)

    print("=" * 80)
    print(" 4-APPROACH TRAFFIC SIGNAL DECISION ENGINE - SIMULATION RUNNER")
    print("=" * 80)

    # Initial traffic state
    traffic = {
        Approach.NORTH: DirectionTraffic(
            direction=Approach.NORTH,
            vehicle_counts={"car": 8, "bus": 1},
            queue_pcu=13.4,
            wait_time=0.0,
            flow_rate=8.0,
            time_since_last_vehicle_passed=0.5
        ),
        Approach.SOUTH: DirectionTraffic(
            direction=Approach.SOUTH,
            vehicle_counts={"two_wheeler": 12, "car": 4},
            queue_pcu=5.56,
            wait_time=15.0,
            flow_rate=4.0,
            time_since_last_vehicle_passed=2.0
        ),
        Approach.EAST: DirectionTraffic(
            direction=Approach.EAST,
            vehicle_counts={"car": 15, "truck": 2},
            queue_pcu=22.4,
            wait_time=35.0,
            flow_rate=6.0,
            time_since_last_vehicle_passed=1.0
        ),
        Approach.WEST: DirectionTraffic(
            direction=Approach.WEST,
            vehicle_counts={"two_wheeler": 4, "car": 2},
            queue_pcu=2.52,
            wait_time=10.0,
            flow_rate=2.0,
            time_since_last_vehicle_passed=5.0
        ),
    }

    sim_time = 0.0
    for tick in range(1, 36):
        sim_time += 1.0
        
        # Dynamic adjustments over time
        if tick >= 11:
            # North queue dissipates
            traffic[Approach.NORTH].queue_pcu = max(0.0, traffic[Approach.NORTH].queue_pcu - 2.0)
            traffic[Approach.NORTH].flow_rate = max(0.0, traffic[Approach.NORTH].flow_rate - 1.5)
            traffic[Approach.NORTH].time_since_last_vehicle_passed += 1.0
            
        decision = controller.step(traffic, dt=1.0)
        print_decision_block(tick, sim_time, traffic, decision)


if __name__ == "__main__":
    run_deterministic_simulation()
