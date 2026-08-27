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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    print("=" * 80)
    print(" 4-APPROACH TRAFFIC SIGNAL CONTROLLER: DETERMINISTIC SIMULATION DEMO")
    print("=" * 80)

    config = JunctionConfig(
        timing=SignalTimingConfig(
            g_min=6.0,
            g_max=20.0,
            yellow_time=2.0,
            all_red_time=1.0,
            gap_out_time=3.0,
            decision_interval=1.0,
            priority_switch_margin=0.15,
            empty_queue_threshold_pcu=0.5
        )
    )

    controller = SignalController(config=config, initial_green=Approach.NORTH)

    # Sequence of 35 discrete 1-second simulation ticks covering multiple test scenarios
    # Scenario Timeline:
    # 0s - 5s: North holds Green (G_MIN protection)
    # 6s - 9s: East demand surges, triggers Priority Switch to East (Yellow -> All-Red -> Green)
    # 10s - 16s: East is Green, clears queue, becomes empty at t=16s -> Early Termination Switch to South
    # 17s - 22s: South is Green, experiences gap-out at t=21s -> Gap-Out Switch to West
    # 23s - 35s: West stays Green under heavy load until G_MAX is reached -> Forced Switch
    
    sim_time = 0.0
    
    for tick in range(1, 36):
        sim_time += 1.0
        
        # Determine dynamic traffic inputs per scenario
        if sim_time <= 6.0:
            # Stage 1: North is active, East building queue
            traffic = {
                Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=12.0, flow_rate=6.0, time_since_last_vehicle_passed=0.5),
                Approach.SOUTH: DirectionTraffic(Approach.SOUTH, queue_pcu=4.0, flow_rate=1.0, time_since_last_vehicle_passed=5.0),
                Approach.EAST: DirectionTraffic(Approach.EAST, queue_pcu=32.0, flow_rate=10.0, time_since_last_vehicle_passed=0.2),
                Approach.WEST: DirectionTraffic(Approach.WEST, queue_pcu=2.0, flow_rate=0.5, time_since_last_vehicle_passed=8.0),
            }
        elif sim_time <= 10.0:
            # Stage 2: East high priority switch active
            traffic = {
                Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=5.0, flow_rate=2.0, time_since_last_vehicle_passed=2.0),
                Approach.SOUTH: DirectionTraffic(Approach.SOUTH, queue_pcu=6.0, flow_rate=1.5, time_since_last_vehicle_passed=4.0),
                Approach.EAST: DirectionTraffic(Approach.EAST, queue_pcu=28.0, flow_rate=11.0, time_since_last_vehicle_passed=0.1),
                Approach.WEST: DirectionTraffic(Approach.WEST, queue_pcu=2.0, flow_rate=0.5, time_since_last_vehicle_passed=8.0),
            }
        elif sim_time <= 15.0:
            # Stage 3: East is green, draining rapidly
            east_q = max(0.0, 28.0 - (sim_time - 10.0) * 5.0)
            traffic = {
                Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=5.0, flow_rate=1.0, time_since_last_vehicle_passed=3.0),
                Approach.SOUTH: DirectionTraffic(Approach.SOUTH, queue_pcu=22.0, flow_rate=7.0, time_since_last_vehicle_passed=0.3),
                Approach.EAST: DirectionTraffic(Approach.EAST, queue_pcu=east_q, flow_rate=4.0, time_since_last_vehicle_passed=0.5),
                Approach.WEST: DirectionTraffic(Approach.WEST, queue_pcu=3.0, flow_rate=0.5, time_since_last_vehicle_passed=9.0),
            }
        elif sim_time <= 17.0:
            # Stage 4: East queue depleted to 0.0 PCU -> Early termination
            traffic = {
                Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=5.0, flow_rate=1.0, time_since_last_vehicle_passed=3.0),
                Approach.SOUTH: DirectionTraffic(Approach.SOUTH, queue_pcu=25.0, flow_rate=8.0, time_since_last_vehicle_passed=0.2),
                Approach.EAST: DirectionTraffic(Approach.EAST, queue_pcu=0.0, flow_rate=0.0, time_since_last_vehicle_passed=6.0, vehicles_crossed_recently=0),
                Approach.WEST: DirectionTraffic(Approach.WEST, queue_pcu=3.0, flow_rate=0.5, time_since_last_vehicle_passed=9.0),
            }
        elif sim_time <= 22.0:
            # Stage 5: South is green, queue clears and headway opens up -> Gap-out
            traffic = {
                Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=8.0, flow_rate=2.0, time_since_last_vehicle_passed=2.0),
                Approach.SOUTH: DirectionTraffic(Approach.SOUTH, queue_pcu=0.2, flow_rate=0.5, time_since_last_vehicle_passed=4.0),
                Approach.EAST: DirectionTraffic(Approach.EAST, queue_pcu=2.0, flow_rate=0.5, time_since_last_vehicle_passed=8.0),
                Approach.WEST: DirectionTraffic(Approach.WEST, queue_pcu=30.0, flow_rate=11.0, time_since_last_vehicle_passed=0.1),
            }
        else:
            # Stage 6: West is heavily congested, stays green until G_MAX is hit
            traffic = {
                Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=18.0, flow_rate=5.0, time_since_last_vehicle_passed=0.5),
                Approach.SOUTH: DirectionTraffic(Approach.SOUTH, queue_pcu=4.0, flow_rate=1.0, time_since_last_vehicle_passed=6.0),
                Approach.EAST: DirectionTraffic(Approach.EAST, queue_pcu=6.0, flow_rate=1.5, time_since_last_vehicle_passed=5.0),
                Approach.WEST: DirectionTraffic(Approach.WEST, queue_pcu=35.0, flow_rate=12.0, time_since_last_vehicle_passed=0.1),
            }

        decision = controller.step(traffic, dt=1.0)
        print_decision_block(tick, sim_time, traffic, decision)

    print("\n[SUCCESS] Deterministic simulation run completed successfully.")


if __name__ == "__main__":
    run_deterministic_simulation()
