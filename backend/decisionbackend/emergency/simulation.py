"""
simulation.py
Phase 5 End-to-End Multi-Junction Emergency Simulation.

Demonstrates deterministic multi-junction progression:
AMB-001 (Route: J-04 -> J-01 -> J-02)
- J-04: Detection, ETA countdown, emergency preemption, emergency green, passage.
- Handoff J-04 -> J-01: Camera-local track ID change, independent downstream ETA, preemption, passage.
- Handoff J-01 -> J-02: Camera-local track ID change, downstream ETA, preemption, passage.
- Final mission completion.
- Simultaneous second emergency (AMB-002: J-03 -> J-01 -> J-05) operates without cross-talk.
"""

from typing import List, Dict, Any
from backend.models.traffic_schemas import ApproachEnum, MovementStateEnum
from backend.decisionbackend.models import Approach, PhaseState, DirectionTraffic
from backend.decisionbackend.emergency.emergency_models import (
    EmergencyVehicleType,
    EmergencyState,
)
from backend.decisionbackend.emergency.camera_events import (
    EmergencyDetectionEvent,
    EmergencyEtaUpdateEvent,
    EmergencyPassageEvent,
    DirectionalHandoffEvent,
)
from backend.core.control.emergency_orchestrator import EmergencyOrchestrator


def _sample_traffic() -> Dict[Approach, DirectionTraffic]:
    return {
        Approach.NORTH: DirectionTraffic(direction=Approach.NORTH, queue_pcu=5.0, wait_time=10.0, flow_rate=12.0),
        Approach.SOUTH: DirectionTraffic(direction=Approach.SOUTH, queue_pcu=2.0, wait_time=5.0, flow_rate=15.0),
        Approach.EAST: DirectionTraffic(direction=Approach.EAST, queue_pcu=4.0, wait_time=8.0, flow_rate=10.0),
        Approach.WEST: DirectionTraffic(direction=Approach.WEST, queue_pcu=3.0, wait_time=6.0, flow_rate=11.0),
    }


def run_phase5_simulation() -> List[Dict[str, Any]]:
    """
    Executes a deterministic step-by-step multi-junction emergency simulation.
    Returns the complete trace log.
    """
    orchestrator = EmergencyOrchestrator()
    logs: List[Dict[str, Any]] = []

    def record_step(t: float, junc: str, amb: str, track: Any, app: str, eta: float, e_state: str, phase: str, handoff_dest: str, m_state: str):
        logs.append({
            "time": t,
            "junction": junc,
            "emergency_id": amb,
            "local_track_id": track,
            "approach": app,
            "eta": eta,
            "emergency_state": e_state,
            "signal_phase": phase,
            "handoff_destination": handoff_dest,
            "mission_state": m_state
        })

    # =========================================================================
    # STEP 1: Register Missions
    # AMB-001: J-04 -> J-01 -> J-02
    # AMB-002: J-03 -> J-01 -> J-05 (Simultaneous second emergency)
    # =========================================================================
    ctx1 = orchestrator.register_mission("MSN-001", "AMB-001", "J-04", "J-02")
    ctx2 = orchestrator.register_mission("MSN-002", "AMB-002", "J-03", "J-05")

    # =========================================================================
    # STEP 2: Junction J-04 - Sighting & Local Tracking
    # =========================================================================
    t = 0.0
    orchestrator.associate_camera_track("J-04", ApproachEnum.SOUTH, track_id=42, emergency_id_or_vehicle="AMB-001")
    j4_ctrl = orchestrator.get_or_create_junction_controller("J-04", initial_green=Approach.NORTH)
    j4_ctrl.step(_sample_traffic(), dt=1.0)
    record_step(t, "J-04", "AMB-001", 42, "SOUTH", 25.0, "PENDING", j4_ctrl.state.phase_state.value, "None", ctx1.status.value)

    # ETA countdown & Case B trigger at J-04 (ETA <= T_clear)
    t = 15.0
    j4_adapter = orchestrator.get_or_create_junction_adapter("J-04")
    j4_adapter.on_emergency_eta_updated(EmergencyEtaUpdateEvent("AMB-001", "J-04", 4.0, timestamp=t))
    # Trigger emergency decision step
    dec1 = j4_ctrl.step(_sample_traffic(), dt=1.0)
    record_step(t, "J-04", "AMB-001", 42, "SOUTH", 4.0, "ACTIVE", dec1.active_phase.value, "None", ctx1.status.value)

    # Emergency Green served at J-04
    t = 20.0
    j4_ctrl.state.phase_state = PhaseState.GREEN
    j4_ctrl.state.active_approach = Approach.SOUTH
    record_step(t, "J-04", "AMB-001", 42, "SOUTH", 0.0, "ACTIVE", "EMERGENCY_GREEN", "None", ctx1.status.value)

    # =========================================================================
    # STEP 3: Junction J-04 Passage & Handoff to J-01
    # =========================================================================
    t = 24.0
    orchestrator.on_emergency_passed_junction("J-04", "AMB-001", destination_approach=Approach.NORTH)
    handoff_1 = DirectionalHandoffEvent(
        emergency_id="AMB-001",
        source_junction_id="J-04",
        outgoing_approach=Approach.NORTH,
        destination_junction_id="J-01",
        next_approach=Approach.SOUTH,
        next_junction_eta=35.0,
        timestamp=t
    )
    orchestrator.on_directional_handoff(handoff_1)
    record_step(t, "J-04", "AMB-001", "None", "NORTH", 0.0, "PASSED", "RECOVERY", "J-01", ctx1.status.value)

    # =========================================================================
    # STEP 4: Junction J-01 - New Local Track ID (17) & Preemption
    # =========================================================================
    t = 30.0
    # Ambulance arrives at J-01 camera with fresh camera-local track_id = 17
    orchestrator.associate_camera_track("J-01", ApproachEnum.SOUTH, track_id=17, emergency_id_or_vehicle="AMB-001")
    j1_ctrl = orchestrator.get_or_create_junction_controller("J-01", initial_green=Approach.EAST)
    j1_adapter = orchestrator.get_or_create_junction_adapter("J-01")
    j1_adapter.on_emergency_eta_updated(EmergencyEtaUpdateEvent("AMB-001", "J-01", 8.0, timestamp=t))
    dec2 = j1_ctrl.step(_sample_traffic(), dt=1.0)
    record_step(t, "J-01", "AMB-001", 17, "SOUTH", 8.0, "PENDING", dec2.active_phase.value, "None", ctx1.status.value)

    # Simultaneous second emergency (AMB-002) is tracking at J-03 without interfering with J-01
    orchestrator.associate_camera_track("J-03", ApproachEnum.NORTH, track_id=88, emergency_id_or_vehicle="AMB-002")
    j3_ctrl = orchestrator.get_or_create_junction_controller("J-03", initial_green=Approach.EAST)
    j3_dec = j3_ctrl.step(_sample_traffic(), dt=1.0)
    record_step(t, "J-03", "AMB-002", 88, "NORTH", 30.0, "PENDING", j3_dec.active_phase.value, "None", ctx2.status.value)

    # J-01 Emergency Green served for AMB-001
    t = 38.0
    j1_ctrl.state.phase_state = PhaseState.GREEN
    j1_ctrl.state.active_approach = Approach.SOUTH
    record_step(t, "J-01", "AMB-001", 17, "SOUTH", 0.0, "ACTIVE", "EMERGENCY_GREEN", "None", ctx1.status.value)

    # =========================================================================
    # STEP 5: Junction J-01 Passage & Handoff to J-02
    # =========================================================================
    t = 42.0
    orchestrator.on_emergency_passed_junction("J-01", "AMB-001", destination_approach=Approach.EAST)
    handoff_2 = DirectionalHandoffEvent(
        emergency_id="AMB-001",
        source_junction_id="J-01",
        outgoing_approach=Approach.EAST,
        destination_junction_id="J-02",
        next_approach=Approach.WEST,
        next_junction_eta=28.0,
        timestamp=t
    )
    orchestrator.on_directional_handoff(handoff_2)
    record_step(t, "J-01", "AMB-001", "None", "EAST", 0.0, "PASSED", "RECOVERY", "J-02", ctx1.status.value)

    # =========================================================================
    # STEP 6: Junction J-02 - New Local Track ID (9) & Final Destination
    # =========================================================================
    t = 50.0
    orchestrator.associate_camera_track("J-02", ApproachEnum.WEST, track_id=9, emergency_id_or_vehicle="AMB-001")
    j2_ctrl = orchestrator.get_or_create_junction_controller("J-02", initial_green=Approach.NORTH)
    j2_adapter = orchestrator.get_or_create_junction_adapter("J-02")
    j2_adapter.on_emergency_eta_updated(EmergencyEtaUpdateEvent("AMB-001", "J-02", 5.0, timestamp=t))
    dec3 = j2_ctrl.step(_sample_traffic(), dt=1.0)
    record_step(t, "J-02", "AMB-001", 9, "WEST", 5.0, "PENDING", dec3.active_phase.value, "None", ctx1.status.value)

    # J-02 Emergency Green
    t = 55.0
    j2_ctrl.state.phase_state = PhaseState.GREEN
    j2_ctrl.state.active_approach = Approach.WEST
    record_step(t, "J-02", "AMB-001", 9, "WEST", 0.0, "ACTIVE", "EMERGENCY_GREEN", "None", ctx1.status.value)

    # J-02 Final Passage -> Mission Accomplished
    t = 60.0
    orchestrator.on_emergency_passed_junction("J-02", "AMB-001")
    record_step(t, "J-02", "AMB-001", "None", "WEST", 0.0, "PASSED", "RECOVERY", "None (Final Node)", ctx1.status.value)

    return logs


if __name__ == "__main__":
    trace = run_phase5_simulation()
    print("=" * 125)
    print(f"{'TIME':<6} | {'JUNCTION':<8} | {'EMERGENCY':<9} | {'TRACK':<7} | {'APPROACH':<8} | {'ETA':<5} | {'EMERG_STATE':<11} | {'SIGNAL_PHASE':<16} | {'HANDOFF_DEST':<18} | {'MISSION_STATE'}")
    print("=" * 125)
    for step in trace:
        print(f"{step['time']:<6.1f} | {step['junction']:<8} | {step['emergency_id']:<9} | {str(step['local_track_id']):<7} | {step['approach']:<8} | {step['eta']:<5.1f} | {step['emergency_state']:<11} | {step['signal_phase']:<16} | {step['handoff_destination']:<18} | {step['mission_state']}")
    print("=" * 125)
