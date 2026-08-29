"""
tests.py
Unit test suite verifying all 12 core requirements for the normal traffic signal decision backend.
"""

import unittest
from backend.decisionbackend.models import (
    Approach,
    SignalColor,
    PhaseState,
    DirectionTraffic,
)
from backend.decisionbackend.junction_config import (
    JunctionConfig,
    PCUConfig,
    NormalizationConfig,
    PriorityWeights,
    SignalTimingConfig,
)
from backend.decisionbackend.pcu import calculate_queue_pcu
from backend.decisionbackend.traffic_metrics import (
    normalize_queue_pcu,
    normalize_wait_time,
    normalize_flow_rate,
    normalize_metrics,
)
from backend.decisionbackend.priority import calculate_priority_score, compute_all_priorities
from backend.decisionbackend.signal_controller import SignalController


class TestDecisionBackend(unittest.TestCase):

    def setUp(self):
        self.config = JunctionConfig(
            timing=SignalTimingConfig(
                g_min=10.0,
                g_max=40.0,
                yellow_time=3.0,
                all_red_time=2.0,
                gap_out_time=3.0,
                decision_interval=1.0,
                empty_queue_threshold_pcu=0.5
            )
        )

    # -------------------------------------------------------------
    # TEST 10: PCU calculation produces correct values
    # -------------------------------------------------------------
    def test_10_pcu_calculation(self):
        counts = {"two_wheeler": 10, "car": 8, "bus": 2}
        pcu = calculate_queue_pcu(counts, self.config.pcu)
        self.assertAlmostEqual(pcu, 20.1, places=2)

        counts_mixed = {"auto_rickshaw": 4, "truck": 2}
        self.assertAlmostEqual(calculate_queue_pcu(counts_mixed, self.config.pcu), 10.4, places=2)

    # -------------------------------------------------------------
    # TEST 01: Normalization strictly produces values in [0.0, 1.0]
    # -------------------------------------------------------------
    def test_01_normalization_bounds(self):
        norm = normalize_metrics(queue_pcu=60.0, wait_time_sec=120.0, flow_rate_pcu_min=20.0, config=self.config.normalization)
        self.assertEqual(norm.queue_norm, 1.0)
        self.assertEqual(norm.wait_norm, 1.0)
        self.assertEqual(norm.flow_norm, 1.0)

        norm_zero = normalize_metrics(queue_pcu=0.0, wait_time_sec=0.0, flow_rate_pcu_min=0.0, config=self.config.normalization)
        self.assertEqual(norm_zero.queue_norm, 0.0)
        self.assertEqual(norm_zero.wait_norm, 0.0)
        self.assertEqual(norm_zero.flow_norm, 0.0)

    # -------------------------------------------------------------
    # TEST 02: Priority score calculation
    # -------------------------------------------------------------
    def test_02_priority_calculation(self):
        # Q = 20 PCU (norm 0.5), W = 45s (norm 0.5), F = 6 PCU/min (norm 0.5)
        # Expected: 0.45*0.5 + 0.35*0.5 + 0.20*0.5 = 0.50
        score_obj = calculate_priority_score(
            direction=Approach.NORTH,
            queue_pcu=20.0,
            wait_time_sec=45.0,
            flow_rate_pcu_min=6.0,
            weights=self.config.weights,
            norm_config=self.config.normalization
        )
        self.assertAlmostEqual(score_obj.score, 0.50, places=3)
        self.assertTrue(0.0 <= score_obj.score <= 1.0)

    # -------------------------------------------------------------
    # TEST 03: G_MIN cannot be interrupted
    # -------------------------------------------------------------
    def test_03_gmin_protection(self):
        ctrl = SignalController(config=self.config, initial_green=Approach.NORTH)
        
        # High demand on EAST, low on NORTH
        traffic = {
            Approach.NORTH: DirectionTraffic(direction=Approach.NORTH, queue_pcu=0.1, flow_rate=0.0, time_since_last_vehicle_passed=5.0),
            Approach.SOUTH: DirectionTraffic(direction=Approach.SOUTH, queue_pcu=0.0),
            Approach.EAST: DirectionTraffic(direction=Approach.EAST, queue_pcu=40.0, wait_time=80.0, flow_rate=12.0),
            Approach.WEST: DirectionTraffic(direction=Approach.WEST, queue_pcu=0.0),
        }

        # Step through 5 seconds (< G_MIN of 10s)
        for _ in range(5):
            dec = ctrl.step(traffic, dt=1.0)
            self.assertEqual(dec.active_phase, PhaseState.GREEN)
            self.assertEqual(dec.current_green, Approach.NORTH)
            self.assertEqual(dec.signal_states[Approach.NORTH], SignalColor.GREEN)
            self.assertFalse(dec.is_switch_in_progress)

    # -------------------------------------------------------------
    # TEST 04: G_MAX forced switch
    # -------------------------------------------------------------
    def test_04_gmax_forced_switch(self):
        ctrl = SignalController(config=self.config, initial_green=Approach.NORTH)
        traffic = {
            Approach.NORTH: DirectionTraffic(direction=Approach.NORTH, queue_pcu=30.0, flow_rate=10.0),
            Approach.SOUTH: DirectionTraffic(direction=Approach.SOUTH, queue_pcu=0.0),
            Approach.EAST: DirectionTraffic(direction=Approach.EAST, queue_pcu=30.0, wait_time=60.0),
            Approach.WEST: DirectionTraffic(direction=Approach.WEST, queue_pcu=0.0),
        }

        # Step until G_MAX (40s)
        dec = None
        for _ in range(40):
            dec = ctrl.step(traffic, dt=1.0)

        self.assertEqual(dec.active_phase, PhaseState.YELLOW)
        self.assertTrue(dec.is_switch_in_progress)
        self.assertEqual(dec.next_green_candidate, Approach.EAST)

    # -------------------------------------------------------------
    # TEST 05: Empty approach early termination
    # -------------------------------------------------------------
    def test_05_empty_approach_early_termination(self):
        ctrl = SignalController(config=self.config, initial_green=Approach.NORTH)
        
        # NORTH has traffic during G_MIN
        traffic = {
            Approach.NORTH: DirectionTraffic(direction=Approach.NORTH, queue_pcu=10.0, flow_rate=5.0),
            Approach.SOUTH: DirectionTraffic(direction=Approach.SOUTH, queue_pcu=0.0),
            Approach.EAST: DirectionTraffic(direction=Approach.EAST, queue_pcu=25.0, wait_time=30.0),
            Approach.WEST: DirectionTraffic(direction=Approach.WEST, queue_pcu=0.0),
        }
        for _ in range(10): # G_MIN reached
            ctrl.step(traffic, dt=1.0)

        # Now NORTH queue empties completely
        traffic[Approach.NORTH].queue_pcu = 0.1
        traffic[Approach.NORTH].flow_rate = 0.0

        dec = ctrl.step(traffic, dt=1.0)
        self.assertEqual(dec.active_phase, PhaseState.YELLOW)
        self.assertEqual(dec.next_green_candidate, Approach.EAST)

    # -------------------------------------------------------------
    # TEST 06: Gap-out early termination
    # -------------------------------------------------------------
    def test_06_gap_out_early_termination(self):
        ctrl = SignalController(config=self.config, initial_green=Approach.NORTH)
        traffic = {
            Approach.NORTH: DirectionTraffic(direction=Approach.NORTH, queue_pcu=5.0, flow_rate=1.0, time_since_last_vehicle_passed=0.0),
            Approach.SOUTH: DirectionTraffic(direction=Approach.SOUTH, queue_pcu=0.0),
            Approach.EAST: DirectionTraffic(direction=Approach.EAST, queue_pcu=20.0, wait_time=40.0),
            Approach.WEST: DirectionTraffic(direction=Approach.WEST, queue_pcu=0.0),
        }
        for _ in range(10): # Reach G_MIN
            ctrl.step(traffic, dt=1.0)

        # Advance gap headway beyond 3.0s
        traffic[Approach.NORTH].time_since_last_vehicle_passed = 3.5
        dec = ctrl.step(traffic, dt=1.0)
        self.assertEqual(dec.active_phase, PhaseState.YELLOW)

    # -------------------------------------------------------------
    # TEST 07: Priority-based preemption after G_MIN
    # -------------------------------------------------------------
    def test_07_priority_switching(self):
        ctrl = SignalController(config=self.config, initial_green=Approach.NORTH)
        traffic = {
            Approach.NORTH: DirectionTraffic(direction=Approach.NORTH, queue_pcu=4.0, flow_rate=2.0), # Priority ~0.15
            Approach.SOUTH: DirectionTraffic(direction=Approach.SOUTH, queue_pcu=0.0),
            Approach.EAST: DirectionTraffic(direction=Approach.EAST, queue_pcu=35.0, wait_time=70.0, flow_rate=10.0), # Priority > 0.8
            Approach.WEST: DirectionTraffic(direction=Approach.WEST, queue_pcu=0.0),
        }
        for _ in range(10):
            ctrl.step(traffic, dt=1.0)

        # At tick 11, priority diff >= 0.25 -> switch
        dec = ctrl.step(traffic, dt=1.0)
        self.assertEqual(dec.active_phase, PhaseState.YELLOW)
        self.assertEqual(dec.next_green_candidate, Approach.EAST)

    # -------------------------------------------------------------
    # TEST 08: Exact clearance transition sequence
    # -------------------------------------------------------------
    def test_08_clearance_sequence(self):
        ctrl = SignalController(config=self.config, initial_green=Approach.NORTH)
        traffic = {
            Approach.NORTH: DirectionTraffic(direction=Approach.NORTH, queue_pcu=0.0, flow_rate=0.0),
            Approach.SOUTH: DirectionTraffic(direction=Approach.SOUTH, queue_pcu=0.0),
            Approach.EAST: DirectionTraffic(direction=Approach.EAST, queue_pcu=30.0, wait_time=50.0),
            Approach.WEST: DirectionTraffic(direction=Approach.WEST, queue_pcu=0.0),
        }
        for _ in range(10):
            ctrl.step(traffic, dt=1.0)

        # 1. Trigger transition -> YELLOW (duration 3s)
        dec = ctrl.step(traffic, dt=1.0)
        self.assertEqual(dec.active_phase, PhaseState.YELLOW)
        self.assertEqual(dec.signal_states[Approach.NORTH], SignalColor.YELLOW)

        # Step remaining yellow (2 ticks)
        dec = ctrl.step(traffic, dt=1.0)
        self.assertEqual(dec.active_phase, PhaseState.YELLOW)
        dec = ctrl.step(traffic, dt=1.0)

        # 2. Enters ALL_RED (duration 2s)
        self.assertEqual(dec.active_phase, PhaseState.ALL_RED)
        for app in Approach:
            self.assertEqual(dec.signal_states[app], SignalColor.RED)

        # Step remaining all-red (1 tick)
        dec = ctrl.step(traffic, dt=1.0)
        self.assertEqual(dec.active_phase, PhaseState.ALL_RED)

        # 3. Transitions to GREEN for EAST
        dec = ctrl.step(traffic, dt=1.0)
        self.assertEqual(dec.active_phase, PhaseState.GREEN)
        self.assertEqual(dec.current_green, Approach.EAST)
        self.assertEqual(dec.signal_states[Approach.EAST], SignalColor.GREEN)

    # -------------------------------------------------------------
    # TEST 09: Single green invariant
    # -------------------------------------------------------------
    def test_09_single_green_invariant(self):
        ctrl = SignalController(config=self.config, initial_green=Approach.NORTH)
        traffic = {
            Approach.NORTH: DirectionTraffic(direction=Approach.NORTH, queue_pcu=10.0),
            Approach.SOUTH: DirectionTraffic(direction=Approach.SOUTH, queue_pcu=15.0),
            Approach.EAST: DirectionTraffic(direction=Approach.EAST, queue_pcu=20.0),
            Approach.WEST: DirectionTraffic(direction=Approach.WEST, queue_pcu=25.0),
        }
        for _ in range(60):
            dec = ctrl.step(traffic, dt=1.0)
            greens = [app for app, color in dec.signal_states.items() if color == SignalColor.GREEN]
            self.assertTrue(len(greens) <= 1, f"Invariant violated: Multiple greens detected: {greens}")

    # -------------------------------------------------------------
    # TEST 11: Wait-time tracking
    # -------------------------------------------------------------
    def test_11_wait_time_tracking(self):
        ctrl = SignalController(config=self.config, initial_green=Approach.NORTH)
        traffic = {app: DirectionTraffic(direction=app, queue_pcu=5.0) for app in Approach}

        for _ in range(10):
            ctrl.step(traffic, dt=1.0)

        # NORTH wait time should remain 0 while GREEN, others increment by 10s
        self.assertEqual(ctrl.state.wait_times[Approach.NORTH], 0.0)
        self.assertAlmostEqual(ctrl.state.wait_times[Approach.SOUTH], 10.0, places=1)
        self.assertAlmostEqual(ctrl.state.wait_times[Approach.EAST], 10.0, places=1)
        self.assertAlmostEqual(ctrl.state.wait_times[Approach.WEST], 10.0, places=1)


if __name__ == "__main__":
    unittest.main()
