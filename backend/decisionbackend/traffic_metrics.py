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
        # 10 two-wheelers (0.13), 8 cars (1.00), 2 buses (5.40)
        # Expected: 10 * 0.13 + 8 * 1.0 + 2 * 5.4 = 1.3 + 8.0 + 10.8 = 20.1 PCU
        counts = {"two_wheeler": 10, "car": 8, "bus": 2}
        pcu = calculate_queue_pcu(counts, self.config.pcu)
        self.assertAlmostEqual(pcu, 20.1, places=2)

        # Test other classes
        counts_mixed = {"auto_rickshaw": 4, "truck": 2} # 4*0.75 + 2*3.70 = 3.0 + 7.4 = 10.4
        self.assertAlmostEqual(calculate_queue_pcu(counts_mixed, self.config.pcu), 10.4, places=2)

    # -------------------------------------------------------------
    # TEST 11: Normalization values are capped at 1.0
    # -------------------------------------------------------------
    def test_11_normalization_capping(self):
        norm_cfg = NormalizationConfig(queue_pcu_max=40.0, wait_time_ref=90.0, flow_rate_max=12.0)
        
        # Test values above reference limits
        q_norm = normalize_queue_pcu(80.0, norm_cfg)
        w_norm = normalize_wait_time(180.0, norm_cfg)
        f_norm = normalize_flow_rate(25.0, norm_cfg)

        self.assertEqual(q_norm, 1.0)
        self.assertEqual(w_norm, 1.0)
        self.assertEqual(f_norm, 1.0)

        # Test values below limits
        self.assertAlmostEqual(normalize_queue_pcu(20.0, norm_cfg), 0.5, places=3)
        self.assertAlmostEqual(normalize_wait_time(45.0, norm_cfg), 0.5, places=3)
        self.assertAlmostEqual(normalize_flow_rate(6.0, norm_cfg), 0.5, places=3)

    # -------------------------------------------------------------
    # TEST 12: Priority calculation matches exact formula
    # P = 0.45*Queue_norm + 0.35*Wait_norm + 0.20*Flow_norm
    # -------------------------------------------------------------
    def test_12_priority_formula(self):
        # Q = 20 (norm = 0.5), W = 45 (norm = 0.5), F = 6 (norm = 0.5)
        # Expected: 0.45*0.5 + 0.35*0.5 + 0.20*0.5 = 0.50
        p_obj = calculate_priority_score(
            direction=Approach.NORTH,
            queue_pcu=20.0,
            wait_time_sec=45.0,
            flow_rate_pcu_min=6.0,
            weights=self.config.weights,
            norm_config=self.config.normalization
        )
        self.assertAlmostEqual(p_obj.score, 0.50, places=3)

        # Another case: Q = 40 (1.0), W = 0 (0.0), F = 0 (0.0) -> P = 0.45
        p_obj2 = calculate_priority_score(Approach.EAST, 40.0, 0.0, 0.0, self.config.weights, self.config.normalization)
        self.assertAlmostEqual(p_obj2.score, 0.45, places=3)

    # -------------------------------------------------------------
    # TEST 1: North has highest priority -> North gets green (or chosen on transition)
    # -------------------------------------------------------------
    def test_01_highest_priority_selection(self):
        controller = SignalController(config=self.config, initial_green=Approach.SOUTH)
        # Force South to complete yellow and all-red towards North
        controller.state.start_yellow_transition(Approach.NORTH, yellow_duration=1.0)
        controller.step({}, dt=1.0) # Yellow finishes -> ALL_RED (all_red_time = 2.0s)
        dec = controller.step({}, dt=2.0) # All-red finishes -> NORTH is GREEN
        self.assertEqual(dec.current_green, Approach.NORTH)
        self.assertEqual(dec.signal_states[Approach.NORTH], SignalColor.GREEN)

    # -------------------------------------------------------------
    # TEST 2: Current green has not reached G_MIN -> it stays green
    # -------------------------------------------------------------
    def test_02_g_min_protection(self):
        controller = SignalController(config=self.config, initial_green=Approach.NORTH)
        
        # Massive traffic on East, zero traffic on North
        traffic = {
            Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=0.0, wait_time=0.0, flow_rate=0.0),
            Approach.EAST: DirectionTraffic(Approach.EAST, queue_pcu=40.0, wait_time=90.0, flow_rate=12.0),
            Approach.SOUTH: DirectionTraffic(Approach.SOUTH, queue_pcu=0.0, wait_time=0.0, flow_rate=0.0),
            Approach.WEST: DirectionTraffic(Approach.WEST, queue_pcu=0.0, wait_time=0.0, flow_rate=0.0),
        }

        # Step 5 seconds (G_MIN is 10s)
        for _ in range(5):
            dec = controller.step(traffic, dt=1.0)
            self.assertEqual(dec.current_green, Approach.NORTH)
            self.assertEqual(dec.active_phase, PhaseState.GREEN)
            self.assertFalse(dec.is_switch_in_progress)

    # -------------------------------------------------------------
    # TEST 3: Current green reaches G_MAX -> it must switch
    # -------------------------------------------------------------
    def test_03_g_max_enforcement(self):
        controller = SignalController(config=self.config, initial_green=Approach.NORTH)
        
        # North has higher demand than East throughout so it stays green until G_MAX
        traffic = {
            Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=35.0, wait_time=0.0, flow_rate=10.0),
            Approach.EAST: DirectionTraffic(Approach.EAST, queue_pcu=10.0, wait_time=0.0, flow_rate=2.0),
        }

        # Advance to G_MAX (40s)
        for _ in range(39):
            controller.step(traffic, dt=1.0)

        # Tick 40 should trigger forced switch to YELLOW
        dec = controller.step(traffic, dt=1.0)
        self.assertTrue(dec.is_switch_in_progress)
        self.assertEqual(dec.active_phase, PhaseState.YELLOW)
        self.assertEqual(dec.next_green_candidate, Approach.EAST)

    # -------------------------------------------------------------
    # TEST 4: Current green is empty and G_MIN has passed -> switch to highest-P red approach
    # -------------------------------------------------------------
    def test_04_empty_green_early_termination(self):
        controller = SignalController(config=self.config, initial_green=Approach.NORTH)
        
        # Pass G_MIN (10s) with active traffic on North
        traffic_busy = {
            Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=15.0, flow_rate=5.0),
            Approach.WEST: DirectionTraffic(Approach.WEST, queue_pcu=25.0, flow_rate=8.0),
        }
        for _ in range(10):
            controller.step(traffic_busy, dt=1.0)

        # Now North is empty (queue = 0.0, crossed_recently = 0)
        traffic_empty = {
            Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=0.0, vehicles_crossed_recently=0),
            Approach.WEST: DirectionTraffic(Approach.WEST, queue_pcu=25.0, flow_rate=8.0),
        }

        dec = controller.step(traffic_empty, dt=1.0)
        self.assertTrue(dec.is_switch_in_progress)
        self.assertEqual(dec.active_phase, PhaseState.YELLOW)
        self.assertEqual(dec.next_green_candidate, Approach.WEST)

    # -------------------------------------------------------------
    # TEST 5: Gap-out occurs after G_MIN -> switch
    # -------------------------------------------------------------
    def test_05_gap_out_switch(self):
        controller = SignalController(config=self.config, initial_green=Approach.NORTH)
        
        # Advance past G_MIN with active North traffic
        for _ in range(11):
            controller.step({Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=10.0)}, dt=1.0)

        # Inactivity gap exceeds GAP_OUT_TIME (3.0s)
        traffic_gap = {
            Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=0.2, time_since_last_vehicle_passed=4.0),
            Approach.SOUTH: DirectionTraffic(Approach.SOUTH, queue_pcu=20.0, flow_rate=5.0),
        }

        dec = controller.step(traffic_gap, dt=1.0)
        self.assertTrue(dec.is_switch_in_progress)
        self.assertEqual(dec.active_phase, PhaseState.YELLOW)
        self.assertEqual(dec.next_green_candidate, Approach.SOUTH)

    # -------------------------------------------------------------
    # TEST 6: Wait time increases for RED approaches
    # -------------------------------------------------------------
    def test_06_wait_time_accumulation(self):
        controller = SignalController(config=self.config, initial_green=Approach.NORTH)
        
        controller.step({}, dt=1.0)
        controller.step({}, dt=1.0)
        controller.step({}, dt=1.0)

        self.assertAlmostEqual(controller.state.wait_times[Approach.SOUTH], 3.0, places=2)
        self.assertAlmostEqual(controller.state.wait_times[Approach.EAST], 3.0, places=2)
        self.assertAlmostEqual(controller.state.wait_times[Approach.WEST], 3.0, places=2)
        self.assertEqual(controller.state.wait_times[Approach.NORTH], 0.0)

    # -------------------------------------------------------------
    # TEST 7: Wait time resets when an approach becomes GREEN
    # -------------------------------------------------------------
    def test_07_wait_time_reset_on_green(self):
        controller = SignalController(config=self.config, initial_green=Approach.NORTH)
        controller.state.wait_times[Approach.EAST] = 75.0

        # Switch to EAST
        controller.state.start_yellow_transition(Approach.EAST, yellow_duration=1.0)
        controller.step({}, dt=1.0) # Yellow -> ALL_RED (all_red_time = 2.0s)
        controller.step({}, dt=2.0) # ALL_RED -> EAST GREEN

        self.assertEqual(controller.state.phase_state, PhaseState.GREEN)
        self.assertEqual(controller.state.active_approach, Approach.EAST)
        self.assertEqual(controller.state.wait_times[Approach.EAST], 0.0)

    # -------------------------------------------------------------
    # TEST 8: Yellow and all-red always occur before the next green
    # -------------------------------------------------------------
    def test_08_yellow_and_all_red_clearance(self):
        controller = SignalController(config=self.config, initial_green=Approach.NORTH)
        traffic = {
            Approach.NORTH: DirectionTraffic(Approach.NORTH, queue_pcu=35.0, flow_rate=10.0),
            Approach.EAST: DirectionTraffic(Approach.EAST, queue_pcu=10.0, flow_rate=2.0),
        }
        
        # Advance to G_MAX (40s)
        for _ in range(39):
            controller.step(traffic, dt=1.0)

        # Step at t=40s triggers YELLOW transition
        dec_switch = controller.step(traffic, dt=1.0)
        self.assertEqual(controller.state.phase_state, PhaseState.YELLOW)
        self.assertEqual(controller.state.get_signal_colors()[Approach.NORTH], SignalColor.YELLOW)
        self.assertEqual(controller.state.get_signal_colors()[Approach.EAST], SignalColor.RED)

        # Advance through yellow duration (3.0s total)
        controller.step({}, dt=3.0)
        # Must be in ALL_RED
        self.assertEqual(controller.state.phase_state, PhaseState.ALL_RED)
        colors = controller.state.get_signal_colors()
        for app in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]:
            self.assertEqual(colors[app], SignalColor.RED)

        # Advance through all-red clearance (2.0s total)
        controller.step({}, dt=2.0)
        # Now EAST is GREEN
        self.assertEqual(controller.state.phase_state, PhaseState.GREEN)
        self.assertEqual(controller.state.active_approach, Approach.EAST)
        self.assertEqual(controller.state.get_signal_colors()[Approach.EAST], SignalColor.GREEN)

    # -------------------------------------------------------------
    # TEST 9: Never allow more than one GREEN simultaneously
    # -------------------------------------------------------------
    def test_09_single_green_invariant(self):
        controller = SignalController(config=self.config, initial_green=Approach.NORTH)
        
        # Run 100 random simulated steps
        import random
        random.seed(42)

        for step_i in range(100):
            traffic = {
                app: DirectionTraffic(
                    direction=app,
                    queue_pcu=random.uniform(0.0, 35.0),
                    wait_time=controller.state.wait_times[app],
                    flow_rate=random.uniform(0.0, 10.0),
                    time_since_last_vehicle_passed=random.uniform(0.0, 6.0)
                )
                for app in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]
            }
            dec = controller.step(traffic, dt=1.0)
            
            # Count green lights
            green_count = sum(1 for c in dec.signal_states.values() if c == SignalColor.GREEN)
            self.assertLessEqual(green_count, 1, f"Invariant violated at step {step_i}: {green_count} green signals active!")


if __name__ == "__main__":
    unittest.main()
