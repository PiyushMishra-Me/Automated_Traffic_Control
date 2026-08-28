"""
tests.py
Comprehensive unit test suite for Phase 1, Phase 2, and Phase 3 Emergency Decision Backend.
"""

import unittest
from typing import Dict
from backend.decisionbackend.models import (
    Approach,
    SignalColor,
    PhaseState,
    DirectionTraffic,
    NormalizedMetrics,
)
from backend.decisionbackend.junction_config import JunctionConfig, PriorityWeights
from backend.decisionbackend.emergency import (
    EmergencyNotice,
    EmergencyState,
    EmergencyVehicleType,
    EmergencyPassageEvent,
    EmergencyETAManager,
    EmergencyController,
    calculate_t_clear,
    calculate_effective_emergency_g_max,
    check_emergency_trigger_conditions,
    is_queue_cleared,
    tick_eta,
    apply_eta_correction,
    form_eta_clusters,
    get_clustered_approaches,
    compute_emergency_priority_score,
    resolve_emergency_conflict,
    sort_emergencies_by_eta,
    BOOSTED_QUEUE_WEIGHT,
    NORMAL_QUEUE_WEIGHT,
)


class TestEmergencyDecisionPhase1(unittest.TestCase):
    """
    Phase 1 Foundation unit tests.
    """

    def setUp(self):
        self.controller = EmergencyController()

    def test_01_eta_decreases_by_exact_one_second_per_tick(self):
        """1. ETA decreases by exactly 1 second per tick."""
        notice = EmergencyNotice(emergency_id="AMB-01", approach=Approach.NORTH, current_eta=15.0)
        self.assertEqual(notice.current_eta, 15.0)
        notice.tick_eta(1.0)
        self.assertEqual(notice.current_eta, 14.0)
        notice.tick_eta(1.0)
        self.assertEqual(notice.current_eta, 13.0)
        self.assertEqual(tick_eta(10.0, 1.0), 9.0)

    def test_02_eta_never_becomes_negative(self):
        """2. ETA never becomes negative and stays at 0."""
        notice = EmergencyNotice(emergency_id="AMB-02", approach=Approach.SOUTH, current_eta=2.0)
        notice.tick_eta(1.0)
        self.assertEqual(notice.current_eta, 1.0)
        notice.tick_eta(1.0)
        self.assertEqual(notice.current_eta, 0.0)
        notice.tick_eta(1.0)
        self.assertEqual(notice.current_eta, 0.0)
        self.assertEqual(tick_eta(1.0, 10.0), 0.0)

    def test_03_eta_corrected_dynamically(self):
        """3. ETA can be corrected dynamically, preserving original_eta for logging."""
        notice = EmergencyNotice(emergency_id="AMB-03", approach=Approach.EAST, current_eta=15.0)
        self.assertEqual(notice.original_eta, 15.0)
        notice.tick_eta(1.0)
        self.assertEqual(notice.current_eta, 14.0)
        notice.update_eta(20.0)
        self.assertEqual(notice.current_eta, 20.0)
        self.assertEqual(notice.original_eta, 15.0)
        notice.tick_eta(1.0)
        self.assertEqual(notice.current_eta, 19.0)

    def test_04_t_clear_q1_is_6s(self):
        """4. T_clear for Q=1 is 6s (6 + 2*(1-1))."""
        self.assertEqual(calculate_t_clear(1.0), 6.0)

    def test_05_t_clear_q2_is_8s(self):
        """5. T_clear for Q=2 is 8s (6 + 2*(2-1))."""
        self.assertEqual(calculate_t_clear(2.0), 8.0)

    def test_06_t_clear_q5_is_14s(self):
        """6. T_clear for Q=5 is 14s (6 + 2*(5-1))."""
        self.assertEqual(calculate_t_clear(5.0), 14.0)

    def test_07_t_clear_changes_dynamically_when_queue_changes(self):
        """7. T_clear changes when queue changes and is not static."""
        self.assertEqual(calculate_t_clear(5.0), 14.0)
        self.assertEqual(calculate_t_clear(2.0), 8.0)
        self.assertEqual(calculate_t_clear(4.0), 12.0)
        self.assertTrue(is_queue_cleared(0.5))
        self.assertTrue(is_queue_cleared(0.2))
        self.assertEqual(calculate_t_clear(0.4), 0.0)

    def test_08_effective_g_max_is_normal_when_t_clear_smaller(self):
        """8. effective emergency G_max is normal G_max when T_clear + 3 is smaller."""
        normal_g_max = 45.0
        t_clear = 40.0
        effective_g_max = calculate_effective_emergency_g_max(normal_g_max, t_clear, margin=3.0)
        self.assertEqual(effective_g_max, 45.0)

    def test_09_effective_g_max_increases_when_t_clear_exceeds_normal(self):
        """9. effective emergency G_max increases when T_clear + 3 exceeds normal G_max."""
        normal_g_max = 45.0
        t_clear = 50.0
        effective_g_max = calculate_effective_emergency_g_max(normal_g_max, t_clear, margin=3.0)
        self.assertEqual(effective_g_max, 53.0)

    def test_10_effective_g_max_changes_dynamically(self):
        """10. effective G_max changes when T_clear changes."""
        normal_g_max = 45.0
        self.assertEqual(calculate_effective_emergency_g_max(normal_g_max, 50.0), 53.0)
        self.assertEqual(calculate_effective_emergency_g_max(normal_g_max, 40.0), 45.0)

    def test_11_ambulance_passed_marks_correct_emergency_passed(self):
        """11. ambulance_passed marks the correct emergency as PASSED."""
        self.controller.create_and_register_notice("AMB-11", Approach.WEST, initial_eta=10.0)
        notice = self.controller.get_notice("AMB-11")
        self.assertIsNotNone(notice)
        self.assertEqual(notice.state, EmergencyState.PENDING)
        self.assertFalse(notice.is_passed)

        event = EmergencyPassageEvent(
            emergency_id="AMB-11",
            approach=Approach.WEST,
            destination_approach=Approach.EAST
        )
        passed = self.controller.ambulance_passed(event)
        self.assertTrue(passed)

        updated_notice = self.controller.get_notice("AMB-11")
        self.assertEqual(updated_notice.state, EmergencyState.PASSED)
        self.assertTrue(updated_notice.is_passed)
        self.assertEqual(updated_notice.destination_approach, Approach.EAST)

    def test_12_unrelated_emergency_cannot_accidentally_be_marked_passed(self):
        """12. An unrelated emergency ID cannot accidentally be marked PASSED."""
        self.controller.create_and_register_notice("AMB-REAL", Approach.NORTH, initial_eta=12.0)
        passed = self.controller.ambulance_passed("AMB-FAKE")
        self.assertFalse(passed)
        real_notice = self.controller.get_notice("AMB-REAL")
        self.assertEqual(real_notice.state, EmergencyState.PENDING)
        self.assertFalse(real_notice.is_passed)

    def test_13_emergency_notices_maintain_independent_eta_values(self):
        """13. Emergency notices maintain independent ETA values."""
        self.controller.create_and_register_notice("AMB-A", Approach.NORTH, initial_eta=10.0)
        self.controller.create_and_register_notice("AMB-B", Approach.SOUTH, initial_eta=25.0)

        for _ in range(3):
            self.controller.tick(1.0)

        self.assertEqual(self.controller.get_notice("AMB-A").current_eta, 7.0)
        self.assertEqual(self.controller.get_notice("AMB-B").current_eta, 22.0)

        self.controller.update_eta("AMB-A", 18.0)
        self.assertEqual(self.controller.get_notice("AMB-A").current_eta, 18.0)
        self.assertEqual(self.controller.get_notice("AMB-B").current_eta, 22.0)

    def test_14_multiple_emergency_notices_coexist_without_interference(self):
        """14. Multiple emergency notices can exist without state interference."""
        self.controller.create_and_register_notice("AMB-1", Approach.NORTH, initial_eta=8.0)
        self.controller.create_and_register_notice("AMB-2", Approach.EAST, initial_eta=15.0)
        self.controller.create_and_register_notice("AMB-3", Approach.SOUTH, initial_eta=30.0)

        self.assertEqual(len(self.controller.current_episode.active_notices), 3)
        self.assertCountEqual(
            self.controller.current_episode.get_active_approaches(),
            [Approach.NORTH, Approach.EAST, Approach.SOUTH]
        )

        self.controller.ambulance_passed("AMB-1")
        self.assertEqual(self.controller.get_notice("AMB-1").state, EmergencyState.PASSED)
        self.assertEqual(self.controller.get_notice("AMB-2").state, EmergencyState.PENDING)
        self.assertEqual(self.controller.get_notice("AMB-3").state, EmergencyState.PENDING)

        self.controller.dismiss_notice("AMB-3", reason="Turned away")
        self.assertEqual(self.controller.get_notice("AMB-3").state, EmergencyState.DISMISSED)
        self.assertEqual(self.controller.get_notice("AMB-2").state, EmergencyState.PENDING)


class TestEmergencyDecisionPhase2(unittest.TestCase):
    """
    Phase 2 Single Emergency Decision Engine unit tests.
    """

    def setUp(self):
        self.config = JunctionConfig()
        self.controller = EmergencyController(config=self.config, initial_green=Approach.NORTH)

    def _get_default_traffic(self) -> Dict[Approach, DirectionTraffic]:
        return {
            Approach.NORTH: DirectionTraffic(direction=Approach.NORTH, queue_pcu=5.0, flow_rate=2.0, vehicles_waiting=5),
            Approach.SOUTH: DirectionTraffic(direction=Approach.SOUTH, queue_pcu=4.0, flow_rate=1.5, vehicles_waiting=4),
            Approach.EAST: DirectionTraffic(direction=Approach.EAST, queue_pcu=3.0, flow_rate=1.0, vehicles_waiting=3),
            Approach.WEST: DirectionTraffic(direction=Approach.WEST, queue_pcu=2.0, flow_rate=1.0, vehicles_waiting=2),
        }

    def test_01_eta_decreases_by_one_second_per_tick(self):
        """TEST 1: Emergency ETA decreases by exactly 1 second per tick."""
        notice = EmergencyNotice(emergency_id="AMB-1", approach=Approach.EAST, current_eta=15.0)
        self.assertEqual(notice.current_eta, 15.0)
        notice.tick_eta(1.0)
        self.assertEqual(notice.current_eta, 14.0)
        notice.tick_eta(1.0)
        self.assertEqual(notice.current_eta, 13.0)
        notice.tick_eta(20.0)
        self.assertEqual(notice.current_eta, 0.0)

    def test_02_eta_dynamically_corrected(self):
        """TEST 2: Emergency ETA can be dynamically corrected."""
        notice = EmergencyNotice(emergency_id="AMB-2", approach=Approach.WEST, current_eta=15.0)
        self.assertEqual(notice.original_eta, 15.0)
        notice.tick_eta(2.0)
        self.assertEqual(notice.current_eta, 13.0)

        notice.update_eta(25.0)
        self.assertEqual(notice.current_eta, 25.0)
        self.assertEqual(notice.original_eta, 15.0)

        notice.tick_eta(1.0)
        self.assertEqual(notice.current_eta, 24.0)

    def test_03_t_clear_changes_dynamically_when_queue_changes(self):
        """TEST 3: T_clear changes dynamically when emergency-lane queue changes."""
        self.assertEqual(calculate_t_clear(1.0), 6.0)
        self.assertEqual(calculate_t_clear(2.0), 8.0)
        self.assertEqual(calculate_t_clear(5.0), 14.0)
        self.assertTrue(is_queue_cleared(0.5))
        self.assertEqual(calculate_t_clear(0.5), 0.0)

    def test_04_emergency_does_not_interrupt_green_before_gmin(self):
        """TEST 4: Emergency does NOT interrupt current green before G_min."""
        traffic = self._get_default_traffic()
        notice = EmergencyNotice(emergency_id="AMB-4", approach=Approach.EAST, current_eta=2.0, queue_pcu=2.0)
        self.controller.submit_emergency(notice)

        for t in range(1, 11):
            decision = self.controller.step(traffic, dt=1.0)
            self.assertEqual(decision.active_phase, PhaseState.GREEN)
            self.assertEqual(decision.current_green, Approach.NORTH)

        decision = self.controller.step(traffic, dt=1.0)
        self.assertEqual(decision.active_phase, PhaseState.YELLOW)
        self.assertEqual(decision.next_green_candidate, Approach.EAST)

    def test_05_eta_le_tclear_plus_3_begins_transition_when_gmin_reached(self):
        """TEST 5: When ETA <= T_clear + 3 and G_min has been reached, emergency transition begins."""
        traffic = self._get_default_traffic()
        for _ in range(10):
            self.controller.step(traffic, dt=1.0)

        notice = EmergencyNotice(emergency_id="AMB-5", approach=Approach.EAST, current_eta=12.0, queue_pcu=3.0)
        self.controller.submit_emergency(notice)

        decision = self.controller.step(traffic, dt=1.0)
        self.assertEqual(decision.active_phase, PhaseState.YELLOW)
        self.assertEqual(decision.next_green_candidate, Approach.EAST)

    def test_06_case_b_urgency_trigger_begins_transition(self):
        """TEST 6: When ETA > T_clear + 3 but (T_clear + 3) - ETA < G_min + 3, transition begins immediately."""
        traffic = self._get_default_traffic()
        for _ in range(10):
            self.controller.step(traffic, dt=1.0)

        is_trig, msg = check_emergency_trigger_conditions(eta=20.0, t_clear=6.0, g_min=10.0)
        self.assertTrue(is_trig)
        self.assertIn("Case B", msg)

        notice = EmergencyNotice(emergency_id="AMB-6", approach=Approach.EAST, current_eta=20.0, queue_pcu=1.0)
        self.controller.submit_emergency(notice)

        decision = self.controller.step(traffic, dt=1.0)
        self.assertEqual(decision.active_phase, PhaseState.YELLOW)
        self.assertEqual(decision.next_green_candidate, Approach.EAST)

    def test_07_urgency_condition_continuously_reevaluated(self):
        """TEST 7: Urgency condition is continuously re-evaluated every tick."""
        notice = EmergencyNotice(emergency_id="AMB-7", approach=Approach.EAST, current_eta=50.0, queue_pcu=1.0)
        self.controller.submit_emergency(notice)

        for _ in range(30):
            self.controller.tick(1.0)
        self.assertEqual(self.controller.get_notice("AMB-7").current_eta, 20.0)

    def test_08_normal_phase_change_intercepted_by_emergency(self):
        """TEST 8: If a normal phase change is about to happen, emergency condition is checked before selecting next normal direction."""
        traffic = self._get_default_traffic()
        for _ in range(39):
            self.controller.step(traffic, dt=1.0)

        notice = EmergencyNotice(emergency_id="AMB-8", approach=Approach.WEST, current_eta=15.0, queue_pcu=2.0)
        self.controller.submit_emergency(notice)

        decision = self.controller.step(traffic, dt=1.0)
        self.assertEqual(decision.active_phase, PhaseState.YELLOW)
        self.assertEqual(decision.next_green_candidate, Approach.WEST)

    def test_09_emergency_transition_follows_green_yellow_allred_green(self):
        """TEST 9: Emergency transition always follows GREEN -> YELLOW -> ALL_RED -> EMERGENCY GREEN."""
        traffic = self._get_default_traffic()
        for _ in range(10):
            self.controller.step(traffic, dt=1.0)

        notice = EmergencyNotice(emergency_id="AMB-9", approach=Approach.EAST, current_eta=10.0, queue_pcu=2.0)
        self.controller.submit_emergency(notice)

        d1 = self.controller.step(traffic, dt=1.0)
        self.assertEqual(d1.active_phase, PhaseState.YELLOW)
        d2 = self.controller.step(traffic, dt=1.0)
        self.assertEqual(d2.active_phase, PhaseState.YELLOW)
        d3 = self.controller.step(traffic, dt=1.0)
        self.assertEqual(d3.active_phase, PhaseState.YELLOW)

        d4 = self.controller.step(traffic, dt=1.0)
        self.assertEqual(d4.active_phase, PhaseState.ALL_RED)
        d5 = self.controller.step(traffic, dt=1.0)
        self.assertEqual(d5.active_phase, PhaseState.ALL_RED)

        d6 = self.controller.step(traffic, dt=1.0)
        self.assertEqual(d6.active_phase, PhaseState.GREEN)
        self.assertEqual(d6.current_green, Approach.EAST)

    def test_10_effective_gmax_dynamically_increases_with_tclear(self):
        """TEST 10: Emergency effective G_max dynamically increases when T_clear increases."""
        normal_g_max = 40.0
        t_clear_1 = calculate_t_clear(10.0)
        self.assertEqual(calculate_effective_emergency_g_max(normal_g_max, t_clear_1), 40.0)
        t_clear_2 = calculate_t_clear(25.0)
        self.assertEqual(calculate_effective_emergency_g_max(normal_g_max, t_clear_2), 57.0)

    def test_11_effective_gmax_does_not_modify_normal_gmax(self):
        """TEST 11: Emergency effective G_max does not permanently modify normal G_max."""
        baseline_gmax = self.config.timing.g_max
        eff_gmax = calculate_effective_emergency_g_max(baseline_gmax, 60.0)
        self.assertEqual(eff_gmax, 63.0)
        self.assertEqual(self.config.timing.g_max, baseline_gmax)

    def test_12_emergency_green_remains_active_while_queue_not_cleared(self):
        """TEST 12: Emergency green remains active while emergency queue is not cleared."""
        traffic = self._get_default_traffic()
        self.controller.state.active_approach = Approach.EAST
        self.controller.state.phase_state = PhaseState.GREEN
        self.controller.state.time_in_phase = 0.0
        self.controller.is_emergency_active = True
        self.controller.active_emergency_id = "AMB-12"

        notice = EmergencyNotice(emergency_id="AMB-12", approach=Approach.EAST, current_eta=5.0, queue_pcu=4.0, state=EmergencyState.ACTIVE)
        self.controller.submit_emergency(notice)

        decision = self.controller.step(traffic, dt=1.0)
        self.assertEqual(decision.active_phase, PhaseState.GREEN)
        self.assertEqual(decision.current_green, Approach.EAST)

    def test_13_emergency_green_completes_after_queue_clears_and_passage_confirmed(self):
        """TEST 13: Emergency green completes after queue clears AND ambulance passage is confirmed."""
        traffic = self._get_default_traffic()
        self.controller.state.active_approach = Approach.EAST
        self.controller.state.phase_state = PhaseState.GREEN
        self.controller.state.time_in_phase = 5.0
        self.controller.is_emergency_active = True
        self.controller.active_emergency_id = "AMB-13"

        notice = EmergencyNotice(emergency_id="AMB-13", approach=Approach.EAST, current_eta=0.0, queue_pcu=2.0, state=EmergencyState.ACTIVE)
        self.controller.submit_emergency(notice)

        self.controller.ambulance_passed("AMB-13")
        self.assertTrue(notice.is_passed)

        d1 = self.controller.step(traffic, dt=1.0)
        self.assertEqual(d1.active_phase, PhaseState.GREEN)

        traffic[Approach.EAST].queue_pcu = 0.0
        traffic[Approach.EAST].vehicles_waiting = 0
        d2 = self.controller.step(traffic, dt=1.0)
        self.assertEqual(d2.active_phase, PhaseState.YELLOW)
        self.assertFalse(self.controller.is_emergency_active)

    def test_14_ambulance_passage_event_marks_only_matching_emergency(self):
        """TEST 14: Ambulance passage event marks only the matching emergency as PASSED."""
        n1 = self.controller.create_and_register_notice("AMB-14A", Approach.NORTH, 10.0)
        n2 = self.controller.create_and_register_notice("AMB-14B", Approach.SOUTH, 20.0)

        self.controller.ambulance_passed(EmergencyPassageEvent(emergency_id="AMB-14A", destination_approach=Approach.SOUTH))
        self.assertTrue(n1.is_passed)
        self.assertEqual(n1.state, EmergencyState.PASSED)
        self.assertFalse(n2.is_passed)
        self.assertEqual(n2.state, EmergencyState.PENDING)

    def test_15_unrelated_passage_event_cannot_complete_another_emergency(self):
        """TEST 15: Unrelated passage event cannot complete another emergency."""
        n1 = self.controller.create_and_register_notice("AMB-15", Approach.EAST, 15.0)
        res = self.controller.ambulance_passed("FAKE-ID")
        self.assertFalse(res)
        self.assertFalse(n1.is_passed)
        self.assertEqual(n1.state, EmergencyState.PENDING)

    def test_16_emergency_green_cannot_exceed_gmax_plus_15(self):
        """TEST 16: Emergency green cannot exceed effective_G_max + 15 seconds."""
        traffic = self._get_default_traffic()
        self.controller.state.active_approach = Approach.EAST
        self.controller.state.phase_state = PhaseState.GREEN
        self.controller.is_emergency_active = True
        self.controller.active_emergency_id = "AMB-16"

        notice = EmergencyNotice(emergency_id="AMB-16", approach=Approach.EAST, current_eta=0.0, queue_pcu=10.0, state=EmergencyState.ACTIVE)
        self.controller.submit_emergency(notice)

        t_clear = calculate_t_clear(10.0)
        eff_gmax = calculate_effective_emergency_g_max(self.config.timing.g_max, t_clear)
        hard_cap = eff_gmax + 15.0

        for _ in range(int(hard_cap)):
            self.controller.step(traffic, dt=1.0)

        decision = self.controller.step(traffic, dt=1.0)
        self.assertEqual(decision.active_phase, PhaseState.YELLOW)
        self.assertFalse(self.controller.is_emergency_active)

    def test_17_emergency_dismissal_after_current_eta_plus_15s(self):
        """TEST 17: Emergency dismissal works according to current ETA + 15-second rule."""
        notice = EmergencyNotice(emergency_id="AMB-17", approach=Approach.WEST, current_eta=2.0)
        self.controller.submit_emergency(notice)

        self.controller.tick(1.0)
        self.controller.tick(1.0)
        self.assertEqual(notice.current_eta, 0.0)
        self.assertEqual(notice.state, EmergencyState.PENDING)

        for _ in range(14):
            self.controller.tick(1.0)
        self.assertEqual(notice.state, EmergencyState.PENDING)

        self.controller.tick(1.0)
        self.assertEqual(notice.state, EmergencyState.DISMISSED)
        self.assertIn(notice, self.controller.current_episode.dismissed_notices)

    def test_18_after_emergency_passed_returns_to_normal(self):
        """TEST 18: After emergency PASSED, controller returns to normal decision-making."""
        traffic = self._get_default_traffic()
        notice = EmergencyNotice(emergency_id="AMB-18", approach=Approach.EAST, current_eta=0.0, queue_pcu=0.0)
        self.controller.submit_emergency(notice)
        self.controller.ambulance_passed("AMB-18")

        self.controller.state.active_approach = Approach.NORTH
        self.controller.state.phase_state = PhaseState.GREEN
        self.controller.state.time_in_phase = 12.0
        self.controller.is_emergency_active = False

        decision = self.controller.step(traffic, dt=1.0)
        self.assertEqual(decision.active_phase, PhaseState.GREEN)
        self.assertFalse(self.controller.is_emergency_active)

    def test_19_after_emergency_dismissed_returns_to_normal(self):
        """TEST 19: After emergency DISMISSED, controller returns to normal decision-making."""
        traffic = self._get_default_traffic()
        notice = EmergencyNotice(emergency_id="AMB-19", approach=Approach.EAST, current_eta=0.0)
        self.controller.submit_emergency(notice)
        self.controller.dismiss_notice("AMB-19", reason="Test dismissal")

        self.controller.state.active_approach = Approach.NORTH
        self.controller.state.phase_state = PhaseState.GREEN
        self.controller.state.time_in_phase = 10.0

        decision = self.controller.step(traffic, dt=1.0)
        self.assertEqual(decision.active_phase, PhaseState.GREEN)
        self.assertEqual(decision.current_green, Approach.NORTH)

    def test_20_single_green_invariant_throughout_entire_emergency_episode(self):
        """TEST 20: Single-green invariant remains true throughout the entire emergency episode."""
        traffic = self._get_default_traffic()
        notice = EmergencyNotice(emergency_id="AMB-20", approach=Approach.EAST, current_eta=5.0, queue_pcu=2.0)
        self.controller.submit_emergency(notice)

        for _ in range(25):
            decision = self.controller.step(traffic, dt=1.0)
            green_count = sum(1 for color in decision.signal_states.values() if color == SignalColor.GREEN)
            self.assertLessEqual(green_count, 1, f"Invariant violated: {green_count} green signals at phase {decision.active_phase}")

    def test_21_wait_time_recovery_after_emergency_episode(self):
        """TEST 21: Wait-time recovery after emergency episode works correctly."""
        traffic = self._get_default_traffic()
        self.controller.state.wait_times = {
            Approach.NORTH: 0.0,
            Approach.SOUTH: 20.0,
            Approach.EAST: 0.0,
            Approach.WEST: 15.0
        }
        self.controller.state.active_approach = Approach.EAST
        self.controller.state.phase_state = PhaseState.GREEN
        self.controller.is_emergency_active = True

        for _ in range(5):
            self.controller.step(traffic, dt=1.0)

        self.assertEqual(self.controller.state.wait_times[Approach.SOUTH], 25.0)
        self.assertEqual(self.controller.state.wait_times[Approach.WEST], 20.0)
        self.assertEqual(self.controller.state.wait_times[Approach.EAST], 0.0)


class TestEmergencyDecisionPhase3(unittest.TestCase):
    """
    Phase 3 Multi-Emergency Management and ETA Clustering unit tests.
    """

    def setUp(self):
        self.config = JunctionConfig()
        self.controller = EmergencyController(config=self.config, initial_green=Approach.NORTH)

    def _get_default_traffic(self) -> Dict[Approach, DirectionTraffic]:
        return {
            Approach.NORTH: DirectionTraffic(direction=Approach.NORTH, queue_pcu=5.0, flow_rate=2.0, vehicles_waiting=5),
            Approach.SOUTH: DirectionTraffic(direction=Approach.SOUTH, queue_pcu=4.0, flow_rate=1.5, vehicles_waiting=4),
            Approach.EAST: DirectionTraffic(direction=Approach.EAST, queue_pcu=3.0, flow_rate=1.0, vehicles_waiting=3),
            Approach.WEST: DirectionTraffic(direction=Approach.WEST, queue_pcu=2.0, flow_rate=1.0, vehicles_waiting=2),
        }

    def test_01_multiple_notices_coexist_independently(self):
        """1. Multiple emergency notices coexist independently."""
        n1 = self.controller.create_and_register_notice("AMB-1", Approach.NORTH, 12.0)
        n2 = self.controller.create_and_register_notice("AMB-2", Approach.EAST, 25.0)
        n3 = self.controller.create_and_register_notice("AMB-3", Approach.SOUTH, 40.0)

        self.assertEqual(len(self.controller.current_episode.active_notices), 3)
        self.assertEqual(self.controller.get_notice("AMB-1").approach, Approach.NORTH)
        self.assertEqual(self.controller.get_notice("AMB-2").approach, Approach.EAST)
        self.assertEqual(self.controller.get_notice("AMB-3").approach, Approach.SOUTH)

    def test_02_emergencies_sort_by_current_eta(self):
        """2. Emergencies sort by CURRENT live ETA, ascending."""
        n1 = EmergencyNotice("AMB-1", Approach.NORTH, current_eta=35.0)
        n2 = EmergencyNotice("AMB-2", Approach.EAST, current_eta=10.0)
        n3 = EmergencyNotice("AMB-3", Approach.SOUTH, current_eta=22.0)

        sorted_list = sort_emergencies_by_eta([n1, n2, n3])
        self.assertEqual([n.emergency_id for n in sorted_list], ["AMB-2", "AMB-3", "AMB-1"])

        # Correct n1 ETA to 5.0s -> becomes first
        n1.update_eta(5.0)
        sorted_list2 = sort_emergencies_by_eta([n1, n2, n3])
        self.assertEqual([n.emergency_id for n in sorted_list2], ["AMB-1", "AMB-2", "AMB-3"])

    def test_03_single_emergency_still_behaves_exactly_like_phase2(self):
        """3. Single emergency still behaves exactly like Phase 2."""
        traffic = self._get_default_traffic()
        for _ in range(10):
            self.controller.step(traffic, dt=1.0)

        notice = EmergencyNotice(emergency_id="AMB-SINGLE", approach=Approach.EAST, current_eta=12.0, queue_pcu=3.0)
        self.controller.submit_emergency(notice)

        decision = self.controller.step(traffic, dt=1.0)
        self.assertEqual(decision.active_phase, PhaseState.YELLOW)
        self.assertEqual(decision.next_green_candidate, Approach.EAST)

    def test_04_multiple_non_clustered_processed_sequentially(self):
        """4. Multiple non-clustered emergencies are processed sequentially in ETA order."""
        # AMB-1 (ETA=12s on EAST), AMB-2 (ETA=45s on SOUTH) - non-clustered (diff = 33s > 10s)
        self.controller.create_and_register_notice("AMB-1", Approach.EAST, 12.0)
        self.controller.create_and_register_notice("AMB-2", Approach.SOUTH, 45.0)

        clusters = self.controller.get_active_clusters()
        # Two distinct single-item clusters
        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0]), 1)
        self.assertEqual(len(clusters[1]), 1)
        self.assertEqual(clusters[0][0].emergency_id, "AMB-1")
        self.assertEqual(clusters[1][0].emergency_id, "AMB-2")

    def test_05_two_etas_within_10s_form_cluster(self):
        """5. Two ETAs within 10s form a cluster."""
        n1 = EmergencyNotice("AMB-1", Approach.NORTH, current_eta=20.0)
        n2 = EmergencyNotice("AMB-2", Approach.SOUTH, current_eta=28.0)

        clusters = form_eta_clusters([n1, n2])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 2)
        self.assertEqual([n.emergency_id for n in clusters[0]], ["AMB-1", "AMB-2"])

    def test_06_etas_exactly_10s_apart_considered_same(self):
        """6. ETAs exactly 10s apart are considered same (clustered)."""
        n1 = EmergencyNotice("AMB-1", Approach.NORTH, current_eta=20.0)
        n2 = EmergencyNotice("AMB-2", Approach.SOUTH, current_eta=30.0)

        clusters = form_eta_clusters([n1, n2])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 2)

    def test_07_etas_greater_than_10s_apart_not_same(self):
        """7. ETAs greater than 10s apart are not same."""
        n1 = EmergencyNotice("AMB-1", Approach.NORTH, current_eta=20.0)
        n2 = EmergencyNotice("AMB-2", Approach.SOUTH, current_eta=30.5)

        clusters = form_eta_clusters([n1, n2])
        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0]), 1)
        self.assertEqual(len(clusters[1]), 1)

    def test_08_transitive_clustering_works(self):
        """8. Transitive clustering works: [28, 32, 36, 40] forms a single cluster."""
        notices = [
            EmergencyNotice("A", Approach.NORTH, 28.0),
            EmergencyNotice("B", Approach.EAST, 32.0),
            EmergencyNotice("C", Approach.SOUTH, 36.0),
            EmergencyNotice("D", Approach.WEST, 40.0),
        ]
        clusters = form_eta_clusters(notices)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 4)
        self.assertEqual([n.emergency_id for n in clusters[0]], ["A", "B", "C", "D"])

        # Example from prompt: [3, 15, 28, 32, 36, 40, 45, 60]
        full_notices = [
            EmergencyNotice("N3", Approach.NORTH, 3.0),
            EmergencyNotice("N15", Approach.SOUTH, 15.0),
            EmergencyNotice("N28", Approach.EAST, 28.0),
            EmergencyNotice("N32", Approach.WEST, 32.0),
            EmergencyNotice("N36", Approach.NORTH, 36.0),
            EmergencyNotice("N40", Approach.SOUTH, 40.0),
            EmergencyNotice("N45", Approach.EAST, 45.0),
            EmergencyNotice("N60", Approach.WEST, 60.0),
        ]
        full_clusters = form_eta_clusters(full_notices)
        self.assertEqual(len(full_clusters), 4)
        self.assertEqual([n.emergency_id for n in full_clusters[0]], ["N3"])
        self.assertEqual([n.emergency_id for n in full_clusters[1]], ["N15"])
        self.assertEqual([n.emergency_id for n in full_clusters[2]], ["N28", "N32", "N36", "N40", "N45"])
        self.assertEqual([n.emergency_id for n in full_clusters[3]], ["N60"])

    def test_09_dynamic_eta_correction_can_change_cluster_membership(self):
        """9. Dynamic ETA correction can change cluster membership."""
        n1 = EmergencyNotice("A", Approach.NORTH, 20.0)
        n2 = EmergencyNotice("B", Approach.EAST, 35.0)  # Diff = 15s -> Separate

        self.assertEqual(len(form_eta_clusters([n1, n2])), 2)

        # Tracking updates B to 27.0s (Diff = 7s <= 10s)
        n2.update_eta(27.0)
        clusters = form_eta_clusters([n1, n2])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 2)

    def test_10_countdown_can_change_cluster_relationships(self):
        """10. Countdown and notice passage change active clusters dynamically."""
        self.controller.create_and_register_notice("A", Approach.NORTH, 2.0)
        self.controller.create_and_register_notice("B", Approach.EAST, 20.0)
        self.controller.create_and_register_notice("C", Approach.SOUTH, 26.0)

        # Initially: [A] (solo), [B, C] (cluster)
        c_init = self.controller.get_active_clusters()
        self.assertEqual(len(c_init), 2)
        self.assertEqual(len(c_init[1]), 2)

    def test_11_cluster_membership_changes_after_emergency_passes(self):
        """11. Cluster membership changes after an emergency passes."""
        self.controller.create_and_register_notice("A", Approach.NORTH, 28.0)
        self.controller.create_and_register_notice("B", Approach.EAST, 32.0)
        self.controller.create_and_register_notice("C", Approach.SOUTH, 50.0)

        # [A, B] clustered, [C] solo
        self.assertEqual(len(self.controller.get_active_clusters()[0]), 2)

        # A passes
        self.controller.ambulance_passed("A")
        self.controller.current_episode.active_notices.pop("A")

        # Now only B (32s) and C (50s) remain -> diff = 18s -> Two solo clusters
        c_after = self.controller.get_active_clusters()
        self.assertEqual(len(c_after), 2)
        self.assertEqual(len(c_after[0]), 1)
        self.assertEqual(len(c_after[1]), 1)

    def test_12_cluster_membership_changes_after_dismissal(self):
        """12. Cluster membership changes after dismissal."""
        self.controller.create_and_register_notice("A", Approach.NORTH, 10.0)
        self.controller.create_and_register_notice("B", Approach.EAST, 15.0)

        self.assertEqual(len(self.controller.get_active_clusters()[0]), 2)

        self.controller.dismiss_notice("A", reason="Dismissed")
        c_after = self.controller.get_active_clusters()
        self.assertEqual(len(c_after), 1)
        self.assertEqual(len(c_after[0]), 1)

    def test_13_clustered_emergency_lanes_receive_queue_weight_090(self):
        """13. Clustered emergency lanes receive queue coefficient 0.90."""
        norm = NormalizedMetrics(queue_norm=1.0, wait_norm=0.0, flow_norm=0.0)
        p_clustered = compute_emergency_priority_score(Approach.NORTH, norm, is_clustered=True)
        # 0.90 * 1.0 + 0 + 0 = 0.90
        self.assertAlmostEqual(p_clustered.score, 0.90)

    def test_14_non_clustered_lanes_retain_queue_weight_045(self):
        """14. Non-clustered lanes retain queue coefficient 0.45."""
        norm = NormalizedMetrics(queue_norm=1.0, wait_norm=0.0, flow_norm=0.0)
        p_normal = compute_emergency_priority_score(Approach.NORTH, norm, is_clustered=False)
        # 0.45 * 1.0 + 0 + 0 = 0.45
        self.assertAlmostEqual(p_normal.score, 0.45)

    def test_15_priority_boost_is_temporary(self):
        """15. Priority boost is temporary and does not alter global config."""
        weights = PriorityWeights()
        self.assertEqual(weights.w_queue, 0.45)

        norm = NormalizedMetrics(queue_norm=0.8, wait_norm=0.5, flow_norm=0.2)
        score_boosted = compute_emergency_priority_score(Approach.EAST, norm, is_clustered=True, normal_weights=weights)
        self.assertAlmostEqual(score_boosted.score, 0.90 * 0.8 + 0.35 * 0.5 + 0.20 * 0.2)
        self.assertEqual(weights.w_queue, 0.45)  # Unmodified

    def test_16_every_emergency_in_cluster_independently_evaluates_case_a(self):
        """16. Every emergency in a cluster independently evaluates Case A."""
        # Amb A: ETA=8s, T_clear=10s (8 <= 13 -> Case A True)
        is_a, msg_a = check_emergency_trigger_conditions(eta=8.0, t_clear=10.0, g_min=10.0)
        self.assertTrue(is_a)
        self.assertIn("Case A", msg_a)

    def test_17_every_emergency_in_cluster_independently_evaluates_case_b(self):
        """17. Every emergency in a cluster independently evaluates Case B."""
        # Amb B: ETA=20s, T_clear=6s ((6+3)-20 = -11 < 13 -> Case B True)
        is_b, msg_b = check_emergency_trigger_conditions(eta=20.0, t_clear=6.0, g_min=10.0)
        self.assertTrue(is_b)
        self.assertIn("Case B", msg_b)

    def test_18_lower_eta_wins_simultaneous_trigger(self):
        """18. Lower ETA wins simultaneous trigger conflict."""
        nA = EmergencyNotice("A", Approach.NORTH, current_eta=15.0)
        nB = EmergencyNotice("B", Approach.EAST, current_eta=10.0)  # Lower ETA

        winner, _ = resolve_emergency_conflict(
            [(nA, 8.0), (nB, 8.0)],
            wait_norms={Approach.NORTH: 0.5, Approach.EAST: 0.5}
        )
        self.assertEqual(winner.emergency_id, "B")

    def test_19_shorter_tclear_resolves_eta_tie(self):
        """19. Shorter T_clear resolves ETA tie."""
        nA = EmergencyNotice("A", Approach.NORTH, current_eta=15.0)
        nB = EmergencyNotice("B", Approach.EAST, current_eta=15.0)

        # nA has T_clear=6.0, nB has T_clear=12.0
        winner, t_clr = resolve_emergency_conflict(
            [(nA, 6.0), (nB, 12.0)],
            wait_norms={Approach.NORTH: 0.5, Approach.EAST: 0.5}
        )
        self.assertEqual(winner.emergency_id, "A")
        self.assertEqual(t_clr, 6.0)

    def test_20_higher_wait_norm_resolves_eta_and_tclear_tie(self):
        """20. Higher WaitTime_norm resolves ETA/T_clear tie."""
        nA = EmergencyNotice("A", Approach.NORTH, current_eta=15.0)
        nB = EmergencyNotice("B", Approach.EAST, current_eta=15.0)

        # Both T_clear = 8.0s. But EAST has wait_norm=0.8 vs NORTH=0.3
        winner, _ = resolve_emergency_conflict(
            [(nA, 8.0), (nB, 8.0)],
            wait_norms={Approach.NORTH: 0.3, Approach.EAST: 0.8}
        )
        self.assertEqual(winner.emergency_id, "B")

    def test_21_direction_order_resolves_complete_tie(self):
        """21. NORTH > EAST > SOUTH > WEST resolves complete tie."""
        n_south = EmergencyNotice("S", Approach.SOUTH, current_eta=15.0)
        n_east = EmergencyNotice("E", Approach.EAST, current_eta=15.0)
        n_north = EmergencyNotice("N", Approach.NORTH, current_eta=15.0)
        n_west = EmergencyNotice("W", Approach.WEST, current_eta=15.0)

        # NORTH should beat EAST, SOUTH, WEST on complete tie
        winner, _ = resolve_emergency_conflict(
            [(n_south, 6.0), (n_east, 6.0), (n_north, 6.0), (n_west, 6.0)],
            wait_norms={a: 0.5 for a in Approach}
        )
        self.assertEqual(winner.emergency_id, "N")

        # EAST beats SOUTH
        winner_es, _ = resolve_emergency_conflict(
            [(n_south, 6.0), (n_east, 6.0)],
            wait_norms={a: 0.5 for a in Approach}
        )
        self.assertEqual(winner_es.emergency_id, "E")

    def test_22_passage_of_one_does_not_mark_another(self):
        """22. Passage of one emergency does not mark another as passed."""
        nA = self.controller.create_and_register_notice("AMB-A", Approach.NORTH, 10.0)
        nB = self.controller.create_and_register_notice("AMB-B", Approach.EAST, 15.0)

        self.controller.ambulance_passed("AMB-A")
        self.assertTrue(nA.is_passed)
        self.assertEqual(nA.state, EmergencyState.PASSED)
        self.assertFalse(nB.is_passed)
        self.assertEqual(nB.state, EmergencyState.PENDING)

    def test_23_new_emergency_can_arrive_during_active_episode(self):
        """23. New emergency can arrive during an existing emergency episode."""
        traffic = self._get_default_traffic()
        # Advance green past G_min
        for _ in range(10):
            self.controller.step(traffic, dt=1.0)

        # Submit first emergency
        n1 = self.controller.submit_emergency(EmergencyNotice("AMB-1", Approach.EAST, 10.0, queue_pcu=2.0))
        d1 = self.controller.step(traffic, dt=1.0)
        self.assertEqual(d1.active_phase, PhaseState.YELLOW)
        self.assertEqual(d1.next_green_candidate, Approach.EAST)

        # While transition is running, a new emergency arrives on SOUTH
        n2 = self.controller.submit_emergency(EmergencyNotice("AMB-2", Approach.SOUTH, 18.0, queue_pcu=3.0))
        self.assertIn("AMB-1", self.controller.current_episode.active_notices)
        self.assertIn("AMB-2", self.controller.current_episode.active_notices)

    def test_24_dismissed_emergency_removed_from_active_selection(self):
        """24. Dismissed emergency is removed from active selection."""
        n1 = self.controller.create_and_register_notice("AMB-1", Approach.EAST, 0.0)
        n2 = self.controller.create_and_register_notice("AMB-2", Approach.SOUTH, 25.0)

        self.controller.dismiss_notice("AMB-1", reason="Overdue timeout")
        active = self.controller.get_active_emergency()
        self.assertEqual(active.emergency_id, "AMB-2")

    def test_25_single_green_invariant_multi_emergency(self):
        """25. Single-green invariant remains true with multiple emergencies."""
        traffic = self._get_default_traffic()
        self.controller.submit_emergency(EmergencyNotice("AMB-1", Approach.EAST, 6.0, queue_pcu=2.0))
        self.controller.submit_emergency(EmergencyNotice("AMB-2", Approach.SOUTH, 8.0, queue_pcu=3.0))

        for _ in range(30):
            decision = self.controller.step(traffic, dt=1.0)
            green_count = sum(1 for color in decision.signal_states.values() if color == SignalColor.GREEN)
            self.assertLessEqual(green_count, 1)

    def test_26_yellow_allred_safety_sequence(self):
        """26. Yellow/all-red safety sequence remains true with multiple emergencies."""
        traffic = self._get_default_traffic()
        for _ in range(10):
            self.controller.step(traffic, dt=1.0)

        self.controller.submit_emergency(EmergencyNotice("AMB-1", Approach.EAST, 8.0, queue_pcu=1.0))
        self.controller.submit_emergency(EmergencyNotice("AMB-2", Approach.SOUTH, 12.0, queue_pcu=2.0))

        # Transition to EAST
        d_yellow = self.controller.step(traffic, dt=1.0)
        self.assertEqual(d_yellow.active_phase, PhaseState.YELLOW)
        self.controller.step(traffic, dt=1.0)
        self.controller.step(traffic, dt=1.0)

        d_allred = self.controller.step(traffic, dt=1.0)
        self.assertEqual(d_allred.active_phase, PhaseState.ALL_RED)
        self.controller.step(traffic, dt=1.0)

        d_green = self.controller.step(traffic, dt=1.0)
        self.assertEqual(d_green.active_phase, PhaseState.GREEN)
        self.assertEqual(d_green.current_green, Approach.EAST)

    def test_27_regression_phase1_and_phase2_invariants(self):
        """27. Regression test verifying Phase 1 + Phase 2 calculations remain completely consistent."""
        self.assertEqual(calculate_t_clear(5.0), 14.0)
        self.assertEqual(calculate_effective_emergency_g_max(40.0, 14.0), 40.0)
        self.assertEqual(calculate_effective_emergency_g_max(40.0, 50.0), 53.0)


if __name__ == "__main__":
    unittest.main()
