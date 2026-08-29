"""
integration_tests.py
Unit tests for Camera/Tracking Integration Layer (Phase 4A) and Vision Emergency Bridge (Phase 4B).
Tests strongly-typed event contracts, CameraIntegrationAdapter, EmergencyVisionBridge, validation rules,
live ETA calculation, line crossing passage, and multi-junction handoffs.
"""

import unittest
import time
from backend.models.traffic_schemas import ApproachEnum, MovementStateEnum
from backend.models.ambulance_schemas import AmbulanceStatusEnum
from backend.core.vision.tracker import TrackedVehicle
from backend.core.vision.emergency_bridge import EmergencyVisionBridge
from backend.decisionbackend.models import Approach, PhaseState, DirectionTraffic, SignalColor
from backend.decisionbackend.junction_config import JunctionConfig
from backend.core.control.emergency_orchestrator import EmergencyOrchestrator
from backend.decisionbackend.emergency.emergency_models import (
    EmergencyVehicleType,
    EmergencyState,
)
from backend.decisionbackend.emergency.emergency_controller import EmergencyController
from backend.decisionbackend.emergency.camera_events import (
    EmergencyDetectionEvent,
    EmergencyEtaUpdateEvent,
    EmergencyPassageEvent,
    DirectionalHandoffEvent,
)
from backend.decisionbackend.emergency.camera_interface import CameraIntegrationAdapter


class TestCameraIntegrationPhase4A(unittest.TestCase):
    """
    Phase 4A Integration layer tests.
    """

    def setUp(self):
        self.controller = EmergencyController()
        self.adapter = CameraIntegrationAdapter(controller=self.controller, junction_id="J-NORTH-1")

    def test_01_valid_emergency_detection_registers_emergency(self):
        """1. Valid emergency detection creates/registers the correct emergency."""
        event = EmergencyDetectionEvent(
            emergency_id="AMB-101",
            junction_id="J-NORTH-1",
            approach=Approach.NORTH,
            eta=15.0,
            vehicle_type=EmergencyVehicleType.AMBULANCE,
            lane_id="lane-N1"
        )
        res = self.adapter.on_emergency_detected(event)
        self.assertTrue(res)

        notice = self.controller.get_notice("AMB-101")
        self.assertIsNotNone(notice)
        self.assertEqual(notice.emergency_id, "AMB-101")
        self.assertEqual(notice.approach, Approach.NORTH)
        self.assertEqual(notice.current_eta, 15.0)
        self.assertEqual(notice.original_eta, 15.0)
        self.assertEqual(notice.vehicle_type, EmergencyVehicleType.AMBULANCE)
        self.assertEqual(notice.target_lane, "lane-N1")
        self.assertEqual(notice.state, EmergencyState.PENDING)

    def test_02_invalid_emergency_detection_rejected_safely(self):
        """2. Invalid emergency detection is rejected safely."""
        # Empty emergency_id
        e_empty = EmergencyDetectionEvent(
            emergency_id="",
            junction_id="J-01",
            approach=Approach.NORTH,
            eta=10.0
        )
        self.assertFalse(self.adapter.on_emergency_detected(e_empty))

        # Negative ETA
        e_neg_eta = EmergencyDetectionEvent(
            emergency_id="AMB-NEG",
            junction_id="J-01",
            approach=Approach.NORTH,
            eta=-5.0
        )
        self.assertFalse(self.adapter.on_emergency_detected(e_neg_eta))

        # Invalid confidence
        e_bad_conf = EmergencyDetectionEvent(
            emergency_id="AMB-CONF",
            junction_id="J-01",
            approach=Approach.NORTH,
            eta=10.0,
            confidence=1.5
        )
        self.assertFalse(self.adapter.on_emergency_detected(e_bad_conf))

        # Empty junction_id
        e_empty_j = EmergencyDetectionEvent(
            emergency_id="AMB-OK",
            junction_id="",
            approach=Approach.NORTH,
            eta=10.0
        )
        self.assertFalse(self.adapter.on_emergency_detected(e_empty_j))

    def test_03_eta_update_reaches_only_correct_emergency(self):
        """3. ETA update reaches only the correct emergency."""
        self.adapter.on_emergency_detected(EmergencyDetectionEvent("AMB-A", "J-01", Approach.NORTH, 20.0))
        self.adapter.on_emergency_detected(EmergencyDetectionEvent("AMB-B", "J-01", Approach.SOUTH, 30.0))

        update_event = EmergencyEtaUpdateEvent(
            emergency_id="AMB-A",
            junction_id="J-01",
            new_eta=12.0
        )
        res = self.adapter.on_emergency_eta_updated(update_event)
        self.assertTrue(res)

        self.assertEqual(self.controller.get_notice("AMB-A").current_eta, 12.0)
        self.assertEqual(self.controller.get_notice("AMB-B").current_eta, 30.0)

    def test_04_eta_update_does_not_modify_another_emergency(self):
        """4. ETA update does not modify another emergency notice."""
        self.adapter.on_emergency_detected(EmergencyDetectionEvent("AMB-1", "J-01", Approach.EAST, 25.0))
        self.adapter.on_emergency_detected(EmergencyDetectionEvent("AMB-2", "J-01", Approach.WEST, 40.0))

        self.adapter.on_emergency_eta_updated(EmergencyEtaUpdateEvent("AMB-1", "J-01", 10.0))
        self.assertEqual(self.controller.get_notice("AMB-2").current_eta, 40.0)
        self.assertEqual(self.controller.get_notice("AMB-1").current_eta, 10.0)

    def test_05_valid_passage_event_marks_correct_emergency_passed(self):
        """5. Valid passage event marks the correct emergency as passed."""
        self.adapter.on_emergency_detected(EmergencyDetectionEvent("AMB-P1", "J-01", Approach.EAST, 5.0))

        passage = EmergencyPassageEvent(
            emergency_id="AMB-P1",
            junction_id="J-01",
            approach=Approach.EAST,
            destination_approach=Approach.WEST,
            camera_id="cam-east-01"
        )
        res = self.adapter.on_emergency_passed(passage)
        self.assertTrue(res)

        notice = self.controller.get_notice("AMB-P1")
        self.assertTrue(notice.is_passed)
        self.assertEqual(notice.state, EmergencyState.PASSED)
        self.assertEqual(notice.destination_approach, Approach.WEST)

    def test_06_passage_event_for_unknown_emergency_rejected_safely(self):
        """6. Passage event for unknown emergency is rejected safely."""
        passage = EmergencyPassageEvent(
            emergency_id="AMB-UNKNOWN",
            junction_id="J-01"
        )
        res = self.adapter.on_emergency_passed(passage)
        self.assertFalse(res)

    def test_07_passage_event_for_a_cannot_mark_b_passed(self):
        """7. Passage event for emergency A cannot mark emergency B as passed."""
        self.adapter.on_emergency_detected(EmergencyDetectionEvent("AMB-A", "J-01", Approach.NORTH, 10.0))
        self.adapter.on_emergency_detected(EmergencyDetectionEvent("AMB-B", "J-01", Approach.SOUTH, 20.0))

        self.adapter.on_emergency_passed(EmergencyPassageEvent("AMB-A", "J-01"))
        self.assertTrue(self.controller.get_notice("AMB-A").is_passed)
        self.assertFalse(self.controller.get_notice("AMB-B").is_passed)
        self.assertEqual(self.controller.get_notice("AMB-B").state, EmergencyState.PENDING)

    def test_08_directional_handoff_preserves_emergency_id(self):
        """8. Directional handoff preserves the correct emergency_id."""
        handoff = DirectionalHandoffEvent(
            emergency_id="AMB-HANDOFF",
            source_junction_id="J-01",
            outgoing_approach=Approach.EAST
        )
        res = self.adapter.on_direction_handoff(handoff)
        self.assertTrue(res)
        self.assertEqual(len(self.adapter.handoff_log), 1)
        self.assertEqual(self.adapter.handoff_log[0].emergency_id, "AMB-HANDOFF")

    def test_09_directional_handoff_preserves_source_and_destination(self):
        """9. Directional handoff preserves source and destination junction information."""
        handoff = DirectionalHandoffEvent(
            emergency_id="AMB-HANDOFF-2",
            source_junction_id="J-01",
            outgoing_approach=Approach.NORTH,
            destination_junction_id="J-02",
            next_approach=Approach.SOUTH,
            next_junction_eta=45.0
        )
        res = self.adapter.on_direction_handoff(handoff)
        self.assertTrue(res)
        h = self.adapter.handoff_log[0]
        self.assertEqual(h.source_junction_id, "J-01")
        self.assertEqual(h.destination_junction_id, "J-02")
        self.assertEqual(h.outgoing_approach, Approach.NORTH)
        self.assertEqual(h.next_approach, Approach.SOUTH)
        self.assertEqual(h.next_junction_eta, 45.0)

    def test_10_multiple_emergency_events_remain_independent(self):
        """10. Multiple emergency events remain independent."""
        e1 = EmergencyDetectionEvent("AMB-1", "J-01", Approach.NORTH, 10.0)
        e2 = EmergencyDetectionEvent("AMB-2", "J-01", Approach.EAST, 20.0)
        e3 = EmergencyDetectionEvent("AMB-3", "J-01", Approach.SOUTH, 30.0)

        self.adapter.on_emergency_detected(e1)
        self.adapter.on_emergency_detected(e2)
        self.adapter.on_emergency_detected(e3)

        self.assertEqual(len(self.controller.current_episode.active_notices), 3)

        self.adapter.on_emergency_eta_updated(EmergencyEtaUpdateEvent("AMB-2", "J-01", 15.0))
        self.assertEqual(self.controller.get_notice("AMB-1").current_eta, 10.0)
        self.assertEqual(self.controller.get_notice("AMB-2").current_eta, 15.0)
        self.assertEqual(self.controller.get_notice("AMB-3").current_eta, 30.0)

    def test_11_event_timestamps_and_ordering_handled_deterministically(self):
        """11. Event timestamps and ordering are handled deterministically in event log."""
        t1 = 1000.0
        t2 = 1005.0
        t3 = 1010.0

        e1 = EmergencyDetectionEvent("AMB-T1", "J-01", Approach.NORTH, 20.0, timestamp=t1)
        e2 = EmergencyEtaUpdateEvent("AMB-T1", "J-01", 15.0, timestamp=t2)
        e3 = EmergencyPassageEvent("AMB-T1", "J-01", timestamp=t3)

        self.adapter.on_emergency_detected(e1)
        self.adapter.on_emergency_eta_updated(e2)
        self.adapter.on_emergency_passed(e3)

        self.assertEqual(len(self.adapter.event_log), 3)
        self.assertEqual(self.adapter.event_log[0].timestamp, t1)
        self.assertEqual(self.adapter.event_log[1].timestamp, t2)
        self.assertEqual(self.adapter.event_log[2].timestamp, t3)

    def test_12_downstream_automatic_handoff_dispatch(self):
        """12. Downstream junction registration automatically dispatches detection to next junction."""
        downstream_controller = EmergencyController()
        downstream_adapter = CameraIntegrationAdapter(controller=downstream_controller, junction_id="J-02")
        self.adapter.register_downstream_adapter("J-02", downstream_adapter)

        handoff = DirectionalHandoffEvent(
            emergency_id="AMB-AUTO-HANDOFF",
            source_junction_id="J-01",
            outgoing_approach=Approach.NORTH,
            destination_junction_id="J-02",
            next_approach=Approach.SOUTH,
            next_junction_eta=35.0
        )
        self.adapter.on_direction_handoff(handoff)

        downstream_notice = downstream_controller.get_notice("AMB-AUTO-HANDOFF")
        self.assertIsNotNone(downstream_notice)
        self.assertEqual(downstream_notice.approach, Approach.SOUTH)
        self.assertEqual(downstream_notice.current_eta, 35.0)


class TestVisionEmergencyBridgePhase4B(unittest.TestCase):
    """
    Phase 4B Vision-to-Emergency Bridge unit tests.
    """

    def setUp(self):
        self.controller = EmergencyController()
        self.adapter = CameraIntegrationAdapter(controller=self.controller, junction_id="J-01")
        self.bridge = EmergencyVisionBridge(adapter=self.adapter, junction_id="J-01", approach=ApproachEnum.NORTH)

    def _create_vehicle(
        self,
        track_id: int = 10,
        center: tuple = (300.0, 100.0),
        speed_px: float = 10.0,
        crossed: bool = False,
        direction: MovementStateEnum = MovementStateEnum.INCOMING
    ) -> TrackedVehicle:
        return TrackedVehicle(
            track_id=track_id,
            xyxy=[center[0] - 20, center[1] - 40, center[0] + 20, center[1] + 40],
            confidence=0.95,
            class_id=2,
            class_name="car",
            center=center,
            speed_px=speed_px,
            crossed_counting_line=crossed,
            direction=direction
        )

    def test_01_emergency_mission_associated_with_tracked_vehicle(self):
        """1. Emergency mission can be associated with a tracked vehicle."""
        self.bridge.associate_mission(track_id=12, emergency_id="AMB-MISSION-12")
        self.assertEqual(self.bridge.get_associated_emergency(12), "AMB-MISSION-12")
        self.assertIsNone(self.bridge.get_associated_emergency(999))

    def test_02_same_emergency_does_not_generate_repeated_detections(self):
        """2. Same emergency does not generate repeated detection events every frame (deduplication)."""
        self.bridge.associate_mission(track_id=5, emergency_id="AMB-5")
        v = self._create_vehicle(track_id=5, center=(300.0, 100.0), speed_px=8.0)

        # Frame 1 -> Detection emitted
        self.bridge.process_frame([v], counting_line_config={"p1": [0.1, 0.5], "p2": [0.9, 0.5]}, frame_width=640, frame_height=480)
        self.assertEqual(len(self.adapter.event_log), 1)
        self.assertIsInstance(self.adapter.event_log[0], EmergencyDetectionEvent)

        # Frame 2 -> Same vehicle, should not emit another detection event
        self.bridge.process_frame([v], counting_line_config={"p1": [0.1, 0.5], "p2": [0.9, 0.5]}, frame_width=640, frame_height=480)
        self.assertEqual(sum(1 for e in self.adapter.event_log if isinstance(e, EmergencyDetectionEvent)), 1)

    def test_03_correct_emergency_id_preserved_in_controller(self):
        """3. Correct emergency_id is preserved when registered into EmergencyController."""
        self.bridge.associate_mission(track_id=7, emergency_id="AMB-PRESERVED-ID")
        v = self._create_vehicle(track_id=7, center=(200.0, 50.0), speed_px=10.0)

        self.bridge.process_frame([v], counting_line_config={"p1": [0.1, 0.5], "p2": [0.9, 0.5]}, frame_width=640, frame_height=480)
        notice = self.controller.get_notice("AMB-PRESERVED-ID")
        self.assertIsNotNone(notice)
        self.assertEqual(notice.emergency_id, "AMB-PRESERVED-ID")

    def test_04_live_eta_calculated_from_position_and_speed(self):
        """4. Live ETA is calculated from tracked position and speed (distance_px / (speed_px * fps))."""
        # Distance = 200px, Speed = 10px/frame, FPS = 20 -> speed = 200px/s -> ETA = 1.0s
        eta = EmergencyVisionBridge.calculate_live_eta(distance_px=200.0, speed_px=10.0, fps=20.0)
        self.assertEqual(eta, 1.0)

        # Distance = 500px, Speed = 5px/frame, FPS = 25 -> speed = 125px/s -> ETA = 4.0s
        eta2 = EmergencyVisionBridge.calculate_live_eta(distance_px=500.0, speed_px=5.0, fps=25.0)
        self.assertEqual(eta2, 4.0)

    def test_05_zero_and_invalid_speed_handled_safely(self):
        """5. Invalid/zero speed does not cause division-by-zero or crash."""
        eta_zero = EmergencyVisionBridge.calculate_live_eta(distance_px=100.0, speed_px=0.0, fps=25.0)
        self.assertGreater(eta_zero, 0.0)

        eta_neg = EmergencyVisionBridge.calculate_live_eta(distance_px=100.0, speed_px=-2.0, fps=25.0)
        self.assertGreater(eta_neg, 0.0)

    def test_06_eta_update_reaches_only_correct_emergency(self):
        """6. Live ETA update reaches only the correct emergency."""
        self.bridge.associate_mission(track_id=1, emergency_id="AMB-1")
        self.bridge.associate_mission(track_id=2, emergency_id="AMB-2")

        v1 = self._create_vehicle(track_id=1, center=(320.0, 50.0), speed_px=10.0)
        v2 = self._create_vehicle(track_id=2, center=(320.0, 80.0), speed_px=5.0)

        # Initial detection
        self.bridge.process_frame([v1, v2], counting_line_config={"p1": [0.1, 0.5], "p2": [0.9, 0.5]}, frame_width=640, frame_height=480, fps=25.0)

        # Advance v1 closer (center y: 50 -> 180) -> ETA decreases meaningfully
        v1_closer = self._create_vehicle(track_id=1, center=(320.0, 180.0), speed_px=10.0)
        self.bridge.process_frame([v1_closer, v2], counting_line_config={"p1": [0.1, 0.5], "p2": [0.9, 0.5]}, frame_width=640, frame_height=480, fps=25.0)

        notice1 = self.controller.get_notice("AMB-1")
        notice2 = self.controller.get_notice("AMB-2")
        self.assertIsNotNone(notice1)
        self.assertIsNotNone(notice2)
        self.assertNotEqual(notice1.current_eta, notice2.current_eta)

    def test_07_counting_line_crossing_detects_passage(self):
        """7. Counting-line crossing detects emergency passage and emits EmergencyPassageEvent."""
        self.bridge.associate_mission(track_id=8, emergency_id="AMB-CROSS")
        v = self._create_vehicle(track_id=8, center=(300.0, 100.0), crossed=False)
        self.bridge.process_frame([v], counting_line_config=None, frame_width=640, frame_height=480)

        self.assertFalse(self.controller.get_notice("AMB-CROSS").is_passed)

        # Vehicle crosses counting line
        v_crossed = self._create_vehicle(track_id=8, center=(300.0, 260.0), crossed=True)
        self.bridge.process_frame([v_crossed], counting_line_config=None, frame_width=640, frame_height=480)

        self.assertTrue(self.controller.get_notice("AMB-CROSS").is_passed)
        self.assertEqual(self.controller.get_notice("AMB-CROSS").state, EmergencyState.PASSED)

    def test_08_passage_event_emitted_only_once(self):
        """8. Passage event is emitted only once for the same crossing across multiple frames."""
        self.bridge.associate_mission(track_id=9, emergency_id="AMB-PASS-ONCE")
        v_crossed = self._create_vehicle(track_id=9, center=(300.0, 260.0), crossed=True)

        self.bridge.process_frame([v_crossed], counting_line_config=None, frame_width=640, frame_height=480)
        self.bridge.process_frame([v_crossed], counting_line_config=None, frame_width=640, frame_height=480)
        self.bridge.process_frame([v_crossed], counting_line_config=None, frame_width=640, frame_height=480)

        passage_events = [e for e in self.adapter.event_log if isinstance(e, EmergencyPassageEvent)]
        self.assertEqual(len(passage_events), 1)

    def test_09_passage_event_contains_correct_emergency_id(self):
        """9. Passage event contains the exact matching emergency_id."""
        self.bridge.associate_mission(track_id=14, emergency_id="AMB-EXACT-ID")
        v_crossed = self._create_vehicle(track_id=14, center=(300.0, 260.0), crossed=True)
        self.bridge.process_frame([v_crossed], counting_line_config=None, frame_width=640, frame_height=480)

        passage_events = [e for e in self.adapter.event_log if isinstance(e, EmergencyPassageEvent)]
        self.assertEqual(passage_events[0].emergency_id, "AMB-EXACT-ID")

    def test_10_outgoing_movement_produces_directional_handoff(self):
        """10. Outgoing movement trajectory produces a directional handoff."""
        self.bridge.associate_mission(track_id=20, emergency_id="AMB-HANDOFF-1")
        # Step 1: Crossed line
        v_passed = self._create_vehicle(track_id=20, center=(300.0, 260.0), crossed=True, direction=MovementStateEnum.INCOMING)
        self.bridge.process_frame([v_passed], counting_line_config=None, frame_width=640, frame_height=480)

        # Step 2: Now OUTGOING
        v_outgoing = self._create_vehicle(track_id=20, center=(300.0, 350.0), crossed=True, direction=MovementStateEnum.OUTGOING)
        self.bridge.process_frame([v_outgoing], counting_line_config=None, frame_width=640, frame_height=480)

        handoff_events = [e for e in self.adapter.event_log if isinstance(e, DirectionalHandoffEvent)]
        self.assertEqual(len(handoff_events), 1)
        self.assertEqual(handoff_events[0].emergency_id, "AMB-HANDOFF-1")

    def test_11_directional_handoff_preserves_emergency_id_and_source_junction(self):
        """11. Directional handoff preserves emergency_id and source junction ID."""
        self.bridge.associate_mission(track_id=22, emergency_id="AMB-JUNC-TEST")
        v = self._create_vehicle(track_id=22, center=(300.0, 300.0), crossed=True, direction=MovementStateEnum.OUTGOING)

        self.bridge.process_frame([v], counting_line_config=None, frame_width=640, frame_height=480)
        h = self.adapter.handoff_log[-1]
        self.assertEqual(h.emergency_id, "AMB-JUNC-TEST")
        self.assertEqual(h.source_junction_id, "J-01")

    def test_12_destination_junction_obtained_from_corridor_map(self):
        """12. Destination junction and downstream approach are obtained from CORRIDOR_MAP."""
        # J-01 connects to J-02 via East Arterial Corridor B, J-03 via North Blvd, J-05 via West Linkway
        dest_j, next_app = EmergencyVisionBridge._resolve_downstream_corridor("J-01", ApproachEnum.NORTH)
        self.assertIsNotNone(dest_j)
        self.assertIsNotNone(next_app)

    def test_13_normal_non_emergency_vehicles_ignored_by_bridge(self):
        """13. Normal non-emergency vehicles are ignored by the bridge without emitting emergency events."""
        v_normal1 = self._create_vehicle(track_id=101, center=(100.0, 100.0), speed_px=12.0)
        v_normal2 = self._create_vehicle(track_id=102, center=(200.0, 200.0), speed_px=8.0, crossed=True, direction=MovementStateEnum.OUTGOING)

        self.bridge.process_frame([v_normal1, v_normal2], counting_line_config=None, frame_width=640, frame_height=480)
        self.assertEqual(len(self.adapter.event_log), 0)
        self.assertEqual(len(self.controller.current_episode.active_notices), 0)


class TestEmergencyOrchestratorPhase5(unittest.TestCase):
    """
    Phase 5 Multi-Junction Emergency Orchestrator unit tests.
    """

    def setUp(self):
        from backend.core.control.emergency_orchestrator import EmergencyOrchestrator
        self.orchestrator = EmergencyOrchestrator()

    def test_01_mission_registers_successfully_with_orchestrator(self):
        """1. Mission registers successfully with orchestrator."""
        ctx = self.orchestrator.register_mission(
            mission_id="MSN-101",
            vehicle_id="DL-01-AMB-101",
            origin_junction_id="J-04",
            destination_junction_id="J-02"
        )
        self.assertEqual(ctx.mission_id, "MSN-101")
        self.assertEqual(ctx.vehicle_id, "DL-01-AMB-101")
        self.assertEqual(ctx.current_junction_id, "J-04")
        self.assertFalse(ctx.is_completed)
        self.assertGreater(len(ctx.route_nodes), 0)

    def test_02_emergency_mission_associated_with_first_junction_camera(self):
        """2. Emergency mission gets associated with first-junction camera."""
        self.orchestrator.register_mission("MSN-102", "DL-02-AMB-102", "J-04", "J-02")
        res = self.orchestrator.associate_camera_track(
            junction_id="J-04",
            approach=ApproachEnum.SOUTH,
            track_id=42,
            emergency_id_or_vehicle="DL-02-AMB-102"
        )
        self.assertTrue(res)

        ctx = self.orchestrator.get_mission_context("DL-02-AMB-102")
        self.assertEqual(ctx.current_track_id, 42)
        self.assertEqual(ctx.current_junction_id, "J-04")

    def test_03_persistent_emergency_id_maintained(self):
        """3. Persistent emergency_id is maintained across queries."""
        self.orchestrator.register_mission("MSN-103", "DL-03-AMB-103", "J-01", "J-02")
        ctx = self.orchestrator.get_mission_context("DL-03-AMB-103")
        self.assertEqual(ctx.emergency_id, "DL-03-AMB-103")
        self.assertEqual(ctx.vehicle_id, "DL-03-AMB-103")

    def test_04_camera_local_track_id_allowed_to_change_between_junctions(self):
        """4. Camera-local track_id is allowed to change between junctions."""
        self.orchestrator.register_mission("MSN-104", "DL-04-AMB-104", "J-04", "J-02")

        # Junction J-04: track_id = 42
        self.orchestrator.associate_camera_track("J-04", ApproachEnum.SOUTH, 42, "DL-04-AMB-104")
        ctx = self.orchestrator.get_mission_context("DL-04-AMB-104")
        self.assertEqual(ctx.current_track_id, 42)

        # Passage at J-04 -> track_id reset
        self.orchestrator.on_emergency_passed_junction("J-04", "DL-04-AMB-104")
        self.assertIsNone(ctx.current_track_id)

        # Junction J-01: new track_id = 17
        self.orchestrator.associate_camera_track("J-01", ApproachEnum.NORTH, 17, "DL-04-AMB-104")
        self.assertEqual(ctx.current_track_id, 17)
        self.assertEqual(ctx.current_junction_id, "J-01")

    def test_05_detection_reaches_correct_junction_controller(self):
        """5. Detection reaches correct junction controller."""
        self.orchestrator.register_mission("MSN-105", "DL-05-AMB-105", "J-04", "J-02")
        j4_ctrl = self.orchestrator.get_or_create_junction_controller("J-04")
        notice = j4_ctrl.get_notice("DL-05-AMB-105")
        self.assertIsNotNone(notice)
        self.assertEqual(notice.emergency_id, "DL-05-AMB-105")

    def test_06_detection_does_not_affect_unrelated_junction_controllers(self):
        """6. Detection does not affect unrelated junction controllers."""
        self.orchestrator.register_mission("MSN-106", "DL-06-AMB-106", "J-04", "J-02")
        j2_ctrl = self.orchestrator.get_or_create_junction_controller("J-02")
        # J-02 controller should not have active notice for MSN-106 yet
        self.assertIsNone(j2_ctrl.get_notice("DL-06-AMB-106"))

    def test_07_eta_update_reaches_correct_emergency(self):
        """7. ETA update reaches correct emergency at the active junction."""
        self.orchestrator.register_mission("MSN-107", "DL-07-AMB-107", "J-04", "J-02")
        j4_adapter = self.orchestrator.get_or_create_junction_adapter("J-04")
        res = j4_adapter.on_emergency_eta_updated(
            EmergencyEtaUpdateEvent("DL-07-AMB-107", "J-04", 12.0)
        )
        self.assertTrue(res)
        j4_ctrl = self.orchestrator.get_or_create_junction_controller("J-04")
        self.assertEqual(j4_ctrl.get_notice("DL-07-AMB-107").current_eta, 12.0)

    def test_08_passage_completes_only_current_junction_emergency(self):
        """8. Passage completes only the current junction emergency."""
        self.orchestrator.register_mission("MSN-108", "DL-08-AMB-108", "J-04", "J-02")
        self.orchestrator.on_emergency_passed_junction("J-04", "DL-08-AMB-108")

        j4_ctrl = self.orchestrator.get_or_create_junction_controller("J-04")
        self.assertTrue(j4_ctrl.get_notice("DL-08-AMB-108").is_passed)

    def test_09_passage_does_not_globally_complete_mission(self):
        """9. Passage at non-final junction does not globally complete the mission."""
        self.orchestrator.register_mission("MSN-109", "DL-09-AMB-109", "J-04", "J-02")
        # J-04 -> J-02 (J-04 is first of 2 nodes)
        self.orchestrator.on_emergency_passed_junction("J-04", "DL-09-AMB-109")

        ctx = self.orchestrator.get_mission_context("DL-09-AMB-109")
        self.assertFalse(ctx.is_completed)
        self.assertEqual(ctx.status, AmbulanceStatusEnum.TRANSIT_TO_HOSPITAL)
        self.assertEqual(ctx.current_junction_id, "J-02")

    def test_10_directional_handoff_resolves_correct_downstream_junction(self):
        """10. Directional handoff resolves correct downstream junction."""
        self.orchestrator.register_mission("MSN-110", "DL-10-AMB-110", "J-04", "J-02")

        handoff = DirectionalHandoffEvent(
            emergency_id="DL-10-AMB-110",
            source_junction_id="J-04",
            outgoing_approach=Approach.NORTH,
            destination_junction_id="J-01",
            next_approach=Approach.SOUTH,
            next_junction_eta=40.0
        )
        res = self.orchestrator.on_directional_handoff(handoff)
        self.assertTrue(res)

        ctx = self.orchestrator.get_mission_context("DL-10-AMB-110")
        self.assertEqual(ctx.current_junction_id, "J-01")
        self.assertEqual(ctx.current_approach, Approach.SOUTH)

    def test_11_directional_handoff_preserves_emergency_id(self):
        """11. Directional handoff preserves emergency_id."""
        self.orchestrator.register_mission("MSN-111", "DL-11-AMB-111", "J-01", "J-02")
        handoff = DirectionalHandoffEvent(
            emergency_id="DL-11-AMB-111",
            source_junction_id="J-01",
            outgoing_approach=Approach.EAST,
            destination_junction_id="J-02",
            next_approach=Approach.WEST,
            next_junction_eta=30.0
        )
        self.orchestrator.on_directional_handoff(handoff)
        j2_ctrl = self.orchestrator.get_or_create_junction_controller("J-02")
        self.assertIsNotNone(j2_ctrl.get_notice("DL-11-AMB-111"))

    def test_12_downstream_emergency_notice_created_correctly(self):
        """12. Downstream emergency notice is created correctly."""
        self.orchestrator.register_mission("MSN-112", "DL-12-AMB-112", "J-01", "J-02")
        handoff = DirectionalHandoffEvent(
            emergency_id="DL-12-AMB-112",
            source_junction_id="J-01",
            outgoing_approach=Approach.EAST,
            destination_junction_id="J-02",
            next_approach=Approach.WEST,
            next_junction_eta=25.0
        )
        self.orchestrator.on_directional_handoff(handoff)

        j2_ctrl = self.orchestrator.get_or_create_junction_controller("J-02")
        notice = j2_ctrl.get_notice("DL-12-AMB-112")
        self.assertIsNotNone(notice)
        self.assertEqual(notice.approach, Approach.WEST)
        self.assertEqual(notice.current_eta, 25.0)

    def test_13_downstream_eta_independently_established(self):
        """13. Downstream ETA is independently established and can be updated by downstream camera."""
        self.orchestrator.register_mission("MSN-113", "DL-13-AMB-113", "J-01", "J-02")
        handoff = DirectionalHandoffEvent(
            emergency_id="DL-13-AMB-113",
            source_junction_id="J-01",
            outgoing_approach=Approach.EAST,
            destination_junction_id="J-02",
            next_approach=Approach.WEST,
            next_junction_eta=50.0
        )
        self.orchestrator.on_directional_handoff(handoff)

        # Downstream camera at J-02 observes vehicle with live speed -> updates ETA to 18.0s
        j2_adapter = self.orchestrator.get_or_create_junction_adapter("J-02")
        j2_adapter.on_emergency_eta_updated(EmergencyEtaUpdateEvent("DL-13-AMB-113", "J-02", 18.0))

        j2_ctrl = self.orchestrator.get_or_create_junction_controller("J-02")
        self.assertEqual(j2_ctrl.get_notice("DL-13-AMB-113").current_eta, 18.0)

    def test_14_old_track_id_not_incorrectly_reused_downstream(self):
        """14. Old track_id is not incorrectly reused downstream."""
        self.orchestrator.register_mission("MSN-114", "DL-14-AMB-114", "J-04", "J-02")
        self.orchestrator.associate_camera_track("J-04", ApproachEnum.SOUTH, 99, "DL-14-AMB-114")

        # Handoff to J-01
        handoff = DirectionalHandoffEvent(
            emergency_id="DL-14-AMB-114",
            source_junction_id="J-04",
            outgoing_approach=Approach.NORTH,
            destination_junction_id="J-01",
            next_approach=Approach.SOUTH
        )
        self.orchestrator.on_directional_handoff(handoff)

        ctx = self.orchestrator.get_mission_context("DL-14-AMB-114")
        self.assertIsNone(ctx.current_track_id)  # Cleared

    def test_15_multiple_emergencies_at_different_junctions_remain_independent(self):
        """15. Multiple emergencies at different junctions remain independent."""
        self.orchestrator.register_mission("MSN-A", "AMB-A", "J-01", "J-02")
        self.orchestrator.register_mission("MSN-B", "AMB-B", "J-03", "J-05")

        j1_ctrl = self.orchestrator.get_or_create_junction_controller("J-01")
        j3_ctrl = self.orchestrator.get_or_create_junction_controller("J-03")

        self.assertIsNotNone(j1_ctrl.get_notice("AMB-A"))
        self.assertIsNone(j1_ctrl.get_notice("AMB-B"))

        self.assertIsNotNone(j3_ctrl.get_notice("AMB-B"))
        self.assertIsNone(j3_ctrl.get_notice("AMB-A"))

    def test_16_multiple_emergencies_at_same_junction_use_phase3_logic(self):
        """16. Multiple emergencies at the same junction use Phase 3 logic (clustering & conflict resolution)."""
        self.orchestrator.register_mission("MSN-X", "AMB-X", "J-01", "J-02")
        self.orchestrator.register_mission("MSN-Y", "AMB-Y", "J-01", "J-05")

        j1_ctrl = self.orchestrator.get_or_create_junction_controller("J-01")
        self.assertEqual(len(j1_ctrl.current_episode.active_notices), 2)

    def test_17_new_emergency_can_arrive_while_another_active(self):
        """17. New emergency can arrive while another emergency is active at a junction."""
        self.orchestrator.register_mission("MSN-1", "AMB-1", "J-01", "J-02")
        j1_ctrl = self.orchestrator.get_or_create_junction_controller("J-01")
        j1_ctrl.is_emergency_active = True

        # Second emergency arrives at J-01
        self.orchestrator.register_mission("MSN-2", "AMB-2", "J-01", "J-03")
        self.assertEqual(len(j1_ctrl.current_episode.active_notices), 2)

    def test_18_unresolved_corridor_handoff_does_not_crash_system(self):
        """18. Unresolved corridor/handoff does not crash the system."""
        self.orchestrator.register_mission("MSN-UNRESOLVED", "AMB-UNR", "J-01", "J-01", route_nodes=[])
        handoff = DirectionalHandoffEvent(
            emergency_id="AMB-UNR",
            source_junction_id="J-UNKNOWN",
            outgoing_approach=Approach.NORTH,
            destination_junction_id=None
        )
        res = self.orchestrator.on_directional_handoff(handoff)
        # Safe return False, no crash
        self.assertFalse(res)

    def test_19_invalid_emergency_events_do_not_corrupt_unrelated_emergencies(self):
        """19. Invalid emergency events do not corrupt unrelated emergencies."""
        self.orchestrator.register_mission("MSN-VALID", "AMB-VALID", "J-01", "J-02")
        j1_adapter = self.orchestrator.get_or_create_junction_adapter("J-01")

        # Invalid event with empty ID
        bad_event = EmergencyDetectionEvent("", "J-01", Approach.NORTH, 10.0)
        res = j1_adapter.on_emergency_detected(bad_event)
        self.assertFalse(res)

        # Valid notice unaffected
        j1_ctrl = self.orchestrator.get_or_create_junction_controller("J-01")
        self.assertIsNotNone(j1_ctrl.get_notice("AMB-VALID"))

    def test_20_final_route_completion_terminates_mission_state(self):
        """20. Final route completion correctly terminates mission-level emergency state."""
        ctx = self.orchestrator.register_mission("MSN-FINAL", "AMB-FINAL", "J-04", "J-02")
        # Nodes: J-04 -> J-01 -> J-02

        self.orchestrator.on_emergency_passed_junction("J-04", "AMB-FINAL")
        self.assertFalse(ctx.is_completed)

        self.orchestrator.on_emergency_passed_junction("J-01", "AMB-FINAL")
        self.assertFalse(ctx.is_completed)

        self.orchestrator.on_emergency_passed_junction("J-02", "AMB-FINAL")
        self.assertTrue(ctx.is_completed)
        self.assertEqual(ctx.status, AmbulanceStatusEnum.MISSION_ACCOMPLISHED)

    def test_21_emergency_lifecycle_transitions_deterministic(self):
        """21. Emergency lifecycle transitions are deterministic across multi-junction path."""
        ctx = self.orchestrator.register_mission("MSN-DET", "AMB-DET", "J-01", "J-02")
        self.assertEqual(ctx.status, AmbulanceStatusEnum.TRANSIT_TO_HOSPITAL)
        self.orchestrator.on_emergency_passed_junction("J-01", "AMB-DET")
        self.orchestrator.on_emergency_passed_junction("J-02", "AMB-DET")
        self.assertTrue(ctx.is_completed)


class TestUnannouncedEmergencyPhase5_1(unittest.TestCase):
    """
    Phase 5.1 Test Suite — Unannounced / Immediate Emergency Preemption (Case C).
    Covers deterministic unit tests TEST C1 through TEST C15.
    """

    def setUp(self):
        self.config = JunctionConfig()
        self.controller = EmergencyController(config=self.config, initial_green=Approach.NORTH)
        self.inputs = {
            Approach.NORTH: DirectionTraffic(direction=Approach.NORTH, vehicle_counts={"car": 8}, queue_pcu=8.0, flow_rate=3.0),
            Approach.SOUTH: DirectionTraffic(direction=Approach.SOUTH, vehicle_counts={"car": 4}, queue_pcu=4.0, flow_rate=2.0),
            Approach.EAST: DirectionTraffic(direction=Approach.EAST, vehicle_counts={"car": 3}, queue_pcu=3.0, flow_rate=1.0),
            Approach.WEST: DirectionTraffic(direction=Approach.WEST, vehicle_counts={"car": 2}, queue_pcu=2.0, flow_rate=1.0),
        }

    def test_c1_unannounced_emergency_detected_before_g_min(self):
        """TEST C1: Unannounced emergency detected before G_MIN keeps current green active."""
        # NORTH has been green for 3 seconds (< G_MIN of 10s)
        self.controller.state.time_in_phase = 3.0
        self.controller.create_and_register_notice(
            emergency_id="UNANNOUNCED-01",
            approach=Approach.EAST,
            initial_eta=15.0,
            pre_informed=False
        )

        decision = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(decision.active_phase, PhaseState.GREEN)
        self.assertEqual(decision.current_green, Approach.NORTH)
        self.assertFalse(decision.is_switch_in_progress)
        self.assertIn("Holding NORTH green for G_MIN", decision.reason)

    def test_c2_g_min_becomes_satisfied(self):
        """TEST C2: When G_MIN becomes satisfied, emergency preemption becomes eligible immediately."""
        self.controller.create_and_register_notice(
            emergency_id="UNANNOUNCED-02",
            approach=Approach.EAST,
            initial_eta=15.0,
            pre_informed=False
        )
        # Advance NORTH green to 10.0s (exact G_MIN)
        self.controller.state.time_in_phase = 9.0
        decision1 = self.controller.step(self.inputs, dt=1.0)  # Reaches 10.0s

        # At G_MIN satisfied, next step immediately triggers transition to YELLOW
        decision2 = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(decision2.active_phase, PhaseState.YELLOW)
        self.assertEqual(decision2.next_green_candidate, Approach.EAST)
        self.assertTrue(decision2.is_switch_in_progress)
        self.assertIn("Case C", decision2.reason)

    def test_c3_unannounced_emergency_detected_after_g_min(self):
        """TEST C3: Unannounced emergency detected after G_MIN triggers immediate transition toward YELLOW."""
        self.controller.state.time_in_phase = 14.0  # Already > G_MIN (10.0s)
        self.controller.create_and_register_notice(
            emergency_id="UNANNOUNCED-03",
            approach=Approach.SOUTH,
            initial_eta=20.0,
            pre_informed=False
        )

        decision = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(decision.active_phase, PhaseState.YELLOW)
        self.assertEqual(decision.next_green_candidate, Approach.SOUTH)
        self.assertTrue(decision.is_switch_in_progress)

    def test_c4_clearance_sequence_remains_green_yellow_allred_emergency_green(self):
        """TEST C4: Safe clearance sequence remains strictly GREEN -> YELLOW -> ALL_RED -> EMERGENCY GREEN."""
        self.controller.state.time_in_phase = 10.0
        self.controller.create_and_register_notice("UNANNOUNCED-04", Approach.EAST, 12.0, pre_informed=False)

        # Step 1: GREEN -> YELLOW
        d1 = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(d1.active_phase, PhaseState.YELLOW)

        # Step 2 & 3: YELLOW clearance (3.0s total)
        d2 = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(d2.active_phase, PhaseState.YELLOW)
        d3 = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(d3.active_phase, PhaseState.YELLOW)

        # Step 4: YELLOW completes -> ALL_RED
        d4 = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(d4.active_phase, PhaseState.ALL_RED)

        # Step 5: ALL_RED clearance (2.0s total)
        d5 = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(d5.active_phase, PhaseState.ALL_RED)

        # Step 6: ALL_RED completes -> EMERGENCY GREEN on EAST
        d6 = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(d6.active_phase, PhaseState.GREEN)
        self.assertEqual(d6.current_green, Approach.EAST)
        self.assertTrue(self.controller.is_emergency_active)

    def test_c5_emergency_cannot_directly_switch_green_to_emergency_green(self):
        """TEST C5: Emergency cannot directly switch GREEN -> EMERGENCY GREEN in a single step."""
        self.controller.state.time_in_phase = 15.0
        self.controller.create_and_register_notice("UNANNOUNCED-05", Approach.WEST, 10.0, pre_informed=False)

        decision = self.controller.step(self.inputs, dt=1.0)
        # Must not directly jump to GREEN on WEST
        self.assertNotEqual(decision.active_phase, PhaseState.GREEN)
        self.assertEqual(decision.active_phase, PhaseState.YELLOW)
        self.assertIsNone(decision.current_green)

    def test_c6_emergency_gets_emergency_green_at_earliest_safe_opportunity(self):
        """TEST C6: Emergency gets emergency green at earliest safe opportunity (exact yellow + all_red duration)."""
        self.controller.state.time_in_phase = 10.0
        self.controller.create_and_register_notice("UNANNOUNCED-06", Approach.EAST, 15.0, pre_informed=False)

        # Start transition
        self.controller.step(self.inputs, dt=1.0)  # switches to YELLOW, remaining = 3.0s
        # Advance through yellow (3s) and all-red (2s)
        for _ in range(3):
            self.controller.step(self.inputs, dt=1.0)  # yellow finishes, all_red entered (remaining = 2.0s)
        for _ in range(2):
            self.controller.step(self.inputs, dt=1.0)  # all-red finishes

        self.assertEqual(self.controller.state.phase_state, PhaseState.GREEN)
        self.assertEqual(self.controller.state.active_approach, Approach.EAST)

    def test_c7_single_green_invariant_remains_true(self):
        """TEST C7: Single-green invariant remains true across entire unannounced emergency sequence."""
        self.controller.state.time_in_phase = 5.0
        self.controller.create_and_register_notice("UNANNOUNCED-07", Approach.SOUTH, 18.0, pre_informed=False)

        for _ in range(25):
            dec = self.controller.step(self.inputs, dt=1.0)
            green_count = sum(1 for color in dec.signal_states.values() if color == SignalColor.GREEN)
            self.assertLessEqual(green_count, 1)

    def test_c8_existing_pre_informed_emergency_behavior_remains_unchanged(self):
        """TEST C8: Existing pre-informed emergency behavior remains unchanged (does not trigger before Case A/B)."""
        self.controller.state.time_in_phase = 12.0
        # Pre-informed emergency with ETA = 50.0s and 0 queue (T_clear = 0.0s, T_clear+3 = 3.0s << 50.0s)
        self.controller.create_and_register_notice("PRE-INFORMED-08", Approach.EAST, 50.0, pre_informed=True)

        dec = self.controller.step(self.inputs, dt=1.0)
        # Should NOT trigger Case C because pre_informed=True and ETA (49s) is not near trigger conditions
        self.assertFalse(self.controller.is_emergency_active)

    def test_c9_unannounced_emergency_does_not_modify_unrelated_junction_controllers(self):
        """TEST C9: Unannounced emergency detected at J-01 does not modify unrelated junction controllers (J-02)."""
        orchestrator = EmergencyOrchestrator()
        j1_adapter = orchestrator.get_or_create_junction_adapter("J-01")
        j2_adapter = orchestrator.get_or_create_junction_adapter("J-02")

        # Unannounced emergency arrives locally at J-01
        event = EmergencyDetectionEvent(
            emergency_id="UNANNOUNCED-J1",
            junction_id="J-01",
            approach=Approach.EAST,
            eta=10.0,
            pre_informed=False
        )
        j1_adapter.on_emergency_detected(event)

        j1_ctrl = orchestrator.get_or_create_junction_controller("J-01")
        j2_ctrl = orchestrator.get_or_create_junction_controller("J-02")

        self.assertIsNotNone(j1_ctrl.get_notice("UNANNOUNCED-J1"))
        self.assertFalse(j1_ctrl.get_notice("UNANNOUNCED-J1").pre_informed)
        self.assertIsNone(j2_ctrl.get_notice("UNANNOUNCED-J1"))

    def test_c10_multiple_unannounced_emergencies_compatible_with_phase3_conflict_resolution(self):
        """TEST C10: Multiple unannounced emergencies resolve conflicts using Phase 3 4-tier logic (lower ETA wins)."""
        self.controller.state.time_in_phase = 12.0
        self.controller.create_and_register_notice("UNANNOUNCED-E", Approach.EAST, 15.0, pre_informed=False)
        self.controller.create_and_register_notice("UNANNOUNCED-S", Approach.SOUTH, 8.0, pre_informed=False)

        # SOUTH has lower ETA (8s vs 15s) -> SOUTH should win conflict
        decision = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(decision.active_phase, PhaseState.YELLOW)
        self.assertEqual(decision.next_green_candidate, Approach.SOUTH)
        self.assertEqual(self.controller.active_emergency_id, "UNANNOUNCED-S")

    def test_c11_emergency_passage_still_correctly_completes_emergency_episode(self):
        """TEST C11: Emergency passage still correctly completes the emergency episode."""
        self.controller.state.time_in_phase = 10.0
        self.controller.create_and_register_notice("UNANNOUNCED-11", Approach.EAST, 5.0, pre_informed=False)

        # Transition to EAST green
        for _ in range(6):
            self.controller.step(self.inputs, dt=1.0)

        self.assertEqual(self.controller.state.active_approach, Approach.EAST)
        self.assertTrue(self.controller.is_emergency_active)

        # Confirm passage via camera event
        self.controller.ambulance_passed("UNANNOUNCED-11")
        self.inputs[Approach.EAST].queue_pcu = 0.0

        # Next step should conclude emergency
        dec = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(dec.active_phase, PhaseState.YELLOW)
        self.assertFalse(self.controller.is_emergency_active)
        self.assertIn("UNANNOUNCED-11", [n.emergency_id for n in self.controller.current_episode.passed_notices])

    def test_c12_emergency_dismissal_timeout_returns_controller_to_normal(self):
        """TEST C12: Emergency dismissal/timeout returns controller to normal operation."""
        notice = self.controller.create_and_register_notice("UNANNOUNCED-12", Approach.WEST, 0.0, pre_informed=False)
        notice.overdue_seconds = 16.0  # Exceeds 15.0s dismissal timeout

        self.controller.tick(1.0)
        self.assertNotIn("UNANNOUNCED-12", self.controller.current_episode.active_notices)
        self.assertEqual(notice.state, EmergencyState.DISMISSED)
        self.assertEqual(len(self.controller.current_episode.dismissed_notices), 1)

    def test_c13_normal_traffic_behavior_remains_unchanged_without_emergency(self):
        """TEST C13: Normal traffic behavior remains unchanged when there is no emergency."""
        self.controller.state.time_in_phase = 2.0
        dec = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(dec.active_phase, PhaseState.GREEN)
        self.assertEqual(dec.current_green, Approach.NORTH)
        self.assertFalse(self.controller.is_emergency_active)

    def test_c14_case_c_works_via_phase4a_adapter_and_bridge_interface(self):
        """TEST C14: Case C works correctly through Phase 4A adapter & Phase 4B bridge interface."""
        adapter = CameraIntegrationAdapter(controller=self.controller, junction_id="J-01")
        bridge = EmergencyVisionBridge(adapter=adapter, junction_id="J-01", approach=ApproachEnum.SOUTH)

        # External local vision detection event without prior registration
        det_event = EmergencyDetectionEvent(
            emergency_id="AMB-VISION-UNANNOUNCED",
            junction_id="J-01",
            approach=Approach.SOUTH,
            eta=12.0,
            pre_informed=False
        )
        success = adapter.on_emergency_detected(det_event)
        self.assertTrue(success)

        notice = self.controller.get_notice("AMB-VISION-UNANNOUNCED")
        self.assertIsNotNone(notice)
        self.assertFalse(notice.pre_informed)

        # Verify controller handles preemption
        self.controller.state.time_in_phase = 11.0
        dec = self.controller.step(self.inputs, dt=1.0)
        self.assertEqual(dec.active_phase, PhaseState.YELLOW)
        self.assertEqual(dec.next_green_candidate, Approach.SOUTH)

    def test_c15_phase5_multi_junction_orchestration_remains_intact(self):
        """TEST C15: Phase 5 multi-junction orchestration remains intact with persistent identity and corridor progression."""
        orchestrator = EmergencyOrchestrator()
        ctx = orchestrator.register_mission("MSN-501", "AMB-501", "J-01", "J-02")
        self.assertEqual(ctx.status, AmbulanceStatusEnum.TRANSIT_TO_HOSPITAL)

        j1_ctrl = orchestrator.get_or_create_junction_controller("J-01")
        j1_notice = j1_ctrl.get_notice("AMB-501")
        self.assertIsNotNone(j1_notice)
        self.assertTrue(j1_notice.pre_informed)  # Mission-registered emergency is pre-informed


if __name__ == "__main__":
    unittest.main()


