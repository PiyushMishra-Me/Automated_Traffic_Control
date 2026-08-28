"""
emergency_bridge.py
Phase 4B Vision-to-Emergency Bridge.

Bridges the existing camera / YOLO / ByteTrack / movement pipeline with
the Phase 4A CameraIntegrationAdapter and EmergencyController.

Responsibilities:
1. Emergency Detection Bridge: associates tracked vehicle track_id with emergency mission, emits EmergencyDetectionEvent.
2. Live ETA Update Bridge: calculates live ETA from distance-to-stopline / (speed_px * fps), emits EmergencyEtaUpdateEvent.
3. Passage + Directional Handoff Bridge: triggers EmergencyPassageEvent on line crossing and DirectionalHandoffEvent
   using CORRIDOR_MAP topology upon outgoing trajectory.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import time
import math
import numpy as np

from backend.models.traffic_schemas import ApproachEnum, MovementStateEnum
from backend.decisionbackend.models import Approach
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
from backend.decisionbackend.emergency.camera_interface import CameraIntegrationAdapter
from backend.core.control.ambulance_engine import CORRIDOR_MAP


Union_Approach = Union[ApproachEnum, Approach, str]


def to_decision_approach(app: Union_Approach) -> Approach:
    """Converts ApproachEnum or str to decisionbackend Approach enum."""
    val = app.value if hasattr(app, "value") else str(app)
    return Approach(val)


class EmergencyVisionBridge:
    """
    Integrates vision tracking telemetry with the Phase 4A CameraIntegrationAdapter.
    """

    def __init__(
        self,
        adapter: CameraIntegrationAdapter,
        junction_id: str = "J-01",
        approach: ApproachEnum = ApproachEnum.NORTH
    ):
        self.adapter = adapter
        self.junction_id = junction_id
        self.approach = approach

        # track_id -> emergency_id mapping
        self.track_to_emergency: Dict[int, str] = {}
        self.emergency_vehicle_types: Dict[str, EmergencyVehicleType] = {}

        # Deduplication state sets
        self.detected_emergencies: Set[str] = set()
        self.passed_emergencies: Set[str] = set()
        self.handed_off_emergencies: Set[str] = set()

        # Last emitted ETA to prevent fractional spamming
        self.last_emitted_eta: Dict[str, float] = {}

    def associate_mission(
        self,
        track_id: int,
        emergency_id: str,
        vehicle_type: EmergencyVehicleType = EmergencyVehicleType.AMBULANCE
    ):
        """
        Associates an active tracked vehicle ID with an emergency mission / vehicle identity.
        """
        self.track_to_emergency[track_id] = emergency_id
        self.emergency_vehicle_types[emergency_id] = vehicle_type

    def get_associated_emergency(self, track_id: int) -> Optional[str]:
        """
        Retrieves the emergency_id associated with a track_id if any.
        """
        return self.track_to_emergency.get(track_id)

    @staticmethod
    def calculate_pixel_distance_to_stopline(
        center: Tuple[float, float],
        line_config: Optional[dict],
        frame_width: int,
        frame_height: int
    ) -> float:
        """
        Calculates the perpendicular/Euclidean pixel distance from the vehicle center to the counting line.
        """
        if not line_config or "p1" not in line_config or "p2" not in line_config:
            line_y = frame_height * 0.5
            return abs(float(center[1]) - float(line_y))

        p1_x = float(line_config["p1"][0] * frame_width)
        p1_y = float(line_config["p1"][1] * frame_height)
        p2_x = float(line_config["p2"][0] * frame_width)
        p2_y = float(line_config["p2"][1] * frame_height)

        dx = p2_x - p1_x
        dy = p2_y - p1_y
        l_len = math.hypot(dx, dy)
        if l_len == 0.0:
            return float(math.hypot(center[0] - p1_x, center[1] - p1_y))

        # 2D cross product: |(p2_x - p1_x)*(center_y - p1_y) - (p2_y - p1_y)*(center_x - p1_x)|
        cross_prod = abs(dx * (float(center[1]) - p1_y) - dy * (float(center[0]) - p1_x))
        dist = cross_prod / l_len
        return float(dist)

    @staticmethod
    def calculate_live_eta(
        distance_px: float,
        speed_px: float,
        fps: float = 25.0,
        default_eta: float = 15.0
    ) -> float:
        """
        Calculates live ETA in seconds:
            ETA_live = pixel_distance_to_stopline / (speed_px * fps)
        Handles zero or near-zero speed safely without division-by-zero.
        """
        current_fps = max(1.0, float(fps))
        if speed_px <= 0.5:
            fallback_speed = 5.0
            return max(1.0, round(distance_px / (fallback_speed * current_fps), 1))

        speed_px_sec = float(speed_px) * current_fps
        if speed_px_sec <= 0.0:
            return float(default_eta)

        eta = float(distance_px) / speed_px_sec
        return max(0.0, round(eta, 1))

    def process_frame(
        self,
        vehicles: List[Any],
        counting_line_config: Optional[dict],
        frame_width: int,
        frame_height: int,
        fps: float = 25.0,
        current_timestamp: Optional[float] = None
    ):
        """
        Inspects all tracked vehicles in the current frame, evaluates emergency associations,
        and dispatches detection, ETA update, passage, and handoff events.
        """
        now = current_timestamp if current_timestamp is not None else time.time()
        decision_approach = to_decision_approach(self.approach)

        for v in vehicles:
            emergency_id = self.track_to_emergency.get(v.track_id)
            if not emergency_id:
                continue

            v_type = self.emergency_vehicle_types.get(emergency_id, EmergencyVehicleType.AMBULANCE)

            # Distance & Live ETA calculation
            dist_px = self.calculate_pixel_distance_to_stopline(
                v.center,
                counting_line_config,
                frame_width,
                frame_height
            )
            live_eta = self.calculate_live_eta(dist_px, v.speed_px, fps=fps)

            # -------------------------------------------------------------
            # 1. EMERGENCY DETECTION BRIDGE (Emit only once on first sight)
            # -------------------------------------------------------------
            if emergency_id not in self.detected_emergencies:
                det_event = EmergencyDetectionEvent(
                    emergency_id=emergency_id,
                    junction_id=self.junction_id,
                    approach=decision_approach,
                    eta=live_eta,
                    vehicle_type=v_type,
                    timestamp=now,
                    confidence=float(v.confidence),
                    tracking_metadata={"track_id": v.track_id, "speed_px": v.speed_px}
                )
                self.adapter.on_emergency_detected(det_event)
                self.detected_emergencies.add(emergency_id)
                self.last_emitted_eta[emergency_id] = live_eta

            # -------------------------------------------------------------
            # 2. LIVE ETA UPDATE BRIDGE (Emit when ETA meaningfully changes)
            # -------------------------------------------------------------
            elif emergency_id not in self.passed_emergencies:
                last_eta = self.last_emitted_eta.get(emergency_id, -999.0)
                if abs(live_eta - last_eta) >= 1.0 or live_eta == 0.0:
                    update_event = EmergencyEtaUpdateEvent(
                        emergency_id=emergency_id,
                        junction_id=self.junction_id,
                        new_eta=live_eta,
                        timestamp=now,
                        tracking_metadata={"track_id": v.track_id, "speed_px": v.speed_px}
                    )
                    self.adapter.on_emergency_eta_updated(update_event)
                    self.last_emitted_eta[emergency_id] = live_eta

            # -------------------------------------------------------------
            # 3. PASSAGE DETECTION BRIDGE (Emit on counting line crossing)
            # -------------------------------------------------------------
            is_crossed = getattr(v, "crossed_counting_line", False)
            if is_crossed and emergency_id not in self.passed_emergencies:
                passage_event = EmergencyPassageEvent(
                    emergency_id=emergency_id,
                    junction_id=self.junction_id,
                    approach=decision_approach,
                    timestamp=now,
                    tracking_metadata={"track_id": v.track_id}
                )
                self.adapter.on_emergency_passed(passage_event)
                self.passed_emergencies.add(emergency_id)

            # -------------------------------------------------------------
            # 4. DIRECTIONAL HANDOFF BRIDGE (Emit on outgoing movement)
            # -------------------------------------------------------------
            v_dir = getattr(v, "direction", None)
            is_outgoing = (v_dir in (MovementStateEnum.OUTGOING, MovementStateEnum.STOPPED_OUTGOING))

            if emergency_id in self.passed_emergencies and emergency_id not in self.handed_off_emergencies and is_outgoing:
                dest_junction, next_approach = self._resolve_downstream_corridor(self.junction_id, self.approach)

                handoff_event = DirectionalHandoffEvent(
                    emergency_id=emergency_id,
                    source_junction_id=self.junction_id,
                    outgoing_approach=decision_approach,
                    destination_junction_id=dest_junction,
                    next_approach=to_decision_approach(next_approach) if next_approach else None,
                    next_junction_eta=45.0 if dest_junction else None,
                    timestamp=now,
                    tracking_metadata={"track_id": v.track_id, "source_approach": self.approach.value}
                )
                self.adapter.on_direction_handoff(handoff_event)
                self.handed_off_emergencies.add(emergency_id)

    @staticmethod
    def _resolve_downstream_corridor(
        current_junction_id: str,
        current_approach: ApproachEnum
    ) -> Tuple[Optional[str], Optional[ApproachEnum]]:
        """
        Looks up downstream junction and target approach from CORRIDOR_MAP.
        """
        for (src, dst), (next_app, corridor_name) in CORRIDOR_MAP.items():
            if src == current_junction_id:
                return dst, next_app
        return None, None
