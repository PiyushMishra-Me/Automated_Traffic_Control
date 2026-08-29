"""
camera_interface.py
Camera & Tracking Integration Adapter (Phase 4A).

Acts as a clean, decoupled bridge between external computer vision / tracking pipelines
and the EmergencyController. Translates validated domain events into controller actions.
Contains ZERO OpenCV/YOLO/video-processing dependencies.
"""

from typing import Dict, List, Optional, Union
import logging

from backend.decisionbackend.models import Approach
from backend.decisionbackend.emergency.emergency_models import (
    EmergencyNotice,
    EmergencyState,
    EmergencyVehicleType,
)
from backend.decisionbackend.emergency.emergency_controller import EmergencyController
from backend.decisionbackend.emergency.camera_events import (
    EmergencyDetectionEvent,
    EmergencyEtaUpdateEvent,
    EmergencyPassageEvent,
    DirectionalHandoffEvent,
)

logger = logging.getLogger(__name__)


class CameraIntegrationAdapter:
    """
    Adapter translating external camera/tracking events into validated EmergencyController operations.
    Supports single junction controller or multi-junction coordination via junction_id registry.
    """

    def __init__(
        self,
        controller: Optional[EmergencyController] = None,
        junction_id: str = "J-DEFAULT"
    ):
        self.junction_id = junction_id
        self.controller = controller or EmergencyController()
        # Downstream junction adapter registry for multi-junction handoffs
        self._downstream_adapters: Dict[str, "CameraIntegrationAdapter"] = {}
        # Event history for auditing & replay
        self.event_log: List[Union[EmergencyDetectionEvent, EmergencyEtaUpdateEvent, EmergencyPassageEvent, DirectionalHandoffEvent]] = []
        self.handoff_log: List[DirectionalHandoffEvent] = []

    def register_downstream_adapter(self, junction_id: str, adapter: "CameraIntegrationAdapter"):
        """
        Registers a downstream junction adapter to receive automatic handoffs.
        """
        self._downstream_adapters[junction_id] = adapter

    def on_emergency_detected(self, event: EmergencyDetectionEvent) -> bool:
        """
        Handles incoming emergency detection event from camera/tracking system.
        Validates event, instantiates domain EmergencyNotice, and registers with controller.
        """
        is_valid, err = event.validate()
        if not is_valid:
            logger.warning(f"Rejected invalid EmergencyDetectionEvent: {err}")
            return False

        # Determine pre_informed status:
        existing_notice = self.controller.get_notice(event.emergency_id)
        if existing_notice is not None:
            pre_informed = existing_notice.pre_informed
        else:
            pre_informed = getattr(event, "pre_informed", False)
            if "pre_informed" in event.tracking_metadata:
                pre_informed = bool(event.tracking_metadata["pre_informed"])
            elif "mission_id" in event.tracking_metadata or "handoff_from" in event.tracking_metadata:
                pre_informed = True

        # Convert event to EmergencyNotice domain entity
        notice = EmergencyNotice(
            emergency_id=event.emergency_id,
            approach=event.approach,
            current_eta=float(event.eta),
            vehicle_type=event.vehicle_type,
            target_lane=event.lane_id,
            pre_informed=pre_informed,
            created_at=event.timestamp,
            updated_at=event.timestamp
        )

        self.controller.submit_emergency(notice)
        self.event_log.append(event)
        logger.info(f"Registered emergency notice {event.emergency_id} on {event.approach.value} at {event.junction_id} (ETA: {event.eta}s, Pre-informed: {pre_informed})")
        return True

    def on_emergency_eta_updated(self, event: EmergencyEtaUpdateEvent) -> bool:
        """
        Handles dynamic ETA correction from tracking system.
        Rejects invalid events or non-existent emergency notices safely.
        """
        is_valid, err = event.validate()
        if not is_valid:
            logger.warning(f"Rejected invalid EmergencyEtaUpdateEvent: {err}")
            return False

        updated = self.controller.update_emergency_eta(event.emergency_id, float(event.new_eta))
        if updated:
            self.event_log.append(event)
            logger.info(f"Updated ETA for {event.emergency_id} to {event.new_eta}s at {event.junction_id}")
            return True
        else:
            logger.warning(f"ETA update failed: Emergency {event.emergency_id} not found in active episode at {event.junction_id}")
            return False

    def on_emergency_passed(self, event: EmergencyPassageEvent) -> bool:
        """
        Handles camera passage confirmation event.
        Marks only the matching emergency notice as PASSED.
        """
        is_valid, err = event.validate()
        if not is_valid:
            logger.warning(f"Rejected invalid EmergencyPassageEvent: {err}")
            return False

        passed = self.controller.ambulance_passed(event)
        if passed:
            self.event_log.append(event)
            logger.info(f"Confirmed passage of emergency {event.emergency_id} at {event.junction_id}")
            return True
        else:
            logger.warning(f"Passage event rejected: Emergency {event.emergency_id} not found or already completed at {event.junction_id}")
            return False

    def on_direction_handoff(self, event: DirectionalHandoffEvent) -> bool:
        """
        Handles directional continuation event as emergency leaves junction toward another.
        """
        is_valid, err = event.validate()
        if not is_valid:
            logger.warning(f"Rejected invalid DirectionalHandoffEvent: {err}")
            return False

        self.event_log.append(event)
        self.handoff_log.append(event)
        logger.info(f"Processed handoff for {event.emergency_id} from {event.source_junction_id} heading {event.outgoing_approach.value}")

        # If destination junction adapter is registered and next approach / ETA provided, dispatch downstream detection
        if (
            event.destination_junction_id
            and event.destination_junction_id in self._downstream_adapters
            and event.next_approach is not None
            and event.next_junction_eta is not None
        ):
            downstream_adapter = self._downstream_adapters[event.destination_junction_id]
            downstream_event = EmergencyDetectionEvent(
                emergency_id=event.emergency_id,
                junction_id=event.destination_junction_id,
                approach=event.next_approach,
                eta=event.next_junction_eta,
                timestamp=event.timestamp,
                tracking_metadata={"handoff_from": event.source_junction_id}
            )
            downstream_adapter.on_emergency_detected(downstream_event)

        return True
