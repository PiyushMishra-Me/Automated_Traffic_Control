"""
emergency_orchestrator.py
End-to-End Multi-Junction Emergency Orchestration Layer (Phase 5).

Coordinates emergency missions across multiple junction controllers, camera bridges,
and directional corridor handoffs without modifying the underlying signal decision engine.
"""

from dataclasses import dataclass, field
import time
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from backend.models.traffic_schemas import ApproachEnum, MovementStateEnum
from backend.models.ambulance_schemas import (
    AmbulanceCriticalityEnum,
    AmbulanceStatusEnum,
    RouteJunctionNode,
)
from backend.core.control.ambulance_engine import AmbulanceEngine, CORRIDOR_MAP
from backend.core.vision.emergency_bridge import EmergencyVisionBridge, to_decision_approach
from backend.decisionbackend.models import Approach, PhaseState
from backend.decisionbackend.emergency.emergency_models import (
    EmergencyNotice,
    EmergencyState,
    EmergencyVehicleType,
)
from backend.decisionbackend.emergency.camera_events import (
    EmergencyDetectionEvent,
    EmergencyEtaUpdateEvent,
    EmergencyPassageEvent,
    DirectionalHandoffEvent,
)
from backend.decisionbackend.emergency.camera_interface import CameraIntegrationAdapter
from backend.decisionbackend.emergency.emergency_controller import EmergencyController

logger = logging.getLogger(__name__)


@dataclass
class EmergencyMissionContext:
    """
    Persistent state of an active multi-junction emergency vehicle mission.
    Maintains persistent emergency_id while allowing camera-local track_id to vary per junction.
    """
    mission_id: str
    emergency_id: str
    vehicle_id: str
    vehicle_type: EmergencyVehicleType = EmergencyVehicleType.AMBULANCE
    criticality: AmbulanceCriticalityEnum = AmbulanceCriticalityEnum.HIGH
    route_nodes: List[RouteJunctionNode] = field(default_factory=list)
    current_node_index: int = 0
    current_junction_id: Optional[str] = None
    current_approach: Optional[Approach] = None
    current_camera_id: Optional[str] = None
    current_track_id: Optional[int] = None  # Camera-local integer tracking ID
    status: AmbulanceStatusEnum = AmbulanceStatusEnum.TRANSIT_TO_HOSPITAL
    is_completed: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    handoff_history: List[Dict[str, Any]] = field(default_factory=list)

    def get_current_node(self) -> Optional[RouteJunctionNode]:
        if 0 <= self.current_node_index < len(self.route_nodes):
            return self.route_nodes[self.current_node_index]
        return None

    def get_next_node(self) -> Optional[RouteJunctionNode]:
        next_idx = self.current_node_index + 1
        if 0 <= next_idx < len(self.route_nodes):
            return self.route_nodes[next_idx]
        return None


class EmergencyOrchestrator:
    """
    Multi-Junction Emergency Orchestration Manager.
    Maintains per-junction EmergencyController instances and coordinates
    mission lifecycles, vision bridge associations, and inter-junction handoffs.
    """

    def __init__(self):
        # Per-junction decision controllers & camera adapters
        self.junction_controllers: Dict[str, EmergencyController] = {}
        self.junction_adapters: Dict[str, CameraIntegrationAdapter] = {}

        # Per-camera vision bridges: key = f"{junction_id}:{approach.value}"
        self.camera_bridges: Dict[str, EmergencyVisionBridge] = {}

        # Active mission contexts: key = emergency_id / mission_id
        self.active_missions: Dict[str, EmergencyMissionContext] = {}
        self.vehicle_to_emergency: Dict[str, str] = {}

    def get_or_create_junction_controller(
        self,
        junction_id: str,
        initial_green: Optional[Approach] = Approach.NORTH
    ) -> EmergencyController:
        """
        Retrieves or instantiates an independent EmergencyController for the given junction.
        """
        if junction_id not in self.junction_controllers:
            controller = EmergencyController(initial_green=initial_green)
            self.junction_controllers[junction_id] = controller
            # Wire corresponding adapter
            adapter = CameraIntegrationAdapter(controller=controller, junction_id=junction_id)
            self.junction_adapters[junction_id] = adapter
        return self.junction_controllers[junction_id]

    def get_or_create_junction_adapter(self, junction_id: str) -> CameraIntegrationAdapter:
        """
        Retrieves or creates the CameraIntegrationAdapter for the given junction.
        """
        if junction_id not in self.junction_adapters:
            self.get_or_create_junction_controller(junction_id)
        return self.junction_adapters[junction_id]

    def get_or_create_camera_bridge(
        self,
        junction_id: str,
        approach: Union[ApproachEnum, Approach, str]
    ) -> EmergencyVisionBridge:
        """
        Retrieves or creates the EmergencyVisionBridge for a specific camera approach at a junction.
        """
        app_enum = ApproachEnum(approach.value if hasattr(approach, "value") else str(approach))
        key = f"{junction_id}:{app_enum.value}"
        if key not in self.camera_bridges:
            adapter = self.get_or_create_junction_adapter(junction_id)
            bridge = EmergencyVisionBridge(adapter=adapter, junction_id=junction_id, approach=app_enum)
            self.camera_bridges[key] = bridge
        return self.camera_bridges[key]

    def register_mission(
        self,
        mission_id: str,
        vehicle_id: str,
        origin_junction_id: str,
        destination_junction_id: str,
        criticality: AmbulanceCriticalityEnum = AmbulanceCriticalityEnum.HIGH,
        vehicle_type: EmergencyVehicleType = EmergencyVehicleType.AMBULANCE,
        route_nodes: Optional[List[RouteJunctionNode]] = None
    ) -> EmergencyMissionContext:
        """
        Registers an emergency mission and pre-configures route nodes across junctions.
        """
        if route_nodes is None:
            route_nodes = AmbulanceEngine.plan_emergency_route(origin_junction_id, destination_junction_id)

        emergency_id = vehicle_id or mission_id

        first_jid = route_nodes[0].junction_id if route_nodes else origin_junction_id
        first_app = to_decision_approach(route_nodes[0].approach) if route_nodes else Approach.NORTH
        first_eta = float(route_nodes[0].eta_seconds) if route_nodes else 30.0

        context = EmergencyMissionContext(
            mission_id=mission_id,
            emergency_id=emergency_id,
            vehicle_id=vehicle_id,
            vehicle_type=vehicle_type,
            criticality=criticality,
            route_nodes=route_nodes,
            current_node_index=0,
            current_junction_id=first_jid,
            current_approach=first_app,
            status=AmbulanceStatusEnum.TRANSIT_TO_HOSPITAL,
            is_completed=False
        )

        self.active_missions[emergency_id] = context
        self.active_missions[mission_id] = context
        self.vehicle_to_emergency[vehicle_id] = emergency_id

        # Pre-register junction controllers along the planned corridor
        for node in route_nodes:
            self.get_or_create_junction_controller(node.junction_id)

        # Notify first junction controller of incoming emergency
        first_adapter = self.get_or_create_junction_adapter(first_jid)
        first_adapter.on_emergency_detected(
            EmergencyDetectionEvent(
                emergency_id=emergency_id,
                junction_id=first_jid,
                approach=first_app,
                eta=first_eta,
                vehicle_type=vehicle_type,
                pre_informed=True,
                timestamp=time.time(),
                tracking_metadata={"mission_id": mission_id, "origin": origin_junction_id, "pre_informed": True}
            )
        )

        logger.info(f"Registered emergency mission {mission_id} ({emergency_id}) at {first_jid} {first_app.value}")
        return context

    def associate_camera_track(
        self,
        junction_id: str,
        approach: Union[ApproachEnum, Approach, str],
        track_id: int,
        emergency_id_or_vehicle: str
    ) -> bool:
        """
        Associates a camera-local ByteTrack track_id with an active emergency mission.
        Enables the vision bridge to detect, track, and emit events for this vehicle.
        """
        emergency_id = self.vehicle_to_emergency.get(emergency_id_or_vehicle, emergency_id_or_vehicle)
        context = self.active_missions.get(emergency_id)
        if not context:
            logger.warning(f"Failed to associate track {track_id}: Emergency {emergency_id_or_vehicle} not found.")
            return False

        app_decision = to_decision_approach(approach)
        app_enum = ApproachEnum(app_decision.value)

        # Update mission context with local camera tracking metadata
        context.current_junction_id = junction_id
        context.current_approach = app_decision
        context.current_track_id = track_id
        context.updated_at = time.time()

        # Associate in camera vision bridge
        bridge = self.get_or_create_camera_bridge(junction_id, app_enum)
        bridge.associate_mission(track_id=track_id, emergency_id=emergency_id, vehicle_type=context.vehicle_type)

        logger.info(f"Associated track {track_id} with emergency {emergency_id} at {junction_id} ({app_decision.value})")
        return True

    def on_emergency_passed_junction(
        self,
        junction_id: str,
        emergency_id: str,
        destination_approach: Optional[Approach] = None
    ) -> bool:
        """
        Handles junction-level passage confirmation. Completes local junction emergency green
        and advances the mission context along its planned route corridor.
        """
        context = self.active_missions.get(emergency_id)
        if not context:
            return False

        # Mark passed on local junction controller via adapter
        adapter = self.get_or_create_junction_adapter(junction_id)
        ctrl = self.get_or_create_junction_controller(junction_id)
        if not ctrl.get_notice(emergency_id):
            adapter.on_emergency_detected(
                EmergencyDetectionEvent(
                    emergency_id=emergency_id,
                    junction_id=junction_id,
                    approach=context.current_approach or Approach.NORTH,
                    eta=0.0,
                    vehicle_type=context.vehicle_type,
                    timestamp=time.time()
                )
            )
        adapter.on_emergency_passed(
            EmergencyPassageEvent(
                emergency_id=emergency_id,
                junction_id=junction_id,
                approach=context.current_approach,
                destination_approach=destination_approach,
                timestamp=time.time()
            )
        )

        context.handoff_history.append({
            "event": "PASSAGE",
            "junction_id": junction_id,
            "timestamp": time.time(),
            "destination_approach": destination_approach.value if destination_approach else None
        })

        # Advance route corridor index
        curr_node = context.get_current_node()
        if curr_node and curr_node.junction_id == junction_id:
            context.current_node_index += 1

        next_node = context.get_current_node()
        if next_node is None:
            # Final destination reached
            context.is_completed = True
            context.status = AmbulanceStatusEnum.MISSION_ACCOMPLISHED
            context.current_track_id = None
            logger.info(f"Emergency mission {context.mission_id} ({emergency_id}) completed all corridor nodes.")
        else:
            context.status = AmbulanceStatusEnum.TRANSIT_TO_HOSPITAL
            context.current_junction_id = next_node.junction_id
            context.current_approach = to_decision_approach(next_node.approach)
            context.current_track_id = None  # Reset camera-local track ID for next junction camera
            logger.info(f"Emergency mission {context.mission_id} passed {junction_id}; en route to {next_node.junction_id}.")

        context.updated_at = time.time()
        return True

    def on_directional_handoff(self, event: DirectionalHandoffEvent) -> bool:
        """
        Processes directional handoff as an emergency vehicle leaves source_junction_id heading
        toward destination_junction_id. Transfers context and pre-notifies downstream junction.
        """
        is_valid, err = event.validate()
        if not is_valid:
            logger.warning(f"Invalid DirectionalHandoffEvent rejected: {err}")
            return False

        context = self.active_missions.get(event.emergency_id)
        if not context:
            logger.warning(f"Directional handoff for unknown emergency {event.emergency_id} ignored.")
            return False

        dest_jid = event.destination_junction_id
        next_app = event.next_approach
        next_eta = event.next_junction_eta or 45.0

        # If downstream destination is unresolved in event, check route corridor
        if not dest_jid:
            next_node = context.get_current_node()
            if next_node:
                dest_jid = next_node.junction_id
                next_app = to_decision_approach(next_node.approach)
                next_eta = float(next_node.eta_seconds)

        if not dest_jid:
            logger.warning(f"Unresolved corridor handoff for emergency {event.emergency_id} at boundary.")
            context.handoff_history.append({
                "event": "HANDOFF_UNRESOLVED",
                "source_junction": event.source_junction_id,
                "timestamp": event.timestamp
            })
            return False

        # Update context for downstream junction
        context.current_junction_id = dest_jid
        context.current_approach = next_app or Approach.NORTH
        context.current_track_id = None  # Clear camera-local track ID to prevent stale reuse
        context.updated_at = time.time()
        context.handoff_history.append({
            "event": "HANDOFF_SUCCESS",
            "source_junction": event.source_junction_id,
            "destination_junction": dest_jid,
            "next_approach": next_app.value if next_app else None,
            "timestamp": event.timestamp
        })

        # Pre-notify downstream junction controller
        downstream_adapter = self.get_or_create_junction_adapter(dest_jid)
        downstream_adapter.on_emergency_detected(
            EmergencyDetectionEvent(
                emergency_id=event.emergency_id,
                junction_id=dest_jid,
                approach=context.current_approach,
                eta=next_eta,
                vehicle_type=context.vehicle_type,
                pre_informed=True,
                timestamp=event.timestamp,
                tracking_metadata={"handoff_from": event.source_junction_id, "pre_informed": True}
            )
        )

        logger.info(f"Handoff completed for {event.emergency_id}: {event.source_junction_id} -> {dest_jid} ({context.current_approach.value})")
        return True

    def get_mission_context(self, emergency_id_or_mission: str) -> Optional[EmergencyMissionContext]:
        """
        Retrieves the mission context for a given emergency_id or mission_id.
        """
        return self.active_missions.get(emergency_id_or_mission)

    def get_junction_preemption_status(self, junction_id: str) -> Dict[str, Any]:
        """
        Returns real-time preemption status and active emergency telemetry for a specific junction.
        """
        controller = self.junction_controllers.get(junction_id)
        if not controller:
            return {"junction_id": junction_id, "is_preempted": False, "active_emergencies": []}

        active_notices = [
            {
                "emergency_id": n.emergency_id,
                "approach": n.approach.value,
                "current_eta": n.current_eta,
                "state": n.state.value,
                "is_passed": n.is_passed
            }
            for n in controller.current_episode.active_notices.values()
            if n.state in (EmergencyState.PENDING, EmergencyState.ACTIVE) and not n.is_passed
        ]

        return {
            "junction_id": junction_id,
            "is_preempted": controller.is_emergency_active,
            "active_emergency_id": controller.active_emergency_id,
            "current_phase": controller.state.phase_state.value,
            "active_approach": controller.state.active_approach.value if controller.state.active_approach else None,
            "active_emergencies": active_notices,
            "active_clusters_count": len(controller.get_active_clusters())
        }


# Singleton orchestrator instance for service registry
emergency_orchestrator = EmergencyOrchestrator()
