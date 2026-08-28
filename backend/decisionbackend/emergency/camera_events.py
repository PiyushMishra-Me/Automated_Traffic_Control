"""
camera_events.py
Strongly-typed camera/tracking event contracts for emergency vehicle integration (Phase 4A).

Defines the contract between external computer vision / tracking systems and the emergency decision backend.
Decoupled completely from OpenCV/YOLO/camera hardware details.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional, Tuple
from backend.decisionbackend.models import Approach
from backend.decisionbackend.emergency.emergency_models import (
    EmergencyVehicleType,
    EmergencyPassageEvent,
)


@dataclass(frozen=True)
class EmergencyDetectionEvent:
    """
    Event emitted when a camera/junction detects an approaching emergency vehicle.
    """
    emergency_id: str
    junction_id: str
    approach: Approach
    eta: float
    vehicle_type: EmergencyVehicleType = EmergencyVehicleType.AMBULANCE
    timestamp: float = field(default_factory=time.time)
    lane_id: Optional[str] = None
    confidence: float = 1.0
    tracking_metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, str]:
        """
        Validates required fields. Returns (is_valid, error_message).
        """
        if not self.emergency_id or not isinstance(self.emergency_id, str) or not self.emergency_id.strip():
            return False, "emergency_id must be a non-empty string"
        if not self.junction_id or not isinstance(self.junction_id, str) or not self.junction_id.strip():
            return False, "junction_id must be a non-empty string"
        if not isinstance(self.approach, Approach):
            return False, f"approach must be a valid Approach enum, got {self.approach}"
        if self.eta < 0.0 or not isinstance(self.eta, (int, float)):
            return False, f"eta must be a non-negative number, got {self.eta}"
        if not (0.0 <= self.confidence <= 1.0):
            return False, f"confidence must be between 0.0 and 1.0, got {self.confidence}"
        return True, ""


@dataclass(frozen=True)
class EmergencyEtaUpdateEvent:
    """
    Event emitted when the tracking system updates/corrects the live ETA of an active emergency vehicle.
    """
    emergency_id: str
    junction_id: str
    new_eta: float
    timestamp: float = field(default_factory=time.time)
    tracking_metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, str]:
        if not self.emergency_id or not isinstance(self.emergency_id, str) or not self.emergency_id.strip():
            return False, "emergency_id must be a non-empty string"
        if not self.junction_id or not isinstance(self.junction_id, str) or not self.junction_id.strip():
            return False, "junction_id must be a non-empty string"
        if self.new_eta < 0.0 or not isinstance(self.new_eta, (int, float)):
            return False, f"new_eta must be a non-negative number, got {self.new_eta}"
        return True, ""


@dataclass(frozen=True)
class DirectionalHandoffEvent:
    """
    Event emitted when an emergency vehicle leaves the current junction and heads toward a downstream junction.
    """
    emergency_id: str
    source_junction_id: str
    outgoing_approach: Approach
    destination_junction_id: Optional[str] = None
    next_approach: Optional[Approach] = None
    next_junction_eta: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    tracking_metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, str]:
        if not self.emergency_id or not isinstance(self.emergency_id, str) or not self.emergency_id.strip():
            return False, "emergency_id must be a non-empty string"
        if not self.source_junction_id or not isinstance(self.source_junction_id, str) or not self.source_junction_id.strip():
            return False, "source_junction_id must be a non-empty string"
        if not isinstance(self.outgoing_approach, Approach):
            return False, f"outgoing_approach must be a valid Approach enum, got {self.outgoing_approach}"
        if self.next_junction_eta is not None and (self.next_junction_eta < 0.0 or not isinstance(self.next_junction_eta, (int, float))):
            return False, f"next_junction_eta must be a non-negative number if provided, got {self.next_junction_eta}"
        return True, ""
