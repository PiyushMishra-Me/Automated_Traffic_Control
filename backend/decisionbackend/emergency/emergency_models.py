"""
emergency_models.py
Data models and enumeration types for the Emergency Vehicle Decision Subsystem.
Reuses standard Approach enumeration from backend.decisionbackend.models.
"""

from enum import Enum
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional
from backend.decisionbackend.models import Approach


class EmergencyVehicleType(str, Enum):
    """
    Types of emergency vehicles. AMBULANCE is primary, extensible for other responders.
    """
    AMBULANCE = "AMBULANCE"
    FIRE_TRUCK = "FIRE_TRUCK"
    POLICE = "POLICE"
    OTHER = "OTHER"


class EmergencyState(str, Enum):
    """
    Lifecycle states for an emergency vehicle notice.
    """
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PASSED = "PASSED"
    DISMISSED = "DISMISSED"


@dataclass(frozen=True)
class EmergencyPassageEvent:
    """
    Event emitted when the camera/tracking system confirms the emergency vehicle has crossed the junction.
    """
    emergency_id: str
    junction_id: str = "J-DEFAULT"
    approach: Optional[Approach] = None
    destination_approach: Optional[Approach] = None
    camera_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    tracking_metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[bool, str]:
        if not self.emergency_id or not isinstance(self.emergency_id, str) or not self.emergency_id.strip():
            return False, "emergency_id must be a non-empty string"
        return True, ""


@dataclass
class EmergencyNotice:
    """
    Strongly-typed representation of an incoming emergency vehicle notice.
    Maintains live mutable ETA, lane queue, and passage state.
    """
    emergency_id: str
    approach: Approach
    current_eta: float
    vehicle_type: EmergencyVehicleType = EmergencyVehicleType.AMBULANCE
    original_eta: float = 0.0
    queue_pcu: float = 0.0
    state: EmergencyState = EmergencyState.PENDING
    is_passed: bool = False
    target_lane: Optional[str] = None
    destination_approach: Optional[Approach] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    overdue_seconds: float = 0.0

    def __post_init__(self):
        if self.original_eta == 0.0 and self.current_eta > 0.0:
            self.original_eta = float(self.current_eta)
        self.current_eta = float(self.current_eta)

    def tick_eta(self, dt: float = 1.0) -> float:
        """
        Decrements current_eta by dt seconds, clamping at 0.0.
        If at 0.0 and pending, increments overdue_seconds for dismissal evaluation.
        """
        if self.current_eta > 0.0:
            self.current_eta = max(0.0, self.current_eta - dt)
        elif self.state == EmergencyState.PENDING and not self.is_passed:
            self.overdue_seconds += dt
        self.updated_at = time.time()
        return self.current_eta

    def update_eta(self, new_eta: float):
        """
        Dynamically corrects live ETA from tracking input, keeping original_eta unchanged.
        """
        self.current_eta = max(0.0, float(new_eta))
        self.overdue_seconds = 0.0
        self.updated_at = time.time()

    def update_queue(self, queue_pcu: float):
        """
        Updates the current queued PCU in the emergency lane.
        """
        self.queue_pcu = max(0.0, float(queue_pcu))
        self.updated_at = time.time()

    def is_dismissal_due(self, timeout_seconds: float = 15.0) -> bool:
        """
        Checks if a pending notice has reached current_ETA == 0 and exceeded the overdue timeout.
        """
        return self.state == EmergencyState.PENDING and not self.is_passed and (self.overdue_seconds >= timeout_seconds)

    def mark_active(self):
        """
        Transitions notice to ACTIVE state when emergency green is initiated.
        """
        if self.state != EmergencyState.PASSED:
            self.state = EmergencyState.ACTIVE
            self.updated_at = time.time()

    def mark_passed(self, destination_approach: Optional[Approach] = None):
        """
        Transitions notice to PASSED state upon confirmation by passage event.
        """
        self.is_passed = True
        self.state = EmergencyState.PASSED
        if destination_approach is not None:
            self.destination_approach = destination_approach
        self.updated_at = time.time()

    def mark_dismissed(self, reason: str = ""):
        """
        Transitions notice to DISMISSED state if not confirmed / expired.
        """
        self.state = EmergencyState.DISMISSED
        self.updated_at = time.time()
