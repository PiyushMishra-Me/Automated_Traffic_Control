"""
emergency_eta.py
Deterministic Live ETA management for emergency vehicle notices.

Rules:
- ETA decreases by exactly 1 second for every 1-second decision tick.
- ETA must never become negative (clamped at 0.0).
- If ETA reaches 0, it stays at 0.
- Tracking system can correct/update ETA at any time with newly estimated values.
- Original ETA is preserved for logging, while current ETA is used for live operations.
"""

from typing import Dict, List, Union
from backend.decisionbackend.emergency.emergency_models import EmergencyNotice, EmergencyState


def tick_eta(current_eta: float, dt: float = 1.0) -> float:
    """
    Pure calculation: Decrement an ETA value by dt seconds, clamped at 0.0.
    """
    return max(0.0, float(current_eta) - float(dt))


def apply_eta_correction(current_eta: float, new_eta: float) -> float:
    """
    Pure calculation: Apply a new ETA correction from tracking, clamped at 0.0.
    """
    return max(0.0, float(new_eta))


class EmergencyETAManager:
    """
    Manages live ETA countdown and tracking updates for emergency notices.
    """

    @staticmethod
    def tick_notice(notice: EmergencyNotice, dt: float = 1.0) -> float:
        """
        Decrements ETA of a single notice by dt seconds if in PENDING or ACTIVE state.
        """
        if notice.state in (EmergencyState.PENDING, EmergencyState.ACTIVE):
            return notice.tick_eta(dt)
        return notice.current_eta

    @staticmethod
    def update_notice_eta(notice: EmergencyNotice, new_eta: float) -> float:
        """
        Updates the current ETA of a notice with a new tracking measurement.
        """
        return notice.update_eta(new_eta)

    @classmethod
    def tick_all(cls, notices: Union[Dict[str, EmergencyNotice], List[EmergencyNotice]], dt: float = 1.0):
        """
        Advances the countdown for all provided active/pending emergency notices.
        """
        iterable = notices.values() if isinstance(notices, dict) else notices
        for notice in iterable:
            cls.tick_notice(notice, dt)
