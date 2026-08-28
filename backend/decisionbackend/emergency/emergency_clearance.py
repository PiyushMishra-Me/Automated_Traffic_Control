"""
emergency_clearance.py
Clearance time (T_clear), Dynamic Emergency G_max, and Urgency Trigger calculations.

Formulas:
- T_clear = 6 + 2 * (Q_emergency_lane - 1)
  where Q_emergency_lane is the number of queued vehicles (or PCU) in the emergency lane.
  For Q = 1 -> 6s, Q = 2 -> 8s, Q = 3 -> 10s, Q = 4 -> 12s, Q = 5 -> 14s.
- Queue Cleared: Queue <= 0.5 PCU (consistent with normal signal controller threshold).
- Dynamic Emergency G_max:
  effective_emergency_g_max = max(normal_G_max, T_clear + 3)
  (+3s is the standard clearance safety margin).
- Emergency Trigger Rules:
  Case A: ETA <= T_clear + 3 (Immediate / G_min-bounded transition)
  Case B: ETA > T_clear + 3 and (T_clear + 3) - ETA < G_min + 3 (Urgency gap < G_min + 3)
"""

from typing import Tuple


# Default threshold from normal backend SignalTimingConfig
DEFAULT_EMPTY_QUEUE_THRESHOLD_PCU: float = 0.5
CLEARANCE_BASE_SECONDS: float = 6.0
CLEARANCE_PER_VEHICLE_SECONDS: float = 2.0
EMERGENCY_G_MAX_MARGIN_SECONDS: float = 3.0


def is_queue_cleared(queue_value: float, empty_threshold: float = DEFAULT_EMPTY_QUEUE_THRESHOLD_PCU) -> bool:
    """
    Checks if a queue is considered cleared based on normal backend threshold (<= 0.5 PCU).
    """
    return float(queue_value) <= float(empty_threshold)


def calculate_t_clear(queue_value: float, empty_threshold: float = DEFAULT_EMPTY_QUEUE_THRESHOLD_PCU) -> float:
    """
    Calculates dynamic clearance time T_clear required to flush the queue in the emergency lane.

    Formula:
      T_clear = 6 + 2 * (Q_emergency_lane - 1)

    If queue <= 0 or queue <= empty_threshold: returns 0.0 (queue already clear).
    """
    q = float(queue_value)
    if q <= 0.0 or is_queue_cleared(q, empty_threshold):
        return 0.0

    return CLEARANCE_BASE_SECONDS + CLEARANCE_PER_VEHICLE_SECONDS * (q - 1.0)


def calculate_effective_emergency_g_max(
    normal_g_max: float,
    t_clear: float,
    margin: float = EMERGENCY_G_MAX_MARGIN_SECONDS
) -> float:
    """
    Calculates dynamic effective G_max for the emergency approach.
    
    Formula:
      effective_emergency_g_max = max(normal_G_max, T_clear + margin)

    This is dynamic and non-destructive: it does not alter the baseline junction configuration.
    """
    return max(float(normal_g_max), float(t_clear) + float(margin))


def check_emergency_trigger_conditions(
    eta: float,
    t_clear: float,
    g_min: float,
    margin: float = EMERGENCY_G_MAX_MARGIN_SECONDS
) -> Tuple[bool, str]:
    """
    Evaluates Single Emergency Trigger Conditions:
    
    CASE A:
      IF ETA <= T_clear + 3:
        Trigger transition toward emergency lane (subject to G_min protection).
    
    CASE B:
      IF ETA > T_clear + 3:
        urgency_gap = (T_clear + 3) - ETA
        IF (T_clear + 3) - ETA < G_min + 3 (i.e. ETA < T_clear + G_min + 6):
          Trigger transition toward emergency lane.
        ELSE:
          Continue normal decision-making.
    """
    eta = float(eta)
    t_clear = float(t_clear)
    g_min = float(g_min)
    t_clear_with_margin = t_clear + margin

    if eta <= t_clear_with_margin:
        return True, f"Case A: ETA ({eta:.1f}s) <= T_clear+3 ({t_clear_with_margin:.1f}s)"

    urgency_gap = t_clear_with_margin - eta
    if urgency_gap < (g_min + margin):
        return True, f"Case B: Urgency gap ({urgency_gap:.1f}s) < G_min+3 ({g_min + margin:.1f}s) [ETA: {eta:.1f}s, T_clear: {t_clear:.1f}s]"

    return False, f"Normal: ETA ({eta:.1f}s) > T_clear+3+G_min+3 ({t_clear_with_margin + g_min + margin:.1f}s)"


class EmergencyClearanceCalculator:
    """
    Clearance calculator helper encapsulating queue and timing thresholds.
    """

    def __init__(
        self,
        empty_threshold: float = DEFAULT_EMPTY_QUEUE_THRESHOLD_PCU,
        g_max_margin: float = EMERGENCY_G_MAX_MARGIN_SECONDS
    ):
        self.empty_threshold = empty_threshold
        self.g_max_margin = g_max_margin

    def get_t_clear(self, queue_value: float) -> float:
        return calculate_t_clear(queue_value, self.empty_threshold)

    def get_effective_g_max(self, normal_g_max: float, queue_value: float) -> float:
        t_clear = self.get_t_clear(queue_value)
        return calculate_effective_emergency_g_max(normal_g_max, t_clear, self.g_max_margin)

    def check_trigger(self, eta: float, queue_value: float, g_min: float) -> Tuple[bool, str]:
        t_clear = self.get_t_clear(queue_value)
        return check_emergency_trigger_conditions(eta, t_clear, g_min, self.g_max_margin)
