"""
junction_config.py
Configuration parameters for the 4-approach junction decision engine.
All PCU coefficients, normalization thresholds, priority weights,
and signal timing intervals are defined here and are fully configurable.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PCUConfig:
    """
    Field-measured Indian PCU coefficients (do NOT use textbook IRC values).
    - TWO_WHEELER: 0.13
    - CAR: 1.00
    - AUTO_RICKSHAW: 0.75
    - BUS: 5.40
    - TRUCK: 3.70
    """
    two_wheeler: float = 0.13
    car: float = 1.00
    auto_rickshaw: float = 0.75
    bus: float = 5.40
    truck: float = 3.70

    def to_dict(self) -> Dict[str, float]:
        return {
            "two_wheeler": self.two_wheeler,
            "motorcycle": self.two_wheeler,
            "bike": self.two_wheeler,
            "car": self.car,
            "auto_rickshaw": self.auto_rickshaw,
            "auto": self.auto_rickshaw,
            "bus": self.bus,
            "truck": self.truck,
        }


@dataclass
class NormalizationConfig:
    """
    Normalization reference ceilings for raw metrics:
    - Queue_norm = min(Queue_PCU / queue_pcu_max, 1.0)
    - WaitTime_norm = min(WaitTime / wait_time_ref, 1.0)
    - FlowRate_norm = min(FlowRate / flow_rate_max, 1.0)
    """
    queue_pcu_max: float = 40.0      # Max expected queued PCU
    wait_time_ref: float = 90.0      # Reference max wait time in seconds
    flow_rate_max: float = 12.0      # Reference max flow rate in PCU/min


@dataclass
class PriorityWeights:
    """
    Weights for calculating priority score P(d) = w_queue*Q_norm + w_wait*W_norm + w_flow*F_norm.
    Must sum to 1.0. Fixed weights: 0.45, 0.35, 0.20.
    """
    w_queue: float = 0.45
    w_wait: float = 0.35
    w_flow: float = 0.20


@dataclass
class SignalTimingConfig:
    """
    Configurable signal timing parameters:

    IMPORTANT NOTE:
    The timing parameters below (including g_min and g_max) are PROVISIONAL / TEMPORARY DEMO DEFAULTS.
    Final production timing values have NOT yet been decided and will be tuned separately.
    No internal decision logic hard-codes or assumes these values as fixed.

    - g_min: Minimum green phase duration in seconds before green can terminate.
             [TEMPORARY DEMO DEFAULT: 10.0s]
    - g_max: Maximum green phase duration in seconds before switch is forced.
             [TEMPORARY DEMO DEFAULT: 45.0s]
    - yellow_time: Yellow change interval duration in seconds.
             [TEMPORARY DEMO DEFAULT: 3.0s]
    - all_red_time: All-red clearance interval duration in seconds.
             [TEMPORARY DEMO DEFAULT: 2.0s]
    - gap_out_time: Inactivity duration threshold after G_MIN to trigger gap-out.
             [TEMPORARY DEMO DEFAULT: 3.5s]
    - decision_interval: Decision loop tick duration in seconds.
             [TEMPORARY DEMO DEFAULT: 1.0s]
    - flow_window_seconds: Rolling time window for flow rate estimation in seconds.
             [TEMPORARY DEMO DEFAULT: 30.0s]
    - priority_switch_margin: Minimum delta in P(d) required to switch green before G_MAX.
             [TEMPORARY DEMO DEFAULT: 0.15]
    - empty_queue_threshold_pcu: PCU below which an approach is considered effectively empty.
             [TEMPORARY DEMO DEFAULT: 0.5 PCU]
    """
    # PROVISIONAL / TEMPORARY DEMO VALUES:
    g_min: float = 10.0
    g_max: float = 45.0
    yellow_time: float = 3.0
    all_red_time: float = 2.0
    gap_out_time: float = 3.5
    decision_interval: float = 1.0
    flow_window_seconds: float = 30.0
    priority_switch_margin: float = 0.15
    empty_queue_threshold_pcu: float = 0.5


@dataclass
class JunctionConfig:
    """
    Complete configuration bundle for a 4-approach junction.
    """
    pcu: PCUConfig = field(default_factory=PCUConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    weights: PriorityWeights = field(default_factory=PriorityWeights)
    timing: SignalTimingConfig = field(default_factory=SignalTimingConfig)
