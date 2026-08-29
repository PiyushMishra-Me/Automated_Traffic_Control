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
    Dynamic signal timing bounds and clearance intervals.
    """
    g_min: float = 10.0                      # Minimum green duration in seconds
    g_max: float = 40.0                      # Maximum green duration in seconds
    yellow_time: float = 3.0                 # Yellow clearance interval in seconds
    all_red_time: float = 2.0                # All-Red clearance interval in seconds
    gap_out_time: float = 3.0                # Max allowed gap headway before terminating green
    decision_interval: float = 1.0           # Seconds per decision tick
    empty_queue_threshold_pcu: float = 0.5   # PCU below which queue is considered cleared
    priority_switch_threshold: float = 0.25  # Priority delta needed to preempt green after G_MIN


@dataclass
class JunctionConfig:
    """
    Composite configuration for a single 4-approach intersection.
    """
    junction_id: str = "J-DEFAULT"
    pcu: PCUConfig = field(default_factory=PCUConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    weights: PriorityWeights = field(default_factory=PriorityWeights)
    timing: SignalTimingConfig = field(default_factory=SignalTimingConfig)
