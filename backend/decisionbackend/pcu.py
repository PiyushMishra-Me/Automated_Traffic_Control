"""
models.py
Data models and enumeration types for the 4-approach traffic signal decision engine.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List


class Approach(str, Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"


class SignalColor(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class PhaseState(str, Enum):
    GREEN = "GREEN"          # Approach has active green
    YELLOW = "YELLOW"        # Approach is in yellow change interval
    ALL_RED = "ALL_RED"      # All approaches are red for intersection clearance


@dataclass
class DirectionTraffic:
    """
    Traffic input metrics for a single approach at a decision tick.
    """
    direction: Approach
    vehicle_counts: Dict[str, int] = field(default_factory=dict)
    queue_pcu: float = 0.0
    wait_time: float = 0.0                      # Seconds since last green
    flow_rate: float = 0.0                      # PCU/min crossing counting line
    vehicles_waiting: int = 0                   # Raw count of queued vehicles
    vehicles_crossed_recently: int = 0          # Vehicles crossed in last window
    time_since_last_vehicle_passed: float = 0.0 # Seconds since last vehicle crossed counting line


@dataclass
class NormalizedMetrics:
    """
    Normalized metric values in range [0.0, 1.0].
    """
    queue_norm: float
    wait_norm: float
    flow_norm: float


@dataclass
class PriorityScore:
    """
    Computed priority score P(d) and normalized components.
    """
    direction: Approach
    score: float
    queue_norm: float
    wait_norm: float
    flow_norm: float


@dataclass
class SignalDecision:
    """
    Output produced by the decision backend at each tick.
    """
    current_green: Optional[Approach]
    active_phase: PhaseState
    signal_states: Dict[Approach, SignalColor]
    priority_scores: Dict[Approach, PriorityScore]
    reason: str
    phase_duration: float
    time_in_phase: float
    next_green_candidate: Optional[Approach] = None
    all_red_remaining: float = 0.0
    yellow_remaining: float = 0.0
    is_switch_in_progress: bool = False
