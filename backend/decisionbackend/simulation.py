"""
signal_state.py
State tracking and persistence for the 4-approach junction signals.
Manages:
- Active green approach (strictly 0 or 1 at any moment)
- Signal states for NORTH, SOUTH, EAST, WEST (RED, YELLOW, GREEN)
- Active phase state (GREEN, YELLOW, ALL_RED)
- Phase elapsed durations
- Approach wait-times (persisting across ticks, incrementing during RED, resetting on GREEN)
"""

from typing import Dict, Optional
from backend.decisionbackend.models import Approach, SignalColor, PhaseState


class JunctionSignalState:
    """
    Maintains persistent junction signal state across discrete decision ticks.
    """

    def __init__(self, initial_green: Optional[Approach] = Approach.NORTH):
        self.active_approach: Optional[Approach] = initial_green
        self.next_approach: Optional[Approach] = None
        self.phase_state: PhaseState = PhaseState.GREEN if initial_green else PhaseState.ALL_RED
        
        # Time counters
        self.time_in_phase: float = 0.0
        self.transition_time_remaining: float = 0.0

        # Wait-time tracking per approach (in seconds)
        # Initialized so active approach is 0, other approaches have 0 or initial wait
        self.wait_times: Dict[Approach, float] = {
            Approach.NORTH: 0.0,
            Approach.SOUTH: 0.0,
            Approach.EAST: 0.0,
            Approach.WEST: 0.0
        }

    def get_signal_colors(self) -> Dict[Approach, SignalColor]:
        """
        Derives discrete light colors for all 4 approaches based on current phase_state.
        Strict invariant: Never more than one GREEN simultaneously.
        """
        colors = {
            Approach.NORTH: SignalColor.RED,
            Approach.SOUTH: SignalColor.RED,
            Approach.EAST: SignalColor.RED,
            Approach.WEST: SignalColor.RED,
        }

        if self.phase_state == PhaseState.GREEN and self.active_approach:
            colors[self.active_approach] = SignalColor.GREEN
        elif self.phase_state == PhaseState.YELLOW and self.active_approach:
            colors[self.active_approach] = SignalColor.YELLOW
        # In ALL_RED, all four approaches remain RED.

        return colors

    def update_wait_times(self, dt: float):
        """
        Increases wait time continuously for RED/non-green approaches.
        Resets wait time to 0.0 for the active GREEN approach.
        """
        for app in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]:
            if app == self.active_approach and self.phase_state == PhaseState.GREEN:
                self.wait_times[app] = 0.0
            else:
                self.wait_times[app] = round(self.wait_times[app] + dt, 3)

    def start_yellow_transition(self, next_green: Approach, yellow_duration: float):
        """
        Initiates GREEN -> YELLOW transition for current approach.
        """
        self.phase_state = PhaseState.YELLOW
        self.next_approach = next_green
        self.time_in_phase = 0.0
        self.transition_time_remaining = yellow_duration

    def start_all_red_transition(self, all_red_duration: float):
        """
        Initiates YELLOW -> ALL_RED clearance interval.
        """
        self.phase_state = PhaseState.ALL_RED
        self.time_in_phase = 0.0
        self.transition_time_remaining = all_red_duration

    def finalize_green_switch(self):
        """
        Completes ALL_RED -> NEW GREEN phase switch.
        """
        self.active_approach = self.next_approach
        self.next_approach = None
        self.phase_state = PhaseState.GREEN
        self.time_in_phase = 0.0
        self.transition_time_remaining = 0.0
        if self.active_approach:
            self.wait_times[self.active_approach] = 0.0
