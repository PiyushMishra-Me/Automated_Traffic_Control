"""
signal_controller.py
Deterministic state-machine signal controller for 4-approach junction.
Implements:
- G_MIN protection (minimum green cannot be interrupted)
- G_MAX enforcement (forced switch upon reaching maximum green duration)
- Empty approach detection (early termination of green when queue/flow is depleted)
- Gap-out detection (early termination if headway/time-since-last-vehicle exceeds threshold)
- Competitive priority switching (switch if rival RED approach has significantly higher priority)
- Strict clearance sequencing: GREEN -> YELLOW -> ALL-RED -> NEW GREEN
- Continuous wait-time tracking
"""

from typing import Dict, Optional, Tuple
from backend.decisionbackend.junction_config import JunctionConfig
from backend.decisionbackend.models import (
    Approach,
    SignalColor,
    PhaseState,
    DirectionTraffic,
    PriorityScore,
    SignalDecision,
)
from backend.decisionbackend.signal_state import JunctionSignalState
from backend.decisionbackend.priority import compute_all_priorities


class SignalController:
    """
    Main signal controller operating on discrete decision ticks.
    """

    def __init__(self, config: Optional[JunctionConfig] = None, initial_green: Optional[Approach] = Approach.NORTH):
        self.config = config or JunctionConfig()
        self.state = JunctionSignalState(initial_green=initial_green)

    def step(self, traffic_inputs: Dict[Approach, DirectionTraffic], dt: Optional[float] = None) -> SignalDecision:
        """
        Processes a single decision tick.
        """
        if dt is None:
            dt = self.config.timing.decision_interval

        # 1. Sync wait times from internal state into traffic inputs if not supplied
        for app in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]:
            if app in traffic_inputs:
                traffic_inputs[app].wait_time = self.state.wait_times[app]

        # 2. Compute priority scores for all 4 approaches
        priority_scores = compute_all_priorities(
            traffic_inputs,
            weights=self.config.weights,
            norm_config=self.config.normalization
        )

        # 3. Process according to current phase state
        if self.state.phase_state == PhaseState.YELLOW:
            decision = self._handle_yellow(priority_scores, dt)
        elif self.state.phase_state == PhaseState.ALL_RED:
            decision = self._handle_all_red(priority_scores, dt)
        else:  # PhaseState.GREEN or None
            decision = self._handle_green(traffic_inputs, priority_scores, dt)

        # 4. Update wait times for all approaches
        self.state.update_wait_times(dt)

        return decision

    def _handle_yellow(self, priority_scores: Dict[Approach, PriorityScore], dt: float) -> SignalDecision:
        self.state.time_in_phase += dt
        self.state.transition_time_remaining -= dt

        if self.state.transition_time_remaining <= 0:
            # Yellow ended -> transition to ALL_RED
            self.state.phase_state = PhaseState.ALL_RED
            self.state.time_in_phase = 0.0
            self.state.transition_time_remaining = self.config.timing.all_red_time
            reason = f"Yellow clearance completed for {self.state.active_approach.value}. Entering All-Red clearance."
        else:
            reason = f"Yellow clearance in progress for {self.state.active_approach.value} ({self.state.transition_time_remaining:.1f}s remaining)."

        return SignalDecision(
            active_phase=self.state.phase_state,
            current_green=None,
            signal_states=self.state.get_signal_colors(),
            phase_duration=self.state.time_in_phase,
            priority_scores=priority_scores,
            reason=reason,
            is_switch_in_progress=True,
            next_green_candidate=self.state.next_approach
        )

    def _handle_all_red(self, priority_scores: Dict[Approach, PriorityScore], dt: float) -> SignalDecision:
        self.state.time_in_phase += dt
        self.state.transition_time_remaining -= dt

        if self.state.transition_time_remaining <= 0:
            # All-Red ended -> activate next green approach
            new_green = self.state.next_approach or Approach.NORTH
            self.state.active_approach = new_green
            self.state.next_approach = None
            self.state.phase_state = PhaseState.GREEN
            self.state.time_in_phase = 0.0
            self.state.transition_time_remaining = 0.0
            reason = f"All-Red clearance completed. Activated GREEN for {new_green.value}."
            is_switching = False
        else:
            reason = f"All-Red intersection clearance in progress ({self.state.transition_time_remaining:.1f}s remaining)."
            is_switching = True

        return SignalDecision(
            active_phase=self.state.phase_state,
            current_green=self.state.active_approach if self.state.phase_state == PhaseState.GREEN else None,
            signal_states=self.state.get_signal_colors(),
            phase_duration=self.state.time_in_phase,
            priority_scores=priority_scores,
            reason=reason,
            is_switch_in_progress=is_switching,
            next_green_candidate=self.state.next_approach
        )

    def _handle_green(
        self,
        traffic_inputs: Dict[Approach, DirectionTraffic],
        priority_scores: Dict[Approach, PriorityScore],
        dt: float
    ) -> SignalDecision:
        curr_green = self.state.active_approach
        if curr_green is None:
            # Pick highest priority approach if no green active
            best_app = max(priority_scores.keys(), key=lambda a: priority_scores[a].score)
            self.state.active_approach = best_app
            self.state.phase_state = PhaseState.GREEN
            self.state.time_in_phase = 0.0
            curr_green = best_app

        self.state.time_in_phase += dt
        curr_time = self.state.time_in_phase
        timing = self.config.timing

        # Find best candidate among RED approaches
        other_approaches = [a for a in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST] if a != curr_green]
        best_red_approach = max(other_approaches, key=lambda a: priority_scores[a].score)
        best_red_score = priority_scores[best_red_approach].score
        curr_green_score = priority_scores[curr_green].score

        active_traffic = traffic_inputs.get(curr_green, DirectionTraffic(direction=curr_green))

        # Check conditions in strict order:
        should_switch = False
        switch_reason = ""

        # 1. G_MIN protection: NEVER switch if time_in_phase < g_min
        if curr_time < timing.g_min:
            reason = f"G_MIN protected: {curr_green.value} green for {curr_time:.1f}s < G_MIN ({timing.g_min}s)."
            return SignalDecision(
                active_phase=PhaseState.GREEN,
                current_green=curr_green,
                signal_states=self.state.get_signal_colors(),
                phase_duration=curr_time,
                priority_scores=priority_scores,
                reason=reason,
                is_switch_in_progress=False,
                next_green_candidate=None
            )

        # 2. G_MAX enforcement: FORCED switch if time_in_phase >= g_max
        if curr_time >= timing.g_max:
            should_switch = True
            switch_reason = f"G_MAX reached ({curr_time:.1f}s >= {timing.g_max}s). Forced transition from {curr_green.value} to {best_red_approach.value}."

        # 3. Empty approach early termination
        elif active_traffic.queue_pcu <= timing.empty_queue_threshold_pcu and active_traffic.flow_rate <= 0.5:
            # If active queue is empty and red has demand
            if best_red_score > 0.05:
                should_switch = True
                switch_reason = f"Empty approach: {curr_green.value} queue depleted ({active_traffic.queue_pcu:.1f} PCU). Early termination for {best_red_approach.value}."

        # 4. Gap-out early termination
        elif active_traffic.time_since_last_vehicle_passed >= timing.gap_out_time and best_red_score > 0.10:
            should_switch = True
            switch_reason = f"Gap-out: No vehicle on {curr_green.value} for {active_traffic.time_since_last_vehicle_passed:.1f}s >= {timing.gap_out_time}s. Switching to {best_red_approach.value}."

        # 5. Competitive priority switching
        elif (best_red_score - curr_green_score) >= timing.priority_switch_threshold:
            should_switch = True
            switch_reason = f"Priority switch: {best_red_approach.value} priority ({best_red_score:.2f}) exceeds {curr_green.value} ({curr_green_score:.2f}) by >= {timing.priority_switch_threshold}."

        if should_switch and best_red_approach:
            # Initiate transition: GREEN -> YELLOW
            self.state.phase_state = PhaseState.YELLOW
            self.state.next_approach = best_red_approach
            self.state.transition_time_remaining = timing.yellow_time
            self.state.time_in_phase = 0.0

            return SignalDecision(
                active_phase=PhaseState.YELLOW,
                current_green=None,
                signal_states=self.state.get_signal_colors(),
                phase_duration=0.0,
                priority_scores=priority_scores,
                reason=switch_reason,
                is_switch_in_progress=True,
                next_green_candidate=best_red_approach
            )

        # Green extension
        reason = f"Extending GREEN for {curr_green.value} (Duration: {curr_time:.1f}s, Score: {curr_green_score:.2f} vs Best RED {best_red_approach.value}: {best_red_score:.2f})."
        return SignalDecision(
            active_phase=PhaseState.GREEN,
            current_green=curr_green,
            signal_states=self.state.get_signal_colors(),
            phase_duration=curr_time,
            priority_scores=priority_scores,
            reason=reason,
            is_switch_in_progress=False,
            next_green_candidate=None
        )
