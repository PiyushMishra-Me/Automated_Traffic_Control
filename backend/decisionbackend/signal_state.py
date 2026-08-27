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

        Args:
            traffic_inputs: Dictionary mapping each Approach to its DirectionTraffic metrics.
            dt: Time elapsed since last decision tick in seconds (defaults to config.timing.decision_interval).

        Returns:
            SignalDecision describing current states, priority scores, and reason for action.
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

        # 3. Handle Active Phase State Machine
        reason = ""
        is_switch_in_progress = False

        if self.state.phase_state == PhaseState.YELLOW:
            is_switch_in_progress = True
            self.state.time_in_phase += dt
            self.state.transition_time_remaining -= dt
            
            if self.state.transition_time_remaining <= 0.0:
                # Yellow expired -> Enter ALL-RED clearance
                self.state.start_all_red_transition(self.config.timing.all_red_time)
                reason = f"Yellow clearance completed. Transitioning to ALL-RED clearance ({self.config.timing.all_red_time}s)."
            else:
                reason = f"Yellow change interval active for {self.state.active_approach.value} ({self.state.transition_time_remaining:.1f}s remaining)."

        elif self.state.phase_state == PhaseState.ALL_RED:
            is_switch_in_progress = True
            self.state.time_in_phase += dt
            self.state.transition_time_remaining -= dt
            
            if self.state.transition_time_remaining <= 0.0:
                # All-red expired -> Give GREEN to next approach
                prev_approach = self.state.active_approach
                self.state.finalize_green_switch()
                reason = f"All-red clearance completed. {self.state.active_approach.value} is now GREEN (switched from {prev_approach.value if prev_approach else 'None'})."
            else:
                reason = f"ALL-RED intersection clearance interval active ({self.state.transition_time_remaining:.1f}s remaining)."

        elif self.state.phase_state == PhaseState.GREEN:
            self.state.time_in_phase += dt
            current_app = self.state.active_approach
            time_green = self.state.time_in_phase

            current_traffic = traffic_inputs.get(current_app) if current_app else None
            current_queue = current_traffic.queue_pcu if current_traffic else 0.0
            current_p = priority_scores[current_app].score if current_app else 0.0

            # Identify highest priority candidate among RED approaches
            best_red_app, best_red_p = self._get_highest_priority_red_approach(priority_scores, current_app)

            # STEP 2: Minimum Green Protection
            if time_green < self.config.timing.g_min:
                reason = (
                    f"Holding {current_app.value} GREEN (elapsed {time_green:.1f}s < G_MIN {self.config.timing.g_min}s)."
                )

            # STEP 5: Maximum Green Enforcement
            elif time_green >= self.config.timing.g_max:
                next_target = best_red_app or self._get_next_cyclic_approach(current_app)
                self.state.start_yellow_transition(next_target, self.config.timing.yellow_time)
                is_switch_in_progress = True
                reason = (
                    f"FORCED SWITCH: {current_app.value} reached G_MAX ({time_green:.1f}s >= {self.config.timing.g_max}s). "
                    f"Switching to {next_target.value} (P={best_red_p:.2f})."
                )

            # STEP 3: Empty Approach Early Termination
            elif current_queue <= self.config.timing.empty_queue_threshold_pcu and (
                current_traffic is None or current_traffic.vehicles_crossed_recently == 0
            ):
                if best_red_app and best_red_p > 0.0:
                    self.state.start_yellow_transition(best_red_app, self.config.timing.yellow_time)
                    is_switch_in_progress = True
                    reason = (
                        f"EARLY TERMINATION: {current_app.value} is effectively empty (Queue={current_queue:.1f} PCU). "
                        f"Switching to {best_red_app.value} (P={best_red_p:.2f})."
                    )
                else:
                    reason = f"Keeping {current_app.value} GREEN (Approach is empty, but no rival red approach has demand)."

            # STEP 9: Gap-Out Detection
            elif (
                current_traffic is not None
                and current_traffic.time_since_last_vehicle_passed >= self.config.timing.gap_out_time
                and current_queue <= (self.config.timing.empty_queue_threshold_pcu * 2.0)
            ):
                if best_red_app and best_red_p > 0.0:
                    self.state.start_yellow_transition(best_red_app, self.config.timing.yellow_time)
                    is_switch_in_progress = True
                    reason = (
                        f"GAP-OUT: Inactivity threshold exceeded ({current_traffic.time_since_last_vehicle_passed:.1f}s >= {self.config.timing.gap_out_time}s). "
                        f"Switching to {best_red_app.value} (P={best_red_p:.2f})."
                    )
                else:
                    reason = f"Holding {current_app.value} GREEN (Gap-out condition met, but rival demand is zero)."

            # STEP 4: Priority-Driven Rival Switch
            elif best_red_app and (best_red_p > (current_p + self.config.timing.priority_switch_margin)):
                self.state.start_yellow_transition(best_red_app, self.config.timing.yellow_time)
                is_switch_in_progress = True
                reason = (
                    f"PRIORITY SWITCH: {best_red_app.value} priority (P={best_red_p:.2f}) exceeds "
                    f"{current_app.value} priority (P={current_p:.2f}) by margin > {self.config.timing.priority_switch_margin}."
                )

            else:
                reason = (
                    f"Continuing {current_app.value} GREEN (Elapsed: {time_green:.1f}s, P={current_p:.2f}). "
                    f"Rival highest: {best_red_app.value if best_red_app else 'None'} (P={best_red_p:.2f})."
                )

        # 4. Update Wait Times for all approaches
        self.state.update_wait_times(dt)

        # 5. Build and return immutable decision output
        return SignalDecision(
            current_green=self.state.active_approach if self.state.phase_state == PhaseState.GREEN else None,
            active_phase=self.state.phase_state,
            signal_states=self.state.get_signal_colors(),
            priority_scores=priority_scores,
            reason=reason,
            phase_duration=self.state.time_in_phase,
            time_in_phase=self.state.time_in_phase,
            next_green_candidate=self.state.next_approach,
            all_red_remaining=max(0.0, self.state.transition_time_remaining) if self.state.phase_state == PhaseState.ALL_RED else 0.0,
            yellow_remaining=max(0.0, self.state.transition_time_remaining) if self.state.phase_state == PhaseState.YELLOW else 0.0,
            is_switch_in_progress=is_switch_in_progress
        )

    def _get_highest_priority_red_approach(
        self,
        scores: Dict[Approach, PriorityScore],
        current_app: Optional[Approach]
    ) -> Tuple[Optional[Approach], float]:
        """
        Finds argmax P(d) among RED approaches (all approaches except current green).
        """
        best_app = None
        best_p = -1.0

        for app, score_obj in scores.items():
            if app == current_app:
                continue
            if score_obj.score > best_p:
                best_p = score_obj.score
                best_app = app

        return best_app, max(0.0, best_p)

    def _get_next_cyclic_approach(self, current_app: Optional[Approach]) -> Approach:
        """
        Cyclic fallback order: NORTH -> EAST -> SOUTH -> WEST -> NORTH
        """
        order = [Approach.NORTH, Approach.EAST, Approach.SOUTH, Approach.WEST]
        if current_app not in order:
            return Approach.NORTH
        idx = order.index(current_app)
        return order[(idx + 1) % len(order)]
