"""
emergency_controller.py
Emergency Decision Controller & State Machine (Phase 1, Phase 2 & Phase 3 Multi-Emergency Engine).

Features:
- Multi-Emergency Management & Dynamic Transitive ETA Clustering (<= 10s rule).
- Dynamic 2x Queue Priority Boost (0.45 -> 0.90) for Clustered Emergency Approaches.
- Independent Trigger Evaluation for all pending emergencies (Case A & Case B).
- 4-Tier Conflict Resolution: Lower ETA -> Shorter T_clear -> Higher WaitTime_norm -> Direction Order (N > E > S > W).
- G_MIN Protection & Strict Clearance Safety Sequencing (GREEN -> YELLOW -> ALL_RED -> GREEN).
- Dynamic T_clear & Dynamic Emergency G_max with Capped Extensions (G_max + 15s).
- Independent Passage Events, Dynamic Dismissal (ETA + 15s), and Wait-Time Recovery.
"""

from dataclasses import dataclass, field
import time
import uuid
from typing import Dict, List, Optional, Tuple, Union

from backend.decisionbackend.models import (
    Approach,
    SignalColor,
    PhaseState,
    DirectionTraffic,
    NormalizedMetrics,
    PriorityScore,
    SignalDecision,
)
from backend.decisionbackend.junction_config import JunctionConfig
from backend.decisionbackend.signal_controller import SignalController
from backend.decisionbackend.traffic_metrics import normalize_metrics

from backend.decisionbackend.emergency.emergency_models import (
    EmergencyNotice,
    EmergencyState,
    EmergencyPassageEvent,
    EmergencyVehicleType,
)
from backend.decisionbackend.emergency.emergency_eta import EmergencyETAManager
from backend.decisionbackend.emergency.emergency_clearance import (
    calculate_t_clear,
    calculate_effective_emergency_g_max,
    check_emergency_trigger_conditions,
    is_queue_cleared,
    DEFAULT_EMPTY_QUEUE_THRESHOLD_PCU,
    EMERGENCY_G_MAX_MARGIN_SECONDS,
)
from backend.decisionbackend.emergency.emergency_clustering import (
    form_eta_clusters,
    get_clustered_approaches,
    compute_emergency_priority_score,
    resolve_emergency_conflict,
    sort_emergencies_by_eta,
    CLUSTER_ETA_THRESHOLD_SECONDS,
)


@dataclass
class EmergencyEpisode:
    """
    State container for an emergency event sequence at an intersection.
    Holds active, passed, and dismissed notices along with dynamic metadata.
    """
    episode_id: str = field(default_factory=lambda: f"EP-{uuid.uuid4().hex[:8].upper()}")
    start_time: float = field(default_factory=time.time)
    active_notices: Dict[str, EmergencyNotice] = field(default_factory=dict)
    passed_notices: List[EmergencyNotice] = field(default_factory=list)
    dismissed_notices: List[EmergencyNotice] = field(default_factory=list)

    def get_active_approaches(self) -> List[Approach]:
        """
        Returns list of approaches that have active or pending emergency notices.
        """
        return [
            notice.approach
            for notice in self.active_notices.values()
            if notice.state in (EmergencyState.PENDING, EmergencyState.ACTIVE) and not notice.is_passed
        ]

    def get_current_etas(self) -> Dict[str, float]:
        """
        Returns mapping of emergency_id -> current_eta for active/pending notices.
        """
        return {
            e_id: notice.current_eta
            for e_id, notice in self.active_notices.items()
            if notice.state in (EmergencyState.PENDING, EmergencyState.ACTIVE) and not notice.is_passed
        }

    def get_notice(self, emergency_id: str) -> Optional[EmergencyNotice]:
        """
        Retrieves a notice by ID across active, passed, and dismissed sets.
        """
        if emergency_id in self.active_notices:
            return self.active_notices[emergency_id]
        for n in self.passed_notices:
            if n.emergency_id == emergency_id:
                return n
        for n in self.dismissed_notices:
            if n.emergency_id == emergency_id:
                return n
        return None


class EmergencyController:
    """
    Multi-Emergency Signal Controller (Phase 3).
    """

    def __init__(
        self,
        config: Optional[JunctionConfig] = None,
        normal_controller: Optional[SignalController] = None,
        initial_green: Optional[Approach] = Approach.NORTH
    ):
        self.config = config or JunctionConfig()
        self.normal_controller = normal_controller or SignalController(config=self.config, initial_green=initial_green)
        self.state = self.normal_controller.state
        self.current_episode: EmergencyEpisode = EmergencyEpisode()

        # Emergency Execution State
        self.is_emergency_active: bool = False
        self.active_emergency_id: Optional[str] = None
        self.emergency_green_elapsed: float = 0.0
        self.empty_threshold_pcu: float = self.config.timing.empty_queue_threshold_pcu
        self.emergency_extension_cap: float = 15.0
        self.dismissal_timeout: float = 15.0
        self.cluster_threshold: float = CLUSTER_ETA_THRESHOLD_SECONDS

    def reset_episode(self, episode_id: Optional[str] = None) -> EmergencyEpisode:
        """
        Initializes a fresh emergency episode container and resets emergency flags.
        """
        self.current_episode = EmergencyEpisode(
            episode_id=episode_id or f"EP-{uuid.uuid4().hex[:8].upper()}"
        )
        self.is_emergency_active = False
        self.active_emergency_id = None
        self.emergency_green_elapsed = 0.0
        return self.current_episode

    def register_notice(self, notice: EmergencyNotice) -> EmergencyNotice:
        """
        Registers an emergency notice into the active episode.
        """
        self.current_episode.active_notices[notice.emergency_id] = notice
        return notice

    def submit_emergency(self, notice: EmergencyNotice) -> EmergencyNotice:
        """Alias for register_notice."""
        return self.register_notice(notice)

    def create_and_register_notice(
        self,
        emergency_id: str,
        approach: Approach,
        initial_eta: float,
        vehicle_type: EmergencyVehicleType = EmergencyVehicleType.AMBULANCE,
        target_lane: Optional[str] = None
    ) -> EmergencyNotice:
        """
        Factory helper to create and register an EmergencyNotice.
        """
        notice = EmergencyNotice(
            emergency_id=emergency_id,
            approach=approach,
            current_eta=initial_eta,
            vehicle_type=vehicle_type,
            target_lane=target_lane
        )
        return self.register_notice(notice)

    def get_notice(self, emergency_id: str) -> Optional[EmergencyNotice]:
        """
        Finds a notice by its unique ID.
        """
        return self.current_episode.get_notice(emergency_id)

    def get_active_emergency(self) -> Optional[EmergencyNotice]:
        """
        Returns the emergency notice currently receiving green, or the lowest-ETA pending notice.
        """
        if self.active_emergency_id and self.active_emergency_id in self.current_episode.active_notices:
            return self.current_episode.active_notices[self.active_emergency_id]

        active_pending = [
            n for n in self.current_episode.active_notices.values()
            if n.state in (EmergencyState.PENDING, EmergencyState.ACTIVE) and not n.is_passed
        ]
        if active_pending:
            sorted_notices = sort_emergencies_by_eta(active_pending)
            return sorted_notices[0]
        return None

    def get_active_clusters(self) -> List[List[EmergencyNotice]]:
        """
        Returns the current transitive ETA clusters for all pending emergency notices.
        """
        return form_eta_clusters(
            list(self.current_episode.active_notices.values()),
            threshold=self.cluster_threshold
        )

    def update_emergency_eta(self, emergency_id: str, new_eta: float) -> bool:
        """
        Updates the ETA of an active emergency notice from tracking measurements.
        """
        notice = self.get_notice(emergency_id)
        if notice and notice.state in (EmergencyState.PENDING, EmergencyState.ACTIVE):
            EmergencyETAManager.update_notice_eta(notice, new_eta)
            return True
        return False

    def update_eta(self, emergency_id: str, new_eta: float) -> bool:
        """Alias for update_emergency_eta."""
        return self.update_emergency_eta(emergency_id, new_eta)

    def update_emergency_queue(self, emergency_id: str, queue_value: float) -> bool:
        """
        Updates the queue count/PCU for the emergency notice's lane.
        """
        notice = self.get_notice(emergency_id)
        if notice:
            notice.update_queue(queue_value)
            return True
        return False

    def tick(self, dt: float = 1.0):
        """
        Advances ETA countdown for all active/pending emergency notices and checks dismissal.
        """
        for e_id in list(self.current_episode.active_notices.keys()):
            notice = self.current_episode.active_notices.get(e_id)
            if not notice:
                continue

            notice.tick_eta(dt)

            # Check dismissal rule (pending ETA + 15s timeout without arrival/passage)
            if notice.is_dismissal_due(self.dismissal_timeout):
                self.dismiss_notice(e_id, reason="ETA + 15s timeout reached without passage confirmation")

    def get_t_clear(self, emergency_id: str, queue_value: float) -> Optional[float]:
        """
        Calculates dynamic T_clear for the lane corresponding to the given emergency.
        """
        notice = self.get_notice(emergency_id)
        if notice is None:
            return None
        return calculate_t_clear(queue_value, self.empty_threshold_pcu)

    def get_effective_g_max(
        self,
        emergency_id: str,
        normal_g_max: float,
        queue_value: float,
        margin: float = EMERGENCY_G_MAX_MARGIN_SECONDS
    ) -> Optional[float]:
        """
        Calculates dynamic effective emergency G_max based on the emergency lane's current queue.
        """
        t_clear = self.get_t_clear(emergency_id, queue_value)
        if t_clear is None:
            return None
        return calculate_effective_emergency_g_max(normal_g_max, t_clear, margin=margin)

    def ambulance_passed(
        self,
        event_or_id: Union[EmergencyPassageEvent, str],
        destination_approach: Optional[Approach] = None,
        camera_id: Optional[str] = None
    ) -> bool:
        """
        Consumes an ambulance passage confirmation event from the camera/tracking system.
        Marks ONLY the matching emergency notice as PASSED.
        """
        if hasattr(event_or_id, "emergency_id"):
            emergency_id = getattr(event_or_id, "emergency_id")
            destination_approach = destination_approach or getattr(event_or_id, "destination_approach", None)
        else:
            emergency_id = str(event_or_id)

        notice = self.current_episode.active_notices.get(emergency_id)
        if notice is not None and notice.state in (EmergencyState.PENDING, EmergencyState.ACTIVE, EmergencyState.PASSED):
            notice.mark_passed(destination_approach=destination_approach)
            return True

        for n in self.current_episode.passed_notices:
            if n.emergency_id == emergency_id:
                return True

        return False

    def dismiss_notice(self, emergency_id: str, reason: str = "") -> bool:
        """
        Dismisses an active emergency notice.
        """
        notice = self.current_episode.active_notices.pop(emergency_id, None)
        if notice is not None:
            notice.mark_dismissed(reason=reason)
            self.current_episode.dismissed_notices.append(notice)
            if self.active_emergency_id == emergency_id:
                self.is_emergency_active = False
                self.active_emergency_id = None
                self.emergency_green_elapsed = 0.0
            return True
        return False

    def step(self, traffic_inputs: Dict[Approach, DirectionTraffic], dt: Optional[float] = None) -> SignalDecision:
        """
        Processes a single decision tick incorporating Multi-Emergency Clustering & Decision Logic.
        """
        if dt is None:
            dt = self.config.timing.decision_interval

        # 1. Advance ETA ticks and check overdue dismissals
        self.tick(dt)

        # 2. Sync wait times into traffic inputs
        for app in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]:
            if app in traffic_inputs:
                traffic_inputs[app].wait_time = self.state.wait_times[app]

        # 3. Compute normalized metrics
        norm_metrics: Dict[Approach, NormalizedMetrics] = {}
        for app in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]:
            t = traffic_inputs.get(app, DirectionTraffic(direction=app))
            norm_metrics[app] = normalize_metrics(
                queue_pcu=t.queue_pcu,
                wait_time_sec=t.wait_time,
                flow_rate_pcu_min=t.flow_rate,
                config=self.config.normalization
            )

        # 4. Form dynamic clusters and compute priority scores (with 2x queue boost for clustered approaches)
        clustered_approaches = get_clustered_approaches(
            list(self.current_episode.active_notices.values()),
            threshold=self.cluster_threshold
        )
        priority_scores: Dict[Approach, PriorityScore] = {}
        for app in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]:
            is_clustered = app in clustered_approaches
            priority_scores[app] = compute_emergency_priority_score(
                app,
                norm_metrics[app],
                is_clustered=is_clustered,
                normal_weights=self.config.weights
            )

        # 5. Handle clearance transitions (YELLOW and ALL_RED)
        if self.state.phase_state == PhaseState.YELLOW:
            decision = self._handle_yellow(priority_scores, dt)
            self.state.update_wait_times(dt)
            return decision

        elif self.state.phase_state == PhaseState.ALL_RED:
            decision = self._handle_all_red(priority_scores, dt)
            self.state.update_wait_times(dt)
            return decision

        # 6. Handle GREEN phase
        decision = self._handle_green(traffic_inputs, norm_metrics, priority_scores, dt)
        self.state.update_wait_times(dt)
        return decision

    def _handle_yellow(self, priority_scores: Dict[Approach, PriorityScore], dt: float) -> SignalDecision:
        self.state.time_in_phase += dt
        self.state.transition_time_remaining -= dt

        if self.state.transition_time_remaining <= 0:
            # Yellow ended -> ALL_RED
            self.state.phase_state = PhaseState.ALL_RED
            self.state.time_in_phase = 0.0
            self.state.transition_time_remaining = self.config.timing.all_red_time
            active_name = self.state.active_approach.value if self.state.active_approach else "None"
            reason = f"Yellow clearance completed for {active_name}. Entering All-Red clearance."
        else:
            active_name = self.state.active_approach.value if self.state.active_approach else "None"
            reason = f"Yellow clearance in progress for {active_name} ({self.state.transition_time_remaining:.1f}s remaining)."

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
            # All-Red ended -> activate next approach
            new_green = self.state.next_approach or Approach.NORTH
            self.state.active_approach = new_green
            self.state.next_approach = None
            self.state.phase_state = PhaseState.GREEN
            self.state.time_in_phase = 0.0
            self.state.transition_time_remaining = 0.0

            # Check if this new green is an emergency approach
            active_notice = self.get_notice(self.active_emergency_id) if self.active_emergency_id else None
            if not active_notice or active_notice.approach != new_green:
                for n in self.current_episode.active_notices.values():
                    if n.approach == new_green and n.state in (EmergencyState.PENDING, EmergencyState.ACTIVE) and not n.is_passed:
                        active_notice = n
                        break

            if active_notice and active_notice.approach == new_green:
                self.is_emergency_active = True
                self.active_emergency_id = active_notice.emergency_id
                self.emergency_green_elapsed = 0.0
                active_notice.mark_active()
                reason = f"All-Red completed. Activated EMERGENCY GREEN for {new_green.value} ({active_notice.emergency_id})."
            else:
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
        norm_metrics: Dict[Approach, NormalizedMetrics],
        priority_scores: Dict[Approach, PriorityScore],
        dt: float
    ) -> SignalDecision:
        curr_green = self.state.active_approach
        if curr_green is None:
            curr_green = Approach.NORTH
            self.state.active_approach = curr_green
            self.state.phase_state = PhaseState.GREEN
            self.state.time_in_phase = 0.0

        timing = self.config.timing
        serving_notice = self.get_notice(self.active_emergency_id) if self.active_emergency_id else None

        # =========================================================================
        # CASE 1: Emergency Approach is Currently Receiving GREEN (Emergency Servicing)
        # =========================================================================
        if self.is_emergency_active and serving_notice and curr_green == serving_notice.approach:
            self.state.time_in_phase += dt
            self.emergency_green_elapsed += dt

            # Live queue & T_clear for active emergency
            emergency_traffic = traffic_inputs.get(curr_green, DirectionTraffic(direction=curr_green))
            queue_val = emergency_traffic.queue_pcu or emergency_traffic.vehicles_waiting
            serving_notice.update_queue(queue_val)
            t_clear = calculate_t_clear(queue_val, self.empty_threshold_pcu)
            effective_g_max = calculate_effective_emergency_g_max(timing.g_max, t_clear)
            max_capped_green = effective_g_max + self.emergency_extension_cap

            # Completion conditions
            is_cleared = is_queue_cleared(queue_val, self.empty_threshold_pcu)
            is_passed = serving_notice.is_passed or serving_notice.state == EmergencyState.PASSED
            is_hard_timeout = self.emergency_green_elapsed >= max_capped_green
            is_gmax_passed = (self.emergency_green_elapsed >= effective_g_max) and is_passed

            should_terminate_emergency = (is_cleared and is_passed) or is_hard_timeout or is_gmax_passed

            if should_terminate_emergency:
                # Conclude this specific emergency
                if is_passed:
                    self.current_episode.active_notices.pop(serving_notice.emergency_id, None)
                    self.current_episode.passed_notices.append(serving_notice)
                    finish_reason = f"Emergency completed: {serving_notice.emergency_id} passed and queue cleared ({queue_val:.1f} PCU)."
                else:
                    self.current_episode.active_notices.pop(serving_notice.emergency_id, None)
                    serving_notice.mark_dismissed("Emergency green extension limit (G_max+15s) reached without passage confirmation.")
                    self.current_episode.dismissed_notices.append(serving_notice)
                    finish_reason = f"Emergency timed out: {serving_notice.emergency_id} reached absolute green cap ({max_capped_green:.1f}s)."

                self.is_emergency_active = False
                self.active_emergency_id = None
                self.emergency_green_elapsed = 0.0

                # Re-evaluate remaining pending emergencies immediately
                remaining_pending = [
                    n for n in self.current_episode.active_notices.values()
                    if n.state in (EmergencyState.PENDING, EmergencyState.ACTIVE) and not n.is_passed
                ]
                eligible_candidates: List[Tuple[EmergencyNotice, float]] = []
                for n in remaining_pending:
                    n_traffic = traffic_inputs.get(n.approach, DirectionTraffic(direction=n.approach))
                    n_q = n_traffic.queue_pcu or n_traffic.vehicles_waiting
                    n_tclear = calculate_t_clear(n_q, self.empty_threshold_pcu)
                    is_trig, _ = check_emergency_trigger_conditions(n.current_eta, n_tclear, timing.g_min)
                    if is_trig:
                        eligible_candidates.append((n, n_tclear))

                target_next_approach = None
                if eligible_candidates:
                    winner_tuple = resolve_emergency_conflict(
                        eligible_candidates,
                        wait_norms={app: norm_metrics[app].wait_norm for app in Approach}
                    )
                    if winner_tuple:
                        winner_notice, _ = winner_tuple
                        target_next_approach = winner_notice.approach
                        self.is_emergency_active = True
                        self.active_emergency_id = winner_notice.emergency_id

                if target_next_approach is None:
                    other_approaches = [a for a in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST] if a != curr_green]
                    target_next_approach = max(other_approaches, key=lambda a: priority_scores[a].score)

                # Initiate transition: GREEN -> YELLOW
                self.state.phase_state = PhaseState.YELLOW
                self.state.next_approach = target_next_approach
                self.state.transition_time_remaining = timing.yellow_time
                self.state.time_in_phase = 0.0

                return SignalDecision(
                    active_phase=PhaseState.YELLOW,
                    current_green=None,
                    signal_states=self.state.get_signal_colors(),
                    phase_duration=0.0,
                    priority_scores=priority_scores,
                    reason=f"{finish_reason} Transitioning to {target_next_approach.value}.",
                    is_switch_in_progress=True,
                    next_green_candidate=target_next_approach
                )

            # Continue emergency green
            status_desc = f"Emergency GREEN: {curr_green.value} ({serving_notice.emergency_id}) | ETA: {serving_notice.current_eta:.1f}s | Queue: {queue_val:.1f} PCU | T_clear: {t_clear:.1f}s | Elapsed: {self.emergency_green_elapsed:.1f}s / Eff_G_max: {effective_g_max:.1f}s | Passed: {is_passed}"
            return SignalDecision(
                active_phase=PhaseState.GREEN,
                current_green=curr_green,
                signal_states=self.state.get_signal_colors(),
                phase_duration=self.state.time_in_phase,
                priority_scores=priority_scores,
                reason=status_desc,
                is_switch_in_progress=False,
                next_green_candidate=None
            )

        # =========================================================================
        # CASE 2: Multi-Emergency Trigger Evaluation while Current Approach has GREEN
        # =========================================================================
        pending_notices = [
            n for n in self.current_episode.active_notices.values()
            if n.state in (EmergencyState.PENDING, EmergencyState.ACTIVE) and not n.is_passed
        ]

        eligible_candidates: List[Tuple[EmergencyNotice, float]] = []
        for n in pending_notices:
            if n.approach == curr_green:
                continue
            n_traffic = traffic_inputs.get(n.approach, DirectionTraffic(direction=n.approach))
            n_q = n_traffic.queue_pcu or n_traffic.vehicles_waiting
            n.update_queue(n_q)
            n_tclear = calculate_t_clear(n_q, self.empty_threshold_pcu)
            is_trig, _ = check_emergency_trigger_conditions(n.current_eta, n_tclear, timing.g_min)
            if is_trig:
                eligible_candidates.append((n, n_tclear))

        if eligible_candidates:
            winner_tuple = resolve_emergency_conflict(
                eligible_candidates,
                wait_norms={app: norm_metrics[app].wait_norm for app in Approach}
            )
            if winner_tuple:
                winner_notice, winner_tclear = winner_tuple
                emp_app = winner_notice.approach
                curr_elapsed = self.state.time_in_phase

                if curr_elapsed < timing.g_min:
                    # G_MIN Protection: MUST hold current normal green until G_MIN
                    self.state.time_in_phase += dt
                    hold_reason = f"Emergency preemption pending for {emp_app.value} ({winner_notice.emergency_id}). Holding {curr_green.value} green for G_MIN ({self.state.time_in_phase:.1f}s <= {timing.g_min:.1f}s)."
                    return SignalDecision(
                        active_phase=PhaseState.GREEN,
                        current_green=curr_green,
                        signal_states=self.state.get_signal_colors(),
                        phase_duration=self.state.time_in_phase,
                        priority_scores=priority_scores,
                        reason=hold_reason,
                        is_switch_in_progress=False,
                        next_green_candidate=emp_app
                    )
                else:
                    # G_MIN fulfilled -> Immediately begin transition toward winner emergency approach
                    self.state.phase_state = PhaseState.YELLOW
                    self.state.next_approach = emp_app
                    self.state.transition_time_remaining = timing.yellow_time
                    self.state.time_in_phase = 0.0
                    self.is_emergency_active = True
                    self.active_emergency_id = winner_notice.emergency_id

                    switch_reason = f"Emergency override triggered for {emp_app.value} ({winner_notice.emergency_id}) [ETA: {winner_notice.current_eta:.1f}s, T_clear: {winner_tclear:.1f}s]. Terminating {curr_green.value} green."
                    return SignalDecision(
                        active_phase=PhaseState.YELLOW,
                        current_green=None,
                        signal_states=self.state.get_signal_colors(),
                        phase_duration=0.0,
                        priority_scores=priority_scores,
                        reason=switch_reason,
                        is_switch_in_progress=True,
                        next_green_candidate=emp_app
                    )

        # =========================================================================
        # CASE 3: Normal Traffic Signal Decision (With Emergency Intercept)
        # =========================================================================
        self.state.time_in_phase += dt
        curr_time = self.state.time_in_phase

        other_approaches = [a for a in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST] if a != curr_green]
        best_red_approach = max(other_approaches, key=lambda a: priority_scores[a].score)
        best_red_score = priority_scores[best_red_approach].score
        curr_green_score = priority_scores[curr_green].score
        active_traffic = traffic_inputs.get(curr_green, DirectionTraffic(direction=curr_green))

        should_switch = False
        switch_reason = ""

        # 1. G_MIN protection
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

        # 2. G_MAX enforcement
        if curr_time >= timing.g_max:
            should_switch = True
            switch_reason = f"G_MAX reached ({curr_time:.1f}s >= {timing.g_max}s). Forced transition from {curr_green.value}."

        # 3. Empty approach
        elif active_traffic.queue_pcu <= timing.empty_queue_threshold_pcu and active_traffic.flow_rate <= 0.5:
            if best_red_score > 0.05:
                should_switch = True
                switch_reason = f"Empty approach: {curr_green.value} queue depleted ({active_traffic.queue_pcu:.1f} PCU). Early termination."

        # 4. Gap-out
        elif active_traffic.time_since_last_vehicle_passed >= timing.gap_out_time and best_red_score > 0.10:
            should_switch = True
            switch_reason = f"Gap-out on {curr_green.value} ({active_traffic.time_since_last_vehicle_passed:.1f}s >= {timing.gap_out_time}s)."

        # 5. Priority switch
        elif (best_red_score - curr_green_score) >= timing.priority_switch_threshold:
            should_switch = True
            switch_reason = f"Priority switch: {best_red_approach.value} ({best_red_score:.2f}) exceeds {curr_green.value} ({curr_green_score:.2f})."

        if should_switch:
            target_next_approach = best_red_approach

            # Check if any emergency approach should intercept this phase change
            eligible_candidates_intercept: List[Tuple[EmergencyNotice, float]] = []
            for n in pending_notices:
                if n.approach != curr_green:
                    n_traffic = traffic_inputs.get(n.approach, DirectionTraffic(direction=n.approach))
                    n_q = n_traffic.queue_pcu or n_traffic.vehicles_waiting
                    n_tclear = calculate_t_clear(n_q, self.empty_threshold_pcu)
                    is_trig, _ = check_emergency_trigger_conditions(n.current_eta, n_tclear, timing.g_min)
                    if is_trig:
                        eligible_candidates_intercept.append((n, n_tclear))

            if eligible_candidates_intercept:
                winner_tuple = resolve_emergency_conflict(
                    eligible_candidates_intercept,
                    wait_norms={app: norm_metrics[app].wait_norm for app in Approach}
                )
                if winner_tuple:
                    winner_notice, _ = winner_tuple
                    target_next_approach = winner_notice.approach
                    self.is_emergency_active = True
                    self.active_emergency_id = winner_notice.emergency_id
                    switch_reason = f"Normal phase switch intercepted by emergency for {target_next_approach.value} ({winner_notice.emergency_id})"

            self.state.phase_state = PhaseState.YELLOW
            self.state.next_approach = target_next_approach
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
                next_green_candidate=target_next_approach
            )

        # Normal Green extension
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
