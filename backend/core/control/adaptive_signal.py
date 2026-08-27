from typing import List, Optional
from backend.models.traffic_schemas import (
    AlertSeverityEnum,
    ApproachEnum,
    ApproachTrafficState,
    JunctionTrafficState,
    SignalPhaseEnum,
    SignalRecommendation,
    TrafficAlert,
    TrafficLevelEnum,
)
from backend.core.weather.weather_service import weather_service
from backend.db.repositories.incident_repo import incident_repo

class AdaptiveSignalController:
    """Multi-criteria weather-aware & incident-aware signal recommendation engine."""

    @staticmethod
    def _score(state: Optional[ApproachTrafficState]) -> float:
        if not state:
            return 0.0
        level_bonus = {
            TrafficLevelEnum.LOW: 0,
            TrafficLevelEnum.MEDIUM: 3,
            TrafficLevelEnum.HIGH: 7,
            TrafficLevelEnum.VERY_HIGH: 12,
        }[state.traffic_level]
        return round(state.vehicle_count + (state.estimated_queue_length * 1.5) + (state.density * 10) + level_bonus, 2)

    @classmethod
    def _alerts(cls, junction_state: JunctionTrafficState, active_incidents: list, weather_telemetry) -> List[TrafficAlert]:
        alerts: List[TrafficAlert] = []

        # 1. Incident Alerts
        for inc in active_incidents:
            alerts.append(TrafficAlert(
                severity=AlertSeverityEnum.CRITICAL,
                approach=ApproachEnum(inc.approach),
                message=f"ACCIDENT / {inc.incident_type.value} on {inc.approach}: {inc.description}. Upstream diversion active via {inc.diversion_plan.recommended_reroute_corridor if inc.diversion_plan else 'alternate route'}."
            ))

        # 2. Weather Alerts
        if weather_telemetry.adjustments.extra_yellow_seconds > 0 or weather_telemetry.adjustments.extra_all_red_seconds > 0:
            alerts.append(TrafficAlert(
                severity=AlertSeverityEnum.WARNING,
                message=f"Weather Safety Advisory: {weather_telemetry.adjustments.safety_advisory} (Advisory speed: {weather_telemetry.adjustments.speed_advisory_kmh} km/h)"
            ))

        # 3. Approach Traffic Alerts
        for approach in ApproachEnum:
            state = getattr(junction_state, approach.value.lower())
            if not state:
                continue
            if state.traffic_level == TrafficLevelEnum.VERY_HIGH or state.estimated_queue_length >= 10:
                alerts.append(TrafficAlert(
                    severity=AlertSeverityEnum.CRITICAL,
                    approach=approach,
                    message=f"Severe congestion on {approach.value}: queue estimate {state.estimated_queue_length}."
                ))
            elif state.traffic_level == TrafficLevelEnum.HIGH or state.estimated_queue_length >= 5:
                alerts.append(TrafficAlert(
                    severity=AlertSeverityEnum.WARNING,
                    approach=approach,
                    message=f"High demand on {approach.value}: {state.vehicle_count} active vehicles."
                ))

        if not alerts and junction_state.total_active_vehicles == 0:
            alerts.append(TrafficAlert(
                severity=AlertSeverityEnum.INFO,
                message="No processed approach observations are currently available."
            ))
        return alerts

    @classmethod
    def recommend(cls, junction_state: JunctionTrafficState) -> SignalRecommendation:
        # Fetch active incidents
        active_incidents = incident_repo.get_active_for_junction(junction_state.junction_id)
        blocked_approaches = {inc.approach for inc in active_incidents}

        # Check for active emergency ambulance preemption
        from backend.db.repositories.ambulance_repo import ambulance_repo
        from backend.core.control.ambulance_engine import ambulance_engine
        active_missions = ambulance_repo.list_missions(status="DISPATCHED_TO_VICTIM") + ambulance_repo.list_missions(status="TRANSIT_TO_HOSPITAL")
        preemption = ambulance_engine.get_preemption_for_junction(junction_state.junction_id, active_missions)

        # Fetch weather adjustments
        weather = weather_service.get_weather_for_junction(junction_state.junction_id)
        yellow_duration = int(round(4 + weather.adjustments.extra_yellow_seconds))
        all_red_duration = int(round(2 + weather.adjustments.extra_all_red_seconds))

        if preemption.is_preempted and preemption.preempted_approach:
            # Emergency Ambulance Green Wave Override
            if preemption.preempted_approach in [ApproachEnum.NORTH, ApproachEnum.SOUTH]:
                phase = SignalPhaseEnum.NORTH_SOUTH_GREEN
            else:
                phase = SignalPhaseEnum.EAST_WEST_GREEN
            green_seconds = 45
            rationale = f"EMERGENCY OVERRIDE: {preemption.advisory} (Clearance window 45s active)."
            north_south = 100.0 if phase == SignalPhaseEnum.NORTH_SOUTH_GREEN else 0.0
            east_west = 100.0 if phase == SignalPhaseEnum.EAST_WEST_GREEN else 0.0
        else:
            north_score = cls._score(junction_state.north)
            south_score = cls._score(junction_state.south)
            east_score = cls._score(junction_state.east)
            west_score = cls._score(junction_state.west)

            # Apply incident penalty if approach is blocked
            if "NORTH" in blocked_approaches:
                north_score *= 0.2
            if "SOUTH" in blocked_approaches:
                south_score *= 0.2
            if "EAST" in blocked_approaches:
                east_score *= 0.2
            if "WEST" in blocked_approaches:
                west_score *= 0.2

            north_south = round(north_score + south_score, 2)
            east_west = round(east_score + west_score, 2)

            if north_south == 0 and east_west == 0:
                phase = SignalPhaseEnum.ALL_RED
                green_seconds = 0
                rationale = "No traffic observations available; maintaining all-red safe state."
            elif north_south >= east_west:
                phase = SignalPhaseEnum.NORTH_SOUTH_GREEN
                green_seconds = min(90, max(30, round(30 + north_south)))
                rationale = (
                    f"North/South selected with higher weighted demand score ({north_south:.1f} vs {east_west:.1f}). "
                    + (f"Adverse weather adjustment active: +{weather.adjustments.extra_yellow_seconds}s amber, +{weather.adjustments.extra_all_red_seconds}s all-red." if weather.adjustments.extra_yellow_seconds > 0 else "")
                )
            else:
                phase = SignalPhaseEnum.EAST_WEST_GREEN
                green_seconds = min(90, max(30, round(30 + east_west)))
                rationale = (
                    f"East/West selected with higher weighted demand score ({east_west:.1f} vs {north_south:.1f}). "
                    + (f"Adverse weather adjustment active: +{weather.adjustments.extra_yellow_seconds}s amber, +{weather.adjustments.extra_all_red_seconds}s all-red." if weather.adjustments.extra_yellow_seconds > 0 else "")
                )

        alerts = cls._alerts(junction_state, active_incidents, weather)
        if preemption.is_preempted:
            alerts.insert(0, TrafficAlert(
                severity=AlertSeverityEnum.CRITICAL,
                message=f"EMERGENCY AMBULANCE CORRIDOR: {preemption.advisory}"
            ))

        return SignalRecommendation(
            junction_id=junction_state.junction_id,
            recommended_phase=phase,
            green_duration_seconds=green_seconds,
            yellow_duration_seconds=yellow_duration,
            all_red_duration_seconds=all_red_duration,
            north_south_score=north_south,
            east_west_score=east_west,
            rationale=rationale,
            alerts=alerts
        )
