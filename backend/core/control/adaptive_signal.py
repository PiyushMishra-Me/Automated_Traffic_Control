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


class AdaptiveSignalController:
    """Rule-based signal recommendation engine. It never controls physical hardware."""

    @staticmethod
    def _score(state: ApproachTrafficState | None) -> float:
        if not state:
            return 0.0
        level_bonus = {
            TrafficLevelEnum.LOW: 0,
            TrafficLevelEnum.MEDIUM: 3,
            TrafficLevelEnum.HIGH: 7,
            TrafficLevelEnum.VERY_HIGH: 12,
        }[state.traffic_level]
        return round(state.vehicle_count + (state.estimated_queue_length * 1.5) + (state.density * 10) + level_bonus, 2)

    @staticmethod
    def _alerts(junction_state: JunctionTrafficState) -> list[TrafficAlert]:
        alerts: list[TrafficAlert] = []
        for approach in ApproachEnum:
            state = getattr(junction_state, approach.value.lower())
            if not state:
                continue
            if state.traffic_level == TrafficLevelEnum.VERY_HIGH or state.estimated_queue_length >= 10:
                alerts.append(TrafficAlert(severity=AlertSeverityEnum.CRITICAL, approach=approach, message=f"Severe congestion on {approach.value}: queue estimate {state.estimated_queue_length}."))
            elif state.traffic_level == TrafficLevelEnum.HIGH or state.estimated_queue_length >= 5:
                alerts.append(TrafficAlert(severity=AlertSeverityEnum.WARNING, approach=approach, message=f"High demand on {approach.value}: {state.vehicle_count} active vehicles."))
        if not alerts and junction_state.total_active_vehicles == 0:
            alerts.append(TrafficAlert(severity=AlertSeverityEnum.INFO, message="No processed approach observations are currently available."))
        return alerts

    @classmethod
    def recommend(cls, junction_state: JunctionTrafficState) -> SignalRecommendation:
        north_south = cls._score(junction_state.north) + cls._score(junction_state.south)
        east_west = cls._score(junction_state.east) + cls._score(junction_state.west)
        if north_south == 0 and east_west == 0:
            phase = SignalPhaseEnum.ALL_RED
            green_seconds = 0
            rationale = "No traffic observations are available; keep the simulator in an all-red safe state."
        elif north_south >= east_west:
            phase = SignalPhaseEnum.NORTH_SOUTH_GREEN
            green_seconds = min(90, max(30, round(30 + north_south)))
            rationale = "North/South has the higher weighted demand score based on vehicles, queue, density, and congestion level."
        else:
            phase = SignalPhaseEnum.EAST_WEST_GREEN
            green_seconds = min(90, max(30, round(30 + east_west)))
            rationale = "East/West has the higher weighted demand score based on vehicles, queue, density, and congestion level."
        return SignalRecommendation(junction_id=junction_state.junction_id, recommended_phase=phase, green_duration_seconds=green_seconds, north_south_score=north_south, east_west_score=east_west, rationale=rationale, alerts=cls._alerts(junction_state))
