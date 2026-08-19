from datetime import datetime, timezone
from typing import Dict, Optional
from backend.models.traffic_schemas import ApproachEnum, ApproachTrafficState, JunctionTrafficState, TrafficLevelEnum

class JunctionAggregator:
    """
    Aggregates up to 4 approach states (North, South, East, West) into a single
    JunctionTrafficState structure.
    """
    @staticmethod
    def aggregate(
        junction_id: str,
        approach_states: Dict[str, ApproachTrafficState]
    ) -> JunctionTrafficState:
        north = approach_states.get(ApproachEnum.NORTH.value)
        south = approach_states.get(ApproachEnum.SOUTH.value)
        east = approach_states.get(ApproachEnum.EAST.value)
        west = approach_states.get(ApproachEnum.WEST.value)

        # Sum total active vehicles across all approaches
        total_active = sum(
            st.vehicle_count for st in [north, south, east, west] if st is not None
        )

        # Determine overall aggregate traffic level
        levels = [st.traffic_level for st in [north, south, east, west] if st is not None]
        if not levels:
            aggregate_level = TrafficLevelEnum.LOW
        elif any(lvl == TrafficLevelEnum.VERY_HIGH for lvl in levels):
            aggregate_level = TrafficLevelEnum.VERY_HIGH
        elif any(lvl == TrafficLevelEnum.HIGH for lvl in levels):
            aggregate_level = TrafficLevelEnum.HIGH
        elif any(lvl == TrafficLevelEnum.MEDIUM for lvl in levels):
            aggregate_level = TrafficLevelEnum.MEDIUM
        else:
            aggregate_level = TrafficLevelEnum.LOW

        return JunctionTrafficState(
            junction_id=junction_id,
            timestamp=datetime.now(timezone.utc),
            north=north,
            south=south,
            east=east,
            west=west,
            total_active_vehicles=total_active,
            aggregate_level=aggregate_level
        )
