from typing import List, Dict, Optional
from backend.models.traffic_schemas import ApproachEnum
from backend.models.incident_schemas import (
    IncidentSeverityEnum,
    DiversionPlan,
    DiversionStep,
)
from backend.db.repositories.junction_repo import junction_repo

# Known grid topology and bypass routes
INTERSECTION_NEIGHBORS: Dict[str, Dict[str, str]] = {
    "J-01": {"NORTH": "J-03", "SOUTH": "J-04", "EAST": "J-02", "WEST": "J-05"},
    "J-02": {"WEST": "J-01", "NORTH": "J-03", "SOUTH": "J-04"},
    "J-03": {"SOUTH": "J-01", "EAST": "J-02", "WEST": "J-05"},
    "J-04": {"NORTH": "J-01", "EAST": "J-02", "WEST": "J-05"},
    "J-05": {"EAST": "J-01", "NORTH": "J-03", "SOUTH": "J-04"},
}

CORRIDOR_NAMES = {
    "NORTH": "North Boulevard & Ring Road",
    "SOUTH": "South Radial Expressway",
    "EAST": "East Arterial Corridor B",
    "WEST": "West Commercial Linkway",
}

class DiversionEngine:
    @staticmethod
    def calculate_diversion(
        junction_id: str,
        approach: ApproachEnum,
        severity: IncidentSeverityEnum
    ) -> DiversionPlan:
        neighbors = INTERSECTION_NEIGHBORS.get(junction_id, {})
        
        # Determine upstream junction feeding this approach
        # (e.g., if NORTH approach is blocked, traffic came from the North or was heading North)
        opposite_approach_map = {
            ApproachEnum.NORTH: ApproachEnum.SOUTH,
            ApproachEnum.SOUTH: ApproachEnum.NORTH,
            ApproachEnum.EAST: ApproachEnum.WEST,
            ApproachEnum.WEST: ApproachEnum.EAST,
        }
        
        detour_direction_map = {
            ApproachEnum.NORTH: (ApproachEnum.EAST, "East Arterial Corridor B"),
            ApproachEnum.SOUTH: (ApproachEnum.WEST, "West Commercial Linkway"),
            ApproachEnum.EAST: (ApproachEnum.NORTH, "North Boulevard"),
            ApproachEnum.WEST: (ApproachEnum.SOUTH, "South Radial Expressway"),
        }

        detour_app, detour_name = detour_direction_map.get(approach, (ApproachEnum.EAST, "Bypass Arterial"))
        bypass_jid = neighbors.get(detour_app.value, "J-02")
        corridor = CORRIDOR_NAMES.get(approach.value, "Main Arterial")

        is_critical = (severity == IncidentSeverityEnum.CRITICAL_ROAD_BLOCKED)
        timing_strategy = (
            f"Throttle {approach.value} phase to G_MIN (10s) or hold RED; "
            f"Extend {detour_app.value} green by +20s at upstream {bypass_jid} to discharge detour volume."
        ) if is_critical else (
            f"Reduce {approach.value} green allocation by 40%; "
            f"Extend {detour_app.value} green by +10s on bypass route {detour_name}."
        )

        steps: List[DiversionStep] = [
            DiversionStep(
                step_number=1,
                instruction=f"Restrict incoming traffic on {corridor} before approaching junction {junction_id}.",
                corridor=corridor,
                upstream_junction_id=neighbors.get(approach.value),
                signal_action=f"Activate Variable Message Sign (VMS): 'ACCIDENT AHEAD ON {corridor} - USE {detour_name}'"
            ),
            DiversionStep(
                step_number=2,
                instruction=f"Divert vehicles heading toward {junction_id} via {detour_name} towards {bypass_jid}.",
                corridor=detour_name,
                upstream_junction_id=bypass_jid,
                signal_action=f"Extend green phase on {detour_app.value} approach by +18s to clear bottleneck."
            ),
            DiversionStep(
                step_number=3,
                instruction=f"Hold secondary queue progression at {junction_id} to ensure emergency vehicle corridor.",
                corridor=corridor,
                upstream_junction_id=junction_id,
                signal_action="Reserve dynamic clearance slot for first responder emergency response."
            )
        ]

        return DiversionPlan(
            affected_junction_id=junction_id,
            affected_approach=approach,
            severity=severity,
            bypass_junction_id=bypass_jid,
            recommended_reroute_corridor=detour_name,
            signal_timing_strategy=timing_strategy,
            steps=steps,
            active=True
        )

diversion_engine = DiversionEngine()
