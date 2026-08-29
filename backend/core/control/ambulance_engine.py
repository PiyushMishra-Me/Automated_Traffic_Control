from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from backend.models.traffic_schemas import ApproachEnum
from backend.models.ambulance_schemas import (
    AmbulanceCriticalityEnum,
    AmbulanceStatusEnum,
    AmbulanceMissionCreate,
    AmbulanceMissionResponse,
    RouteJunctionNode,
    ConflictResolutionResult,
    AmbulancePreemptionStatus
)
from backend.db.repositories.junction_repo import junction_repo

CRITICALITY_PRIORITY_MAP: Dict[AmbulanceCriticalityEnum, int] = {
    AmbulanceCriticalityEnum.CRITICAL_LIFE_THREATENING: 4,
    AmbulanceCriticalityEnum.HIGH: 3,
    AmbulanceCriticalityEnum.MEDIUM: 2,
    AmbulanceCriticalityEnum.LOW: 1,
}

# Grid connections and approaches
CORRIDOR_MAP: Dict[Tuple[str, str], Tuple[ApproachEnum, str]] = {
    ("J-04", "J-01"): (ApproachEnum.SOUTH, "South Radial Expressway"),
    ("J-01", "J-02"): (ApproachEnum.WEST, "East Arterial Corridor B"),
    ("J-02", "J-01"): (ApproachEnum.EAST, "East Arterial Corridor B"),
    ("J-01", "J-03"): (ApproachEnum.SOUTH, "North Boulevard"),
    ("J-03", "J-01"): (ApproachEnum.NORTH, "North Boulevard"),
    ("J-01", "J-05"): (ApproachEnum.EAST, "West Commercial Linkway"),
    ("J-05", "J-01"): (ApproachEnum.WEST, "West Commercial Linkway"),
    ("J-03", "J-02"): (ApproachEnum.NORTH, "Campus Flyover"),
    ("J-02", "J-03"): (ApproachEnum.EAST, "Campus Flyover"),
    ("J-04", "J-02"): (ApproachEnum.SOUTH, "Station Terminal Road"),
    ("J-05", "J-04"): (ApproachEnum.WEST, "Hospital Access Way"),
    ("J-04", "J-05"): (ApproachEnum.SOUTH, "Hospital Access Way"),
}

class AmbulanceEngine:
    @staticmethod
    def plan_emergency_route(origin_id: str, destination_id: str) -> List[RouteJunctionNode]:
        """
        Generates structured route nodes with approaches and predicted ETAs.
        """
        # Graph BFS for shortest junction path
        graph = {
            "J-01": ["J-02", "J-03", "J-04", "J-05"],
            "J-02": ["J-01", "J-03", "J-04"],
            "J-03": ["J-01", "J-02", "J-05"],
            "J-04": ["J-01", "J-02", "J-05"],
            "J-05": ["J-01", "J-03", "J-04"],
        }

        queue = [[origin_id]]
        visited = {origin_id}
        shortest_path = [origin_id, "J-01", destination_id] if origin_id != destination_id else [origin_id]

        while queue:
            path = queue.pop(0)
            node = path[-1]
            if node == destination_id:
                shortest_path = path
                break
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        nodes: List[RouteJunctionNode] = []
        cumulative_eta = 30 # Initial dispatch reaction time

        for i in range(len(shortest_path)):
            curr_jid = shortest_path[i]
            prev_jid = shortest_path[i - 1] if i > 0 else curr_jid
            approach, corridor = CORRIDOR_MAP.get((prev_jid, curr_jid), (ApproachEnum.NORTH, "Main Arterial"))

            nodes.append(RouteJunctionNode(
                junction_id=curr_jid,
                approach=approach,
                corridor_name=corridor,
                eta_seconds=cumulative_eta,
                preemption_active=(i == 0) # First node preemption active immediately
            ))
            cumulative_eta += 60 # ~60s average transit per city sector

        return nodes

    @staticmethod
    def resolve_conflicts(
        active_missions: List[AmbulanceMissionResponse],
        junction_id: str
    ) -> ConflictResolutionResult:
        """
        Multi-Ambulance Priority Conflict Resolver.
        Evaluates competing emergency vehicles at the given junction and awards Green Wave priority.
        """
        competing = [
            m for m in active_missions 
            if m.status in [AmbulanceStatusEnum.DISPATCHED_TO_VICTIM, AmbulanceStatusEnum.TRANSIT_TO_HOSPITAL]
            and any(node.junction_id == junction_id for node in m.route_corridor)
        ]

        if len(competing) <= 1:
            return ConflictResolutionResult(
                has_conflict=False,
                junction_id=junction_id,
                strategy="Single or no ambulance at junction; direct green wave clearance active."
            )

        # Sort by priority level descending (4 > 3 > 2 > 1), then by ETA
        competing.sort(key=lambda m: (m.priority_level, -m.estimated_total_eta_seconds), reverse=True)

        winner = competing[0]
        secondary = competing[1]

        winner_node = next((n for n in winner.route_corridor if n.junction_id == junction_id), None)
        secondary_node = next((n for n in secondary.route_corridor if n.junction_id == junction_id), None)

        winner_app = winner_node.approach if winner_node else ApproachEnum.NORTH
        secondary_app = secondary_node.approach if secondary_node else ApproachEnum.EAST

        strategy = (
            f"PRIORITY OVERRIDE: Mission {winner.mission_id} ({winner.criticality.value}, Priority {winner.priority_level}) "
            f"granted immediate GREEN on {winner_app.value}. "
            f"Mission {secondary.mission_id} ({secondary.criticality.value}, Priority {secondary.priority_level}) "
            f"queued for secondary expedited clearance on {secondary_app.value}."
        )

        return ConflictResolutionResult(
            has_conflict=True,
            junction_id=junction_id,
            winning_mission_id=winner.mission_id,
            winning_criticality=winner.criticality,
            winning_approach=winner_app,
            secondary_mission_id=secondary.mission_id,
            secondary_approach=secondary_app,
            strategy=strategy
        )

    @classmethod
    def get_preemption_for_junction(
        cls,
        junction_id: str,
        active_missions: List[AmbulanceMissionResponse]
    ) -> AmbulancePreemptionStatus:
        conflict_res = cls.resolve_conflicts(active_missions, junction_id)

        if not conflict_res.has_conflict:
            # Check if single ambulance is targeting this junction
            matching = [
                m for m in active_missions 
                if m.status in [AmbulanceStatusEnum.DISPATCHED_TO_VICTIM, AmbulanceStatusEnum.TRANSIT_TO_HOSPITAL]
                and any(n.junction_id == junction_id for n in m.route_corridor)
            ]
            if matching:
                target_mission = matching[0]
                node = next(n for n in target_mission.route_corridor if n.junction_id == junction_id)
                return AmbulancePreemptionStatus(
                    junction_id=junction_id,
                    is_preempted=True,
                    active_mission_id=target_mission.mission_id,
                    priority_level=target_mission.priority_level,
                    preempted_approach=node.approach,
                    clearing_phase_duration_seconds=45,
                    advisory=f"EMERGENCY GREEN WAVE ACTIVE for {target_mission.hospital_name} ({target_mission.criticality.value})."
                )
            return AmbulancePreemptionStatus(
                junction_id=junction_id,
                is_preempted=False,
                advisory="Standard adaptive traffic signals in operation."
            )

        # In conflict resolution mode
        return AmbulancePreemptionStatus(
            junction_id=junction_id,
            is_preempted=True,
            active_mission_id=conflict_res.winning_mission_id,
            priority_level=CRITICALITY_PRIORITY_MAP.get(conflict_res.winning_criticality, 4),
            preempted_approach=conflict_res.winning_approach,
            clearing_phase_duration_seconds=45,
            advisory=conflict_res.strategy
        )

ambulance_engine = AmbulanceEngine()
