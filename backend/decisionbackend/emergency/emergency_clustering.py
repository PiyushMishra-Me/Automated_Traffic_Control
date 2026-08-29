"""
emergency_clustering.py
Multi-Emergency ETA Clustering, Transitive Grouping, Priority Boost, and Conflict Resolution (Phase 3).

Rules:
1. Clustering: Two emergency ETAs are in the same cluster if |ETA_A - ETA_B| <= 10.0s.
2. Transitive: Evaluated sequentially using the most recently active/preceding cluster member.
3. Priority Boost: For pending emergencies in a cluster (size >= 2), queue weight is boosted from 0.45 to 0.90.
   P_emergency(d) = 0.90 * Queue_norm + 0.35 * WaitTime_norm + 0.20 * FlowRate_norm
4. Conflict Resolution:
   1. Lower current live ETA
   2. Shorter T_clear
   3. Higher WaitTime_norm
   4. Direction rank: NORTH > EAST > SOUTH > WEST
"""

from typing import Dict, List, Optional, Tuple
from backend.decisionbackend.models import (
    Approach,
    NormalizedMetrics,
    PriorityScore,
)
from backend.decisionbackend.junction_config import (
    NormalizationConfig,
    PriorityWeights,
)
from backend.decisionbackend.emergency.emergency_models import (
    EmergencyNotice,
    EmergencyState,
)
from backend.decisionbackend.emergency.emergency_clearance import (
    calculate_t_clear,
    check_emergency_trigger_conditions,
    DEFAULT_EMPTY_QUEUE_THRESHOLD_PCU,
)


CLUSTER_ETA_THRESHOLD_SECONDS: float = 10.0
BOOSTED_QUEUE_WEIGHT: float = 0.90
NORMAL_QUEUE_WEIGHT: float = 0.45

# Direction priority order for complete tie-breaks: NORTH > EAST > SOUTH > WEST
DIRECTION_TIE_BREAK_ORDER: Dict[Approach, int] = {
    Approach.NORTH: 0,
    Approach.EAST: 1,
    Approach.SOUTH: 2,
    Approach.WEST: 3,
}


def sort_emergencies_by_eta(notices: List[EmergencyNotice]) -> List[EmergencyNotice]:
    """
    Sorts a list of emergency notices by their CURRENT live ETA ascending.
    """
    return sorted(notices, key=lambda n: (n.current_eta, n.emergency_id))


def form_eta_clusters(
    notices: List[EmergencyNotice],
    threshold: float = CLUSTER_ETA_THRESHOLD_SECONDS
) -> List[List[EmergencyNotice]]:
    """
    Groups sorted emergency notices into transitive ETA clusters.
    Two consecutive notices belong to the same cluster if:
      n[i].current_eta - n[i-1].current_eta <= threshold (10.0s)
    """
    active_pending = [
        n for n in notices
        if n.state in (EmergencyState.PENDING, EmergencyState.ACTIVE) and not n.is_passed
    ]
    if not active_pending:
        return []

    sorted_notices = sort_emergencies_by_eta(active_pending)
    clusters: List[List[EmergencyNotice]] = []
    current_cluster: List[EmergencyNotice] = [sorted_notices[0]]

    for i in range(1, len(sorted_notices)):
        prev = sorted_notices[i - 1]
        curr = sorted_notices[i]
        if (curr.current_eta - prev.current_eta) <= float(threshold):
            current_cluster.append(curr)
        else:
            clusters.append(current_cluster)
            current_cluster = [curr]

    clusters.append(current_cluster)
    return clusters


def get_clustered_approaches(
    notices: List[EmergencyNotice],
    threshold: float = CLUSTER_ETA_THRESHOLD_SECONDS
) -> Dict[Approach, List[EmergencyNotice]]:
    """
    Returns mapping of approaches that belong to a same-ETA cluster of size >= 2.
    """
    clusters = form_eta_clusters(notices, threshold=threshold)
    clustered_map: Dict[Approach, List[EmergencyNotice]] = {}
    for cluster in clusters:
        if len(cluster) >= 2:
            for notice in cluster:
                if notice.approach not in clustered_map:
                    clustered_map[notice.approach] = []
                clustered_map[notice.approach].append(notice)
    return clustered_map


def compute_emergency_priority_score(
    approach: Approach,
    norm: NormalizedMetrics,
    is_clustered: bool = False,
    normal_weights: Optional[PriorityWeights] = None
) -> PriorityScore:
    """
    Computes priority score P(d) with temporary 2x queue boost (0.90) for clustered emergency approaches.
    
    Normal:      0.45 * Q_norm + 0.35 * W_norm + 0.20 * F_norm
    Clustered:   0.90 * Q_norm + 0.35 * W_norm + 0.20 * F_norm
    """
    weights = normal_weights or PriorityWeights()
    w_queue = BOOSTED_QUEUE_WEIGHT if is_clustered else weights.w_queue
    w_wait = weights.w_wait
    w_flow = weights.w_flow

    score = (w_queue * norm.queue_norm) + (w_wait * norm.wait_norm) + (w_flow * norm.flow_norm)

    return PriorityScore(
        direction=approach,
        score=score,
        queue_norm=norm.queue_norm,
        wait_norm=norm.wait_norm,
        flow_norm=norm.flow_norm
    )


def resolve_emergency_conflict(
    candidate_notices: List[Tuple[EmergencyNotice, float]],  # List of (notice, t_clear)
    wait_norms: Dict[Approach, float]
) -> Optional[Tuple[EmergencyNotice, float]]:
    """
    Resolves conflict among multiple simultaneously eligible/triggered emergencies using exact 4-tier order:
    1. LOWER current live ETA
    2. SHORTER T_clear
    3. HIGHER WaitTime_norm
    4. Deterministic direction rank: NORTH > EAST > SOUTH > WEST
    """
    if not candidate_notices:
        return None

    def conflict_key(item: Tuple[EmergencyNotice, float]):
        notice, t_clear = item
        w_norm = wait_norms.get(notice.approach, 0.0)
        dir_rank = DIRECTION_TIE_BREAK_ORDER.get(notice.approach, 99)
        # We minimize key: lower ETA, lower T_clear, lower (-w_norm) -> higher w_norm, lower dir_rank
        return (notice.current_eta, t_clear, -w_norm, dir_rank)

    return min(candidate_notices, key=conflict_key)
