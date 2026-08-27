"""
priority.py
Priority score calculation module.
Computes P(d) = 0.45 * Queue_norm + 0.35 * WaitTime_norm + 0.20 * FlowRate_norm
Ensures priority remains bounded strictly between 0.0 and 1.0.
"""

from typing import Optional, Dict
from backend.decisionbackend.junction_config import PriorityWeights, NormalizationConfig
from backend.decisionbackend.traffic_metrics import normalize_metrics
from backend.decisionbackend.models import Approach, PriorityScore, DirectionTraffic


def calculate_priority_score(
    direction: Approach,
    queue_pcu: float,
    wait_time_sec: float,
    flow_rate_pcu_min: float,
    weights: Optional[PriorityWeights] = None,
    norm_config: Optional[NormalizationConfig] = None
) -> PriorityScore:
    """
    Compute P(d) for a single approach:
    P(d) = w_queue * Queue_norm + w_wait * WaitTime_norm + w_flow * FlowRate_norm
    """
    if weights is None:
        weights = PriorityWeights()

    norm = normalize_metrics(queue_pcu, wait_time_sec, flow_rate_pcu_min, norm_config)

    score = (
        weights.w_queue * norm.queue_norm +
        weights.w_wait * norm.wait_norm +
        weights.w_flow * norm.flow_norm
    )
    score_clamped = min(max(round(score, 4), 0.0), 1.0)

    return PriorityScore(
        direction=direction,
        score=score_clamped,
        queue_norm=norm.queue_norm,
        wait_norm=norm.wait_norm,
        flow_norm=norm.flow_norm
    )


def compute_all_priorities(
    traffic_by_direction: Dict[Approach, DirectionTraffic],
    weights: Optional[PriorityWeights] = None,
    norm_config: Optional[NormalizationConfig] = None
) -> Dict[Approach, PriorityScore]:
    """
    Calculate priority score for each of the four junction approaches.
    """
    scores = {}
    for approach in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]:
        traffic = traffic_by_direction.get(approach)
        if traffic:
            scores[approach] = calculate_priority_score(
                direction=approach,
                queue_pcu=traffic.queue_pcu,
                wait_time_sec=traffic.wait_time,
                flow_rate_pcu_min=traffic.flow_rate,
                weights=weights,
                norm_config=norm_config
            )
        else:
            scores[approach] = PriorityScore(
                direction=approach,
                score=0.0,
                queue_norm=0.0,
                wait_norm=0.0,
                flow_norm=0.0
            )
    return scores
