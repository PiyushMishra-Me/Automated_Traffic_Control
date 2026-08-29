"""
traffic_metrics.py
Traffic metric normalization module.
Converts heterogeneous traffic metrics (Queue PCU, Wait Time in seconds, Flow Rate in PCU/min)
into unitless normalized representations in the range [0.0, 1.0].
"""

from typing import Optional
from backend.decisionbackend.junction_config import NormalizationConfig
from backend.decisionbackend.models import NormalizedMetrics


def normalize_queue_pcu(queue_pcu: float, config: Optional[NormalizationConfig] = None) -> float:
    """
    Queue_norm = min(Queue_PCU / QUEUE_PCU_MAX, 1.0)
    """
    if config is None:
        config = NormalizationConfig()
    if queue_pcu <= 0.0 or config.queue_pcu_max <= 0.0:
        return 0.0
    return min(float(queue_pcu) / float(config.queue_pcu_max), 1.0)


def normalize_wait_time(wait_time_sec: float, config: Optional[NormalizationConfig] = None) -> float:
    """
    WaitTime_norm = min(WaitTime / WAIT_TIME_REF, 1.0)
    """
    if config is None:
        config = NormalizationConfig()
    if wait_time_sec <= 0.0 or config.wait_time_ref <= 0.0:
        return 0.0
    return min(float(wait_time_sec) / float(config.wait_time_ref), 1.0)


def normalize_flow_rate(flow_rate_pcu_min: float, config: Optional[NormalizationConfig] = None) -> float:
    """
    FlowRate_norm = min(FlowRate / FLOW_RATE_MAX, 1.0)
    """
    if config is None:
        config = NormalizationConfig()
    if flow_rate_pcu_min <= 0.0 or config.flow_rate_max <= 0.0:
        return 0.0
    return min(float(flow_rate_pcu_min) / float(config.flow_rate_max), 1.0)


def normalize_metrics(
    queue_pcu: float,
    wait_time_sec: float,
    flow_rate_pcu_min: float,
    config: Optional[NormalizationConfig] = None
) -> NormalizedMetrics:
    """
    Convenience function to normalize all three raw metrics for an approach.
    """
    q_norm = normalize_queue_pcu(queue_pcu, config)
    w_norm = normalize_wait_time(wait_time_sec, config)
    f_norm = normalize_flow_rate(flow_rate_pcu_min, config)
    return NormalizedMetrics(queue_norm=q_norm, wait_norm=w_norm, flow_norm=f_norm)
