"""
decision_engine.py
High-level interface and facade for the 4-approach traffic signal decision engine.
Ingests traffic metrics from external vision/aggregator feeds and outputs signal decisions.
"""

from typing import Dict, Optional, Any
from backend.decisionbackend.junction_config import JunctionConfig
from backend.decisionbackend.models import (
    Approach,
    DirectionTraffic,
    SignalDecision,
)
from backend.decisionbackend.signal_controller import SignalController
from backend.decisionbackend.pcu import calculate_queue_pcu


class DecisionEngine:
    """
    Facade class managing the traffic decision engine lifecycle.
    """

    def __init__(self, config: Optional[JunctionConfig] = None, initial_green: Optional[Approach] = Approach.NORTH):
        self.config = config or JunctionConfig()
        self.controller = SignalController(config=self.config, initial_green=initial_green)

    def process_tick(
        self,
        traffic_inputs: Dict[Approach, DirectionTraffic],
        dt: Optional[float] = None
    ) -> SignalDecision:
        """
        Processes one decision cycle with structured DirectionTraffic objects.
        """
        return self.controller.step(traffic_inputs, dt=dt)

    def process_raw_counts(
        self,
        approach_data: Dict[str, Dict[str, Any]],
        dt: Optional[float] = None
    ) -> SignalDecision:
        """
        Convenience ingestion method accepting dictionary of raw counts.
        
        Example approach_data format:
        {
            "NORTH": {"vehicle_counts": {"car": 5, "bus": 1}, "flow_rate": 4.0, "time_since_last_vehicle": 1.0},
            "SOUTH": {"vehicle_counts": {"two_wheeler": 10}, "flow_rate": 2.0, "time_since_last_vehicle": 5.0},
            ...
        }
        """
        inputs = {}
        for app in [Approach.NORTH, Approach.SOUTH, Approach.EAST, Approach.WEST]:
            raw = approach_data.get(app.value, approach_data.get(app, {}))
            v_counts = raw.get("vehicle_counts", {})
            q_pcu = raw.get("queue_pcu")
            if q_pcu is None:
                q_pcu = calculate_queue_pcu(v_counts, self.config.pcu)

            flow = float(raw.get("flow_rate", 0.0))
            time_since_pass = float(raw.get("time_since_last_vehicle", 0.0))
            crossed_recent = int(raw.get("vehicles_crossed_recently", 0))

            inputs[app] = DirectionTraffic(
                direction=app,
                vehicle_counts=v_counts,
                queue_pcu=q_pcu,
                wait_time=self.controller.state.wait_times[app],
                flow_rate=flow,
                vehicles_waiting=sum(v_counts.values()),
                vehicles_crossed_recently=crossed_recent,
                time_since_last_vehicle_passed=time_since_pass
            )

        return self.process_tick(inputs, dt=dt)
