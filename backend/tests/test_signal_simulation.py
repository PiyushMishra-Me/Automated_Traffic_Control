import pytest

from backend.core.control.signal_simulation import APPROACHES, TrafficSimulator
from backend.models.traffic_schemas import (
    ApproachEnum,
    ApproachTrafficState,
    JunctionTrafficState,
    TrafficLevelEnum,
)


def _busy_state() -> JunctionTrafficState:
    """North/South heavily loaded, East/West nearly empty (unbalanced demand)."""
    return JunctionTrafficState(
        junction_id="J-SIM",
        north=ApproachTrafficState(approach=ApproachEnum.NORTH, vehicle_count=12, density=0.6, estimated_queue_length=9, traffic_level=TrafficLevelEnum.HIGH),
        south=ApproachTrafficState(approach=ApproachEnum.SOUTH, vehicle_count=8, density=0.4, estimated_queue_length=6, traffic_level=TrafficLevelEnum.MEDIUM),
        east=ApproachTrafficState(approach=ApproachEnum.EAST, vehicle_count=2, density=0.1, estimated_queue_length=1, traffic_level=TrafficLevelEnum.LOW),
        west=ApproachTrafficState(approach=ApproachEnum.WEST, vehicle_count=1, density=0.05, estimated_queue_length=0, traffic_level=TrafficLevelEnum.LOW),
    )


def test_timeline_shape_and_nonnegative_queues():
    result = TrafficSimulator.run(_busy_state(), horizon=90)

    assert result.total_seconds == 90
    assert len(result.steps) == 90
    assert [step.t for step in result.steps] == list(range(90))

    for step in result.steps:
        assert set(step.lights.keys()) == set(APPROACHES)
        assert all(color in {"GREEN", "YELLOW", "RED"} for color in step.lights.values())
        assert all(q >= 0 for q in step.queues.values())

    served_totals = [step.served_total for step in result.steps]
    assert served_totals == sorted(served_totals)  # monotonic non-decreasing
    assert result.seeded_demo is False


def test_served_and_comparison_populated():
    result = TrafficSimulator.run(_busy_state(), horizon=180)

    assert result.comparison.adaptive_served > 0
    assert result.comparison.fixed_served > 0
    assert result.comparison.adaptive_avg_wait >= 0
    assert result.comparison.fixed_avg_wait >= 0

    assert {summary.approach for summary in result.per_approach} == set(ApproachEnum)
    by_approach = {summary.approach: summary for summary in result.per_approach}
    # The busy North approach must see more arrivals than the quiet West one.
    assert by_approach[ApproachEnum.NORTH].arrivals >= by_approach[ApproachEnum.WEST].arrivals
    assert result.recommendation.junction_id == "J-SIM"


def test_simulation_is_deterministic():
    first = TrafficSimulator.run(_busy_state(), horizon=120)
    second = TrafficSimulator.run(_busy_state(), horizon=120)

    assert [step.model_dump() for step in first.steps] == [step.model_dump() for step in second.steps]
    assert first.comparison.model_dump() == second.comparison.model_dump()


def test_empty_junction_seeds_demo_scenario():
    empty = JunctionTrafficState(junction_id="J-EMPTY")
    result = TrafficSimulator.run(empty, horizon=60)

    assert result.seeded_demo is True
    assert len(result.steps) == 60
    assert result.comparison.adaptive_served > 0
