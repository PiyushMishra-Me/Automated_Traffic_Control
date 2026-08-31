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


def test_forced_red_approaches():
    # When EAST is forced RED, it should NEVER show GREEN and discharge 0 vehicles
    result = TrafficSimulator.run(_busy_state(), horizon=120, forced_red_approaches=["EAST"])

    for step in result.steps:
        assert step.lights["EAST"] == "RED"

    east_summary = next(s for s in result.per_approach if s.approach == ApproachEnum.EAST)
    assert east_summary.served == 0
    assert "Manual RED override active on: EAST" in result.rationale


def test_corridor_simulation():
    from backend.core.control.signal_simulation import CorridorTrafficSimulator
    from backend.models.traffic_schemas import CorridorLink

    j1 = _busy_state()
    j2 = JunctionTrafficState(
        junction_id="J-02",
        north=ApproachTrafficState(approach=ApproachEnum.NORTH, vehicle_count=4, density=0.2, estimated_queue_length=3, traffic_level=TrafficLevelEnum.LOW),
        south=ApproachTrafficState(approach=ApproachEnum.SOUTH, vehicle_count=3, density=0.15, estimated_queue_length=2, traffic_level=TrafficLevelEnum.LOW),
        east=ApproachTrafficState(approach=ApproachEnum.EAST, vehicle_count=5, density=0.25, estimated_queue_length=4, traffic_level=TrafficLevelEnum.MEDIUM),
        west=ApproachTrafficState(approach=ApproachEnum.WEST, vehicle_count=1, density=0.05, estimated_queue_length=0, traffic_level=TrafficLevelEnum.LOW),
    )

    states = {"J-SIM": j1, "J-02": j2}
    link = CorridorLink(
        upstream_junction_id="J-SIM",
        upstream_approach=ApproachEnum.EAST,
        downstream_junction_id="J-02",
        downstream_approach=ApproachEnum.WEST,
        distance_km=2.5,
        transit_time_seconds=5,
    )

    result = CorridorTrafficSimulator.run_corridor(
        junction_states=states,
        junction_ids=["J-SIM", "J-02"],
        links=[link],
        horizon=100,
    )

    assert result.total_seconds == 100
    assert len(result.steps) == 100
    assert len(result.junction_ids) == 2
    assert "J-SIM" in result.junction_results
    assert "J-02" in result.junction_results
    assert len(result.links) == 1
    assert result.corridor_comparison.adaptive_avg_wait >= 0


def test_5_junction_star_network_simulation():
    from backend.core.control.signal_simulation import CorridorTrafficSimulator

    # 5 Junction network: J-01 Central with surrounding J-02 (East), J-03 (North), J-04 (South), J-05 (West)
    junction_ids = ["J-01", "J-02", "J-03", "J-04", "J-05"]
    states = {
        j_id: JunctionTrafficState(
            junction_id=j_id,
            north=ApproachTrafficState(approach=ApproachEnum.NORTH, vehicle_count=6, density=0.3, estimated_queue_length=4, traffic_level=TrafficLevelEnum.MEDIUM),
            south=ApproachTrafficState(approach=ApproachEnum.SOUTH, vehicle_count=5, density=0.25, estimated_queue_length=3, traffic_level=TrafficLevelEnum.MEDIUM),
            east=ApproachTrafficState(approach=ApproachEnum.EAST, vehicle_count=7, density=0.35, estimated_queue_length=5, traffic_level=TrafficLevelEnum.MEDIUM),
            west=ApproachTrafficState(approach=ApproachEnum.WEST, vehicle_count=4, density=0.2, estimated_queue_length=2, traffic_level=TrafficLevelEnum.LOW),
        )
        for j_id in junction_ids
    }

    result = CorridorTrafficSimulator.run_corridor(
        junction_states=states,
        junction_ids=junction_ids,
        forced_red={"J-01": ["NORTH"]},
        horizon=60,
    )

    assert result.total_seconds == 60
    assert len(result.steps) == 60
    assert len(result.junction_ids) == 5
    assert len(result.links) >= 4  # Auto-discovered directional links across North, South, East, West
    assert "J-01" in result.junction_results
    assert "J-05" in result.junction_results
    assert result.corridor_comparison.adaptive_served > 0
    # J-01 North should be locked RED
    for step in result.steps:
        assert step.junctions["J-01"].lights["NORTH"] == "RED"


