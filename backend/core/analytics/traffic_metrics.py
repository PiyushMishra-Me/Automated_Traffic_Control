from typing import List, Tuple

from backend.config import settings
from backend.core.vision.tracker import TrackedVehicle
from backend.models.traffic_schemas import (
    ApproachEnum,
    ApproachTrafficState,
    MovementStateEnum,
    TrafficLevelEnum,
)


def ccw(
        A: Tuple[float, float],
        B: Tuple[float, float],
        C: Tuple[float, float]
) -> bool:
    """Check counter-clockwise order for line segment intersection."""
    return (
            (C[1] - A[1]) * (B[0] - A[0])
            > (B[1] - A[1]) * (C[0] - A[0])
    )


def intersect(
        A: Tuple[float, float],
        B: Tuple[float, float],
        C: Tuple[float, float],
        D: Tuple[float, float]
) -> bool:
    """Return True if line segment AB and CD intersect."""
    return (
            ccw(A, C, D) != ccw(B, C, D)
            and ccw(A, B, C) != ccw(A, B, D)
    )


class TrafficMetricsCalculator:
    """
    Computes real-time approach-specific traffic statistics
    from tracked vehicles, including ambulances.
    """

    def __init__(self, approach: ApproachEnum, line_config: dict = None):
        self.approach = approach

        self.line_config = line_config or settings.DEFAULT_COUNTING_LINES.get(
            approach.value if hasattr(approach, "value") else str(approach),
            {
                "p1": [0.1, 0.5],
                "p2": [0.9, 0.5],
                "orientation": "horizontal",
            },
        )

        self.crossed_ids: set[int] = set()
        self.incoming_crossed_ids: set[int] = set()
        self.outgoing_crossed_ids: set[int] = set()
        self.all_seen_ids: set[int] = set()

    def reset(self):
        self.crossed_ids.clear()
        self.incoming_crossed_ids.clear()
        self.outgoing_crossed_ids.clear()
        self.all_seen_ids.clear()

    def update_flow_counting(
            self,
            vehicles: List[TrackedVehicle],
            frame_width: int,
            frame_height: int,
    ) -> int:
        """
        Check whether vehicle trajectories crossed the
        configured virtual counting line.
        """

        p1 = (
            self.line_config["p1"][0] * frame_width,
            self.line_config["p1"][1] * frame_height,
        )

        p2 = (
            self.line_config["p2"][0] * frame_width,
            self.line_config["p2"][1] * frame_height,
        )

        for v in vehicles:
            self.all_seen_ids.add(v.track_id)

            if v.track_id in self.crossed_ids:
                continue

            if v.previous_center is not None:
                if intersect(v.previous_center, v.center, p1, p2):
                    self.crossed_ids.add(v.track_id)
                    v.crossed_counting_line = True

                    v_dir = getattr(v, "direction", None)

                    if v_dir in (
                            MovementStateEnum.OUTGOING,
                            MovementStateEnum.STOPPED_OUTGOING,
                    ):
                        self.outgoing_crossed_ids.add(v.track_id)
                    else:
                        self.incoming_crossed_ids.add(v.track_id)

        return len(self.crossed_ids)

    def calculate_metrics(
            self,
            vehicles: List[TrackedVehicle],
            frame_width: int,
            frame_height: int,
            processed_frames: int = 1,
            fps: float = 30.0,
    ) -> ApproachTrafficState:
        """
        Calculate traffic metrics for the current frame.
        """

        # 1. Active vehicles
        active_vehicles = [
            v
            for v in vehicles
            if not getattr(v, "is_parked", False)
               and getattr(v, "direction", None) != MovementStateEnum.PARKED
        ]

        active_count = len(active_vehicles)

        # 2. Count all supported vehicle classes, including ambulance
        class_counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
            "ambulance": 0,
        }

        total_bbox_area = 0.0

        for v in active_vehicles:
            if v.class_name in class_counts:
                class_counts[v.class_name] += 1

            w = max(0.0, v.xyxy[2] - v.xyxy[0])
            h = max(0.0, v.xyxy[3] - v.xyxy[1])
            total_bbox_area += w * h

        # 3. Flow counting
        flow_count = self.update_flow_counting(
            vehicles,
            frame_width,
            frame_height,
        )

        # 4. Density
        frame_area = max(1.0, float(frame_width * frame_height))

        area_occupancy = min(
            1.0,
            (total_bbox_area * 1.5) / frame_area,
            )

        count_density = min(
            1.0,
            active_count / float(settings.THRESHOLD_HIGH),
            )

        density = round(
            0.5 * area_occupancy + 0.5 * count_density,
            3,
            )

        # 5. Queue length
        estimated_queue = sum(
            1
            for v in active_vehicles
            if v.stationary_frames >= 5
            or v.speed_px < settings.QUEUE_SPEED_THRESHOLD
        )

        # 6. Directional counts
        incoming_count = sum(
            1
            for v in active_vehicles
            if getattr(v, "direction", None)
            in (
                MovementStateEnum.INCOMING,
                MovementStateEnum.STOPPED_INCOMING,
            )
        )

        outgoing_count = sum(
            1
            for v in active_vehicles
            if getattr(v, "direction", None)
            in (
                MovementStateEnum.OUTGOING,
                MovementStateEnum.STOPPED_OUTGOING,
            )
        )

        stopped_incoming_count = sum(
            1
            for v in active_vehicles
            if getattr(v, "direction", None)
            == MovementStateEnum.STOPPED_INCOMING
        )

        stopped_outgoing_count = sum(
            1
            for v in active_vehicles
            if getattr(v, "direction", None)
            == MovementStateEnum.STOPPED_OUTGOING
        )

        parked_count = sum(
            1
            for v in vehicles
            if getattr(v, "is_parked", False)
            or getattr(v, "direction", None)
            == MovementStateEnum.PARKED
        )

        unknown_direction_count = sum(
            1
            for v in active_vehicles
            if getattr(v, "direction", None)
            in (MovementStateEnum.UNKNOWN, None)
        )

        # 7. Traffic level
        if (
                active_count <= settings.THRESHOLD_LOW
                and density < 0.25
        ):
            level = TrafficLevelEnum.LOW

        elif (
                active_count <= settings.THRESHOLD_MEDIUM
                and density < 0.50
        ):
            level = TrafficLevelEnum.MEDIUM

        elif (
                active_count <= settings.THRESHOLD_HIGH
                and density < 0.75
        ):
            level = TrafficLevelEnum.HIGH

        else:
            level = TrafficLevelEnum.VERY_HIGH

        return ApproachTrafficState(
            approach=self.approach,
            vehicle_count=active_count,
            class_counts=class_counts,
            density=density,
            estimated_queue_length=estimated_queue,
            flow=float(flow_count),
            traffic_level=level,
            processed_frames=processed_frames,
            total_unique_vehicles=len(self.all_seen_ids),
            incoming_count=incoming_count,
            outgoing_count=outgoing_count,
            stopped_incoming_count=stopped_incoming_count,
            stopped_outgoing_count=stopped_outgoing_count,
            parked_count=parked_count,
            unknown_direction_count=unknown_direction_count,
            incoming_flow=float(len(self.incoming_crossed_ids)),
            outgoing_flow=float(len(self.outgoing_crossed_ids)),
        )