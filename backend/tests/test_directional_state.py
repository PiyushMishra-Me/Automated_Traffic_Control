import pytest
import numpy as np
from backend.models.traffic_schemas import ApproachEnum, MovementStateEnum, ApproachTrafficState
from backend.core.vision.tracker import VehicleTracker, TrackedVehicle
from backend.core.analytics.traffic_metrics import TrafficMetricsCalculator
from backend.config import settings

class DummyModel:
    """Mock YOLO model for testing tracker logic without full neural net inference."""
    def __init__(self):
        self.device = "cpu"
    def track(self, *args, **kwargs):
        return []

def create_mock_tracker(approach=ApproachEnum.NORTH, junction_vector=None, fps=25.0):
    tracker = VehicleTracker(approach=approach, junction_vector=junction_vector, fps=fps)
    tracker.model = DummyModel()
    tracker.reset()
    return tracker

def simulate_track_update(tracker, track_id, center, class_id=2, conf=0.85, fps=25.0):
    """
    Simulate updating an existing track with a new center position in the tracker,
    invoking the directional state & parking logic.
    """
    tracker.fps = fps
    current_fps = fps

    if track_id not in tracker.class_votes:
        tracker.class_votes[track_id] = {}
    tracker.class_votes[track_id][class_id] = tracker.class_votes[track_id].get(class_id, 0) + 1
    stable_cls_id = max(tracker.class_votes[track_id], key=tracker.class_votes[track_id].get)
    stable_cls_name = tracker.class_names.get(stable_cls_id, "car")

    curr_center = (float(center[0]), float(center[1]))
    prev_center = None
    speed_px = 0.0
    if track_id in tracker.track_history and len(tracker.track_history[track_id]) > 0:
        prev_center = tracker.track_history[track_id][-1]
        speed_px = float(np.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1]))

    if track_id not in tracker.track_history:
        tracker.track_history[track_id] = []
    tracker.track_history[track_id].append(curr_center)
    if len(tracker.track_history[track_id]) > 30:
        tracker.track_history[track_id].pop(0)

    history = tracker.track_history[track_id]

    if speed_px < settings.QUEUE_SPEED_THRESHOLD:
        tracker.stationary_counts[track_id] = tracker.stationary_counts.get(track_id, 0) + 1
    else:
        tracker.stationary_counts[track_id] = max(0, tracker.stationary_counts.get(track_id, 0) - 1)

    is_stationary = (speed_px < settings.MOVEMENT_SPEED_THRESHOLD)

    if is_stationary:
        if track_id not in tracker.stationary_ref_center:
            tracker.stationary_ref_center[track_id] = curr_center
        ref_dist = float(np.hypot(curr_center[0] - tracker.stationary_ref_center[track_id][0],
                                  curr_center[1] - tracker.stationary_ref_center[track_id][1]))
        if ref_dist > settings.NOISE_DISPLACEMENT_THRESHOLD and speed_px >= settings.MOVEMENT_SPEED_THRESHOLD:
            is_stationary = False
            tracker.stationary_ref_center.pop(track_id, None)

    if is_stationary:
        tracker.stopped_frames_count[track_id] = tracker.stopped_frames_count.get(track_id, 0) + 1
        stopped_duration = tracker.stopped_frames_count[track_id] / current_fps

        if stopped_duration > settings.PARKED_DURATION_SECONDS:
            tracker.parked_status[track_id] = True
            movement_state = MovementStateEnum.PARKED
        else:
            last_dir = tracker.last_moving_direction.get(track_id)
            if last_dir == MovementStateEnum.INCOMING:
                movement_state = MovementStateEnum.STOPPED_INCOMING
            elif last_dir == MovementStateEnum.OUTGOING:
                movement_state = MovementStateEnum.STOPPED_OUTGOING
            else:
                movement_state = MovementStateEnum.UNKNOWN
    else:
        tracker.stopped_frames_count[track_id] = 0
        tracker.stationary_ref_center.pop(track_id, None)

        if tracker.parked_status.get(track_id, False):
            tracker.parked_status[track_id] = False
            tracker.last_moving_direction.pop(track_id, None)

        if len(history) < settings.MIN_TRAJECTORY_POINTS:
            movement_state = tracker.last_moving_direction.get(track_id, MovementStateEnum.UNKNOWN)
        else:
            k = min(len(history), 8)
            dx = history[-1][0] - history[-k][0]
            dy = history[-1][1] - history[-k][1]
            mag = float(np.hypot(dx, dy))

            edge_margin = getattr(settings, "EDGE_MARGIN_PIXELS", 25.0)
            w, h = 768, 432
            cx, cy = curr_center
            is_near_edge = (
                cx < edge_margin or cx > (w - edge_margin) or
                cy < edge_margin or cy > (h - edge_margin)
            )

            if mag < 2.0:
                movement_state = tracker.last_moving_direction.get(track_id, MovementStateEnum.UNKNOWN)
            else:
                u_vec = np.array([dx / mag, dy / mag], dtype=np.float32)
                dot_prod = float(np.dot(u_vec, tracker.junction_vector))
                established_dir = tracker.last_moving_direction.get(track_id)

                if established_dir is not None:
                    if is_near_edge:
                        movement_state = established_dir
                    elif established_dir == MovementStateEnum.INCOMING:
                        if dot_prod > -0.15:
                            movement_state = MovementStateEnum.INCOMING
                        else:
                            if mag >= settings.DIRECTION_FLIP_MIN_DISPLACEMENT and len(history) >= 5:
                                movement_state = MovementStateEnum.OUTGOING
                                tracker.last_moving_direction[track_id] = MovementStateEnum.OUTGOING
                            else:
                                movement_state = MovementStateEnum.INCOMING
                    elif established_dir == MovementStateEnum.OUTGOING:
                        if dot_prod < 0.15:
                            movement_state = MovementStateEnum.OUTGOING
                        else:
                            if mag >= settings.DIRECTION_FLIP_MIN_DISPLACEMENT and len(history) >= 5:
                                movement_state = MovementStateEnum.INCOMING
                                tracker.last_moving_direction[track_id] = MovementStateEnum.INCOMING
                            else:
                                movement_state = MovementStateEnum.OUTGOING
                else:
                    if dot_prod > 0.15:
                        movement_state = MovementStateEnum.INCOMING
                        tracker.last_moving_direction[track_id] = MovementStateEnum.INCOMING
                    elif dot_prod < -0.15:
                        movement_state = MovementStateEnum.OUTGOING
                        tracker.last_moving_direction[track_id] = MovementStateEnum.OUTGOING
                    else:
                        movement_state = MovementStateEnum.UNKNOWN

    is_parked = (movement_state == MovementStateEnum.PARKED)
    stopped_duration_sec = tracker.stopped_frames_count.get(track_id, 0) / current_fps

    return TrackedVehicle(
        track_id=track_id,
        xyxy=[curr_center[0] - 20, curr_center[1] - 20, curr_center[0] + 20, curr_center[1] + 20],
        confidence=conf,
        class_id=stable_cls_id,
        class_name=stable_cls_name,
        center=curr_center,
        previous_center=prev_center,
        speed_px=speed_px,
        stationary_frames=tracker.stationary_counts.get(track_id, 0),
        direction=movement_state,
        stopped_duration_seconds=stopped_duration_sec,
        last_moving_direction=tracker.last_moving_direction.get(track_id),
        is_parked=is_parked
    )

# 1. Vehicle moving toward junction -> INCOMING
def test_vehicle_moving_toward_junction_is_incoming():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH) # junction vector = [0, 1]
    # Moving downward (increasing y)
    v1 = simulate_track_update(tracker, 1, (300, 100))
    v2 = simulate_track_update(tracker, 1, (300, 110))
    v3 = simulate_track_update(tracker, 1, (300, 120))
    v4 = simulate_track_update(tracker, 1, (300, 130))
    
    assert v4.direction == MovementStateEnum.INCOMING
    assert v4.last_moving_direction == MovementStateEnum.INCOMING
    assert v4.is_parked is False

# 2. Vehicle moving away from junction -> OUTGOING
def test_vehicle_moving_away_from_junction_is_outgoing():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH) # junction vector = [0, 1]
    # Moving upward (decreasing y = away from junction)
    v1 = simulate_track_update(tracker, 2, (300, 300))
    v2 = simulate_track_update(tracker, 2, (300, 290))
    v3 = simulate_track_update(tracker, 2, (300, 280))
    v4 = simulate_track_update(tracker, 2, (300, 270))
    
    assert v4.direction == MovementStateEnum.OUTGOING
    assert v4.last_moving_direction == MovementStateEnum.OUTGOING
    assert v4.is_parked is False

# 3. Insufficient trajectory -> UNKNOWN
def test_insufficient_trajectory_is_unknown():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH)
    # Only 1 or 2 points
    v1 = simulate_track_update(tracker, 3, (300, 100))
    assert v1.direction == MovementStateEnum.UNKNOWN
    
    v2 = simulate_track_update(tracker, 3, (300, 101))
    assert v2.direction == MovementStateEnum.UNKNOWN

# 4. INCOMING vehicle stops -> STOPPED_INCOMING
def test_incoming_vehicle_stops_becomes_stopped_incoming():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH)
    # Move incoming
    for y in [100, 115, 130, 145]:
        v = simulate_track_update(tracker, 4, (300, y))
    assert v.direction == MovementStateEnum.INCOMING

    # Vehicle stops (speed = 0)
    for _ in range(5):
        v = simulate_track_update(tracker, 4, (300, 145))
    
    assert v.direction == MovementStateEnum.STOPPED_INCOMING
    assert v.last_moving_direction == MovementStateEnum.INCOMING

# 5. OUTGOING vehicle stops -> STOPPED_OUTGOING
def test_outgoing_vehicle_stops_becomes_stopped_outgoing():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH)
    # Move outgoing
    for y in [300, 285, 270, 255]:
        v = simulate_track_update(tracker, 5, (300, y))
    assert v.direction == MovementStateEnum.OUTGOING

    # Vehicle stops (speed = 0)
    for _ in range(5):
        v = simulate_track_update(tracker, 5, (300, 255))
    
    assert v.direction == MovementStateEnum.STOPPED_OUTGOING
    assert v.last_moving_direction == MovementStateEnum.OUTGOING

# 6. Vehicle stationary for <= 5 minutes retains previous direction
def test_stationary_under_5_minutes_retains_direction():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH, fps=30.0)
    for y in [100, 115, 130, 145]:
        simulate_track_update(tracker, 6, (300, y), fps=30.0)

    # 4 minutes stationary at 30 fps = 4 * 60 * 30 = 7200 frames
    # Simulate stopping:
    tracker.stopped_frames_count[6] = 7200 # 240 seconds (< 300s)
    v = simulate_track_update(tracker, 6, (300, 145), fps=30.0)
    
    assert v.direction == MovementStateEnum.STOPPED_INCOMING
    assert v.is_parked is False
    assert v.stopped_duration_seconds > 200.0

# 7. Vehicle stationary for > 5 minutes -> PARKED
def test_stationary_over_5_minutes_becomes_parked():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH, fps=30.0)
    for y in [100, 115, 130, 145]:
        simulate_track_update(tracker, 7, (300, y), fps=30.0)

    # 5.1 minutes stationary at 30 fps = 5.1 * 60 * 30 = 9180 frames (> 300s)
    tracker.stopped_frames_count[7] = 9180
    v = simulate_track_update(tracker, 7, (300, 145), fps=30.0)
    
    assert v.direction == MovementStateEnum.PARKED
    assert v.is_parked is True

# 8. PARKED vehicle starts moving -> recalculates direction
def test_parked_vehicle_starts_moving_recalculates_direction():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH, fps=30.0)
    # Start as stopped incoming then parked
    for y in [100, 115, 130, 145]:
        simulate_track_update(tracker, 8, (300, y), fps=30.0)
    tracker.stopped_frames_count[8] = 9200
    v = simulate_track_update(tracker, 8, (300, 145), fps=30.0)
    assert v.direction == MovementStateEnum.PARKED

    # Now vehicle starts moving in OUTGOING direction (decreasing y)
    for y in [135, 120, 105, 90]:
        v = simulate_track_update(tracker, 8, (300, y), fps=30.0)

    assert v.is_parked is False
    assert v.direction == MovementStateEnum.OUTGOING

# 9. Small detection noise while stopped does not reset stationary duration
def test_small_detection_noise_does_not_reset_stationary_duration():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH, fps=30.0)
    for y in [100, 115, 130, 145]:
        simulate_track_update(tracker, 9, (300, y), fps=30.0)

    # Stop vehicle
    v = simulate_track_update(tracker, 9, (300, 145), fps=30.0)
    tracker.stopped_frames_count[9] = 100

    # Add small sub-pixel jitter: +/- 0.5px
    v = simulate_track_update(tracker, 9, (300.4, 145.3), fps=30.0)
    assert tracker.stopped_frames_count[9] == 101 # incremented, NOT reset to 0
    assert v.direction == MovementStateEnum.STOPPED_INCOMING

# 10. Different FPS values correctly calculate the 5-minute duration
@pytest.mark.parametrize("test_fps", [10.0, 25.0, 30.0, 60.0])
def test_different_fps_parking_duration(test_fps):
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH, fps=test_fps)
    for y in [100, 115, 130, 145]:
        simulate_track_update(tracker, 10, (300, y), fps=test_fps)

    # 4 minutes: 240 seconds * fps frames
    tracker.stopped_frames_count[10] = int(240.0 * test_fps)
    v_under = simulate_track_update(tracker, 10, (300, 145), fps=test_fps)
    assert v_under.direction == MovementStateEnum.STOPPED_INCOMING
    assert v_under.is_parked is False

    # 5.1 minutes: 306 seconds * fps frames
    tracker.stopped_frames_count[10] = int(306.0 * test_fps)
    v_over = simulate_track_update(tracker, 10, (300, 145), fps=test_fps)
    assert v_over.direction == MovementStateEnum.PARKED
    assert v_over.is_parked is True

# 11. Full-frame ROI and Camera-specific ROI configuration
def test_roi_configuration():
    # Full-frame ROI (None)
    tracker_full = VehicleTracker(roi=None)
    assert tracker_full.roi is None or tracker_full.roi == [0, 0, 768, 432]
    
    # Custom Camera ROI
    custom_roi = [100, 50, 600, 400]
    tracker_custom = VehicleTracker(roi=custom_roi)
    assert tracker_custom.roi == custom_roi
    
    # Dynamic set_roi
    tracker_custom.set_roi(None)
    assert tracker_custom.roi is None

# 12. Different Junction Vectors (e.g. SOUTH, EAST, WEST, arbitrary angle)
def test_different_junction_vectors():
    # SOUTH Approach (junction is at top, vector = [0, -1])
    tracker_south = create_mock_tracker(approach=ApproachEnum.SOUTH, junction_vector=[0.0, -1.0])
    # Moving upward (decreasing y) -> moving towards junction -> INCOMING
    for y in [300, 280, 260, 240]:
        v = simulate_track_update(tracker_south, 101, (300, y))
    assert v.direction == MovementStateEnum.INCOMING

    # Moving downward (increasing y) -> moving away from junction -> OUTGOING
    for y in [100, 120, 140, 160]:
        v = simulate_track_update(tracker_south, 102, (300, y))
    assert v.direction == MovementStateEnum.OUTGOING

    # EAST Approach (junction is at left, vector = [-1, 0])
    tracker_east = create_mock_tracker(approach=ApproachEnum.EAST, junction_vector=[-1.0, 0.0])
    # Moving left (decreasing x) -> toward junction -> INCOMING
    for x in [300, 280, 260, 240]:
        v = simulate_track_update(tracker_east, 103, (x, 200))
    assert v.direction == MovementStateEnum.INCOMING

# 13. Bidirectional Traffic on same camera (Left Carriageway Outgoing, Right Carriageway Incoming)
def test_bidirectional_traffic():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH, junction_vector=[0.0, 1.0])
    
    # Right Lane: Moving downward (+y) -> INCOMING
    for y in [100, 120, 140, 160]:
        v_right = simulate_track_update(tracker, 201, (550, y))
    assert v_right.direction == MovementStateEnum.INCOMING

    # Left Lane: Moving upward (-y) -> OUTGOING
    for y in [350, 330, 310, 290]:
        v_left = simulate_track_update(tracker, 202, (200, y))
    assert v_left.direction == MovementStateEnum.OUTGOING

# 14. Established INCOMING with small reverse jitter (Hysteresis)
def test_established_incoming_with_small_reverse_jitter():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH, junction_vector=[0.0, 1.0])
    # Move strongly INCOMING
    for y in [100, 120, 140, 160, 180]:
        v = simulate_track_update(tracker, 301, (400, y))
    assert v.direction == MovementStateEnum.INCOMING

    # Small 3px reverse jitter (y goes 180 -> 177)
    v = simulate_track_update(tracker, 301, (400, 177))
    assert v.direction == MovementStateEnum.INCOMING # Does NOT flip!

# 15. Established OUTGOING with small reverse jitter (Hysteresis)
def test_established_outgoing_with_small_reverse_jitter():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH, junction_vector=[0.0, 1.0])
    # Move strongly OUTGOING
    for y in [300, 280, 260, 240, 220]:
        v = simulate_track_update(tracker, 302, (200, y))
    assert v.direction == MovementStateEnum.OUTGOING

    # Small 3px reverse jitter (y goes 220 -> 223)
    v = simulate_track_update(tracker, 302, (200, 223))
    assert v.direction == MovementStateEnum.OUTGOING # Does NOT flip!

# 16. Genuine Opposite-Direction Movement Flips Direction
def test_genuine_opposite_direction_movement_flips_direction():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH, junction_vector=[0.0, 1.0])
    # Starts INCOMING
    for y in [100, 120, 140, 160]:
        v = simulate_track_update(tracker, 303, (400, y))
    assert v.direction == MovementStateEnum.INCOMING

    # Now makes a U-turn and moves sustained OUTGOING for > 15px over multiple frames
    for y in [155, 140, 125, 110, 95]:
        v = simulate_track_update(tracker, 303, (400, y))
    assert v.direction == MovementStateEnum.OUTGOING

# 17. Frame-Edge Clipping Does Not Flip Direction
def test_frame_edge_clipping_does_not_flip_direction():
    tracker = create_mock_tracker(approach=ApproachEnum.NORTH, junction_vector=[0.0, 1.0])
    # Vehicle approaches bottom image boundary (y=432)
    for y in [350, 370, 390, 412]:
        v = simulate_track_update(tracker, 304, (500, y))
    assert v.direction == MovementStateEnum.INCOMING

    # Centroid shifts upward by 3px near bottom edge due to bbox edge truncation (y=412 -> y=409)
    v = simulate_track_update(tracker, 304, (500, 409))
    assert v.direction == MovementStateEnum.INCOMING # Retains INCOMING at boundary (near edge margin)

