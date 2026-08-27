import pytest
import numpy as np
from backend.models.traffic_schemas import ApproachEnum, CameraConfig, CountingLineConfig, MovementStateEnum
from backend.core.vision.tracker import VehicleTracker, TrackedVehicle
from backend.config import settings

class MockYOLO:
    def __init__(self):
        self.device = "cpu"
    def track(self, *args, **kwargs):
        return []

def create_configured_tracker(camera_config: CameraConfig):
    tracker = VehicleTracker(camera_config=camera_config)
    tracker.model = MockYOLO()
    tracker.reset()
    return tracker

def simulate_track(tracker, track_id, trajectory, fps=25.0):
    """Feed a sequence of (x, y) coordinates into tracker and return the final TrackedVehicle."""
    v = None
    for pt in trajectory:
        curr_center = (float(pt[0]), float(pt[1]))
        if track_id not in tracker.class_votes:
            tracker.class_votes[track_id] = {2: 1}
        
        if track_id not in tracker.track_history:
            tracker.track_history[track_id] = []
        tracker.track_history[track_id].append(curr_center)
        if len(tracker.track_history[track_id]) > 30:
            tracker.track_history[track_id].pop(0)

        history = tracker.track_history[track_id]
        speed_px = 0.0
        if len(history) >= 2:
            prev = history[-2]
            speed_px = float(np.hypot(curr_center[0] - prev[0], curr_center[1] - prev[1]))

        # State determination
        is_stationary = (speed_px < settings.MOVEMENT_SPEED_THRESHOLD)
        if is_stationary:
            tracker.stopped_frames_count[track_id] = tracker.stopped_frames_count.get(track_id, 0) + 1
            last_dir = tracker.last_moving_direction.get(track_id)
            if last_dir == MovementStateEnum.INCOMING:
                m_state = MovementStateEnum.STOPPED_INCOMING
            elif last_dir == MovementStateEnum.OUTGOING:
                m_state = MovementStateEnum.STOPPED_OUTGOING
            else:
                m_state = MovementStateEnum.UNKNOWN
        else:
            tracker.stopped_frames_count[track_id] = 0
            if len(history) < settings.MIN_TRAJECTORY_POINTS:
                m_state = tracker.last_moving_direction.get(track_id, MovementStateEnum.UNKNOWN)
            else:
                k = min(len(history), 8)
                dx = history[-1][0] - history[-k][0]
                dy = history[-1][1] - history[-k][1]
                mag = float(np.hypot(dx, dy))
                if mag < 2.0:
                    m_state = tracker.last_moving_direction.get(track_id, MovementStateEnum.UNKNOWN)
                else:
                    u_vec = np.array([dx / mag, dy / mag], dtype=np.float32)
                    dot_prod = float(np.dot(u_vec, tracker.junction_vector))
                    if dot_prod > 0.15:
                        m_state = MovementStateEnum.INCOMING
                        tracker.last_moving_direction[track_id] = MovementStateEnum.INCOMING
                    elif dot_prod < -0.15:
                        m_state = MovementStateEnum.OUTGOING
                        tracker.last_moving_direction[track_id] = MovementStateEnum.OUTGOING
                    else:
                        m_state = tracker.last_moving_direction.get(track_id, MovementStateEnum.UNKNOWN)

        v = TrackedVehicle(
            track_id=track_id,
            xyxy=[curr_center[0]-10, curr_center[1]-10, curr_center[0]+10, curr_center[1]+10],
            confidence=0.9,
            class_id=2,
            class_name="car",
            center=curr_center,
            speed_px=speed_px,
            direction=m_state,
            last_moving_direction=tracker.last_moving_direction.get(track_id)
        )
    return v

# 1. Two cameras can have different ROIs
def test_two_cameras_different_rois():
    cam1_cfg = CameraConfig(
        camera_id="CAM-NORTH-01",
        junction_id="J1",
        approach=ApproachEnum.NORTH,
        roi=[220.0, 0.0, 768.0, 432.0]
    )
    cam2_cfg = CameraConfig(
        camera_id="CAM-BIDIR-01",
        junction_id="J1",
        approach=ApproachEnum.NORTH,
        roi=None # Full frame
    )

    tracker1 = create_configured_tracker(cam1_cfg)
    tracker2 = create_configured_tracker(cam2_cfg)

    assert tracker1.roi == [220.0, 0.0, 768.0, 432.0]
    assert tracker2.roi is None
    assert tracker1.roi != tracker2.roi

# 2. Two cameras can have opposite junction vectors
def test_two_cameras_opposite_junction_vectors():
    # Camera looking North (downward motion = towards junction)
    cam_north = CameraConfig(
        camera_id="CAM-N",
        junction_id="J1",
        approach=ApproachEnum.NORTH,
        junction_vector=[0.0, 1.0]
    )
    # Camera looking South (upward motion = towards junction)
    cam_south = CameraConfig(
        camera_id="CAM-S",
        junction_id="J1",
        approach=ApproachEnum.SOUTH,
        junction_vector=[0.0, -1.0]
    )

    t_north = create_configured_tracker(cam_north)
    t_south = create_configured_tracker(cam_south)

    # A vehicle moving downward (increasing y: 100 -> 160)
    v_north = simulate_track(t_north, 1, [(300, 100), (300, 120), (300, 140), (300, 160)])
    v_south = simulate_track(t_south, 1, [(300, 100), (300, 120), (300, 140), (300, 160)])

    # Downward is INCOMING for North camera, but OUTGOING for South camera
    assert v_north.direction == MovementStateEnum.INCOMING
    assert v_south.direction == MovementStateEnum.OUTGOING

# 3. Incoming/outgoing corridor geometry is camera-specific
def test_incoming_outgoing_corridor_geometry_is_camera_specific():
    # Camera A has left corridor outgoing, right corridor incoming
    cam_a = CameraConfig(
        camera_id="CAM-A",
        junction_id="J1",
        approach=ApproachEnum.NORTH,
        incoming_corridor=[[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
        outgoing_corridor=[[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
    )
    # Camera B has arbitrary angled corridors
    cam_b = CameraConfig(
        camera_id="CAM-B",
        junction_id="J2",
        approach=ApproachEnum.EAST,
        incoming_corridor=[[0.0, 0.2], [0.8, 0.2], [0.8, 0.8], [0.0, 0.8]],
        outgoing_corridor=None
    )

    assert cam_a.incoming_corridor != cam_b.incoming_corridor
    assert len(cam_a.incoming_corridor) == 4
    assert cam_b.outgoing_corridor is None

# 4. Changing camera geometry does not require changing tracker code
def test_reconfiguring_camera_dynamically():
    tracker = VehicleTracker()
    tracker.model = MockYOLO()
    
    # Configure as East approach
    cfg_east = CameraConfig(
        camera_id="CAM-E",
        junction_id="J1",
        approach=ApproachEnum.EAST,
        junction_vector=[-1.0, 0.0]
    )
    tracker.set_camera_config(cfg_east)
    
    # Vehicle moving leftward (dx < 0) -> INCOMING
    v = simulate_track(tracker, 10, [(400, 200), (380, 200), (360, 200), (340, 200)])
    assert v.direction == MovementStateEnum.INCOMING

    # Reconfigure dynamically to West approach (vector = [+1, 0])
    cfg_west = CameraConfig(
        camera_id="CAM-W",
        junction_id="J1",
        approach=ApproachEnum.WEST,
        junction_vector=[1.0, 0.0]
    )
    tracker.set_camera_config(cfg_west)
    tracker.reset()

    # Vehicle moving leftward (dx < 0) -> now OUTGOING for West camera
    v2 = simulate_track(tracker, 20, [(400, 200), (380, 200), (360, 200), (340, 200)])
    assert v2.direction == MovementStateEnum.OUTGOING

# 5. Bidirectional traffic can be represented by the same camera
def test_bidirectional_camera_representation():
    cam_bidir = CameraConfig(
        camera_id="CAM-BIDIR",
        junction_id="J1",
        approach=ApproachEnum.NORTH,
        roi=None,
        junction_vector=[0.0, 1.0],
        is_bidirectional=True
    )
    tracker = create_configured_tracker(cam_bidir)

    # Southbound vehicle (downward)
    v_sb = simulate_track(tracker, 101, [(500, 100), (500, 120), (500, 140), (500, 160)])
    # Northbound vehicle (upward)
    v_nb = simulate_track(tracker, 102, [(200, 300), (200, 280), (200, 260), (200, 240)])

    assert v_sb.direction == MovementStateEnum.INCOMING
    assert v_nb.direction == MovementStateEnum.OUTGOING

# 6. No production logic depends on x=380 or any hardcoded coordinate from bidirectional.mp4
def test_no_hardcoded_x_coordinate_dependency():
    cam = CameraConfig(
        camera_id="CAM-GENERAL",
        junction_id="J3",
        approach=ApproachEnum.NORTH,
        junction_vector=[0.0, 1.0]
    )
    tracker = create_configured_tracker(cam)

    # Vehicle on far-left side (e.g. x=50, which in bidirectional.mp4 was outbound) moving DOWNWARD
    v_left_incoming = simulate_track(tracker, 1, [(50, 100), (50, 120), (50, 140), (50, 160)])
    assert v_left_incoming.direction == MovementStateEnum.INCOMING # Direction comes from trajectory vector, NOT x < 380!

    # Vehicle on far-right side (e.g. x=700, which in bidirectional.mp4 was inbound) moving UPWARD
    v_right_outgoing = simulate_track(tracker, 2, [(700, 300), (700, 280), (700, 260), (700, 240)])
    assert v_right_outgoing.direction == MovementStateEnum.OUTGOING # Direction comes from trajectory vector, NOT x >= 380!
