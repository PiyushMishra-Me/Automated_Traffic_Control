from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ultralytics import YOLO

from backend.config import settings
from backend.models.traffic_schemas import (
    ApproachEnum,
    CameraConfig,
    MovementStateEnum,
)


# =========================================================
# DATA MODEL
# =========================================================

@dataclass
class TrackedVehicle:
    track_id: int
    xyxy: List[float]
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[float, float]

    previous_center: Optional[Tuple[float, float]] = None
    speed_px: float = 0.0
    stationary_frames: int = 0
    crossed_counting_line: bool = False

    raw_class_id: Optional[int] = None
    raw_class_name: Optional[str] = None

    direction: MovementStateEnum = MovementStateEnum.UNKNOWN
    stopped_duration_seconds: float = 0.0
    last_moving_direction: Optional[MovementStateEnum] = None
    is_parked: bool = False


# =========================================================
# ROI SENTINEL
# =========================================================

_DEFAULT_ROI = object()


# =========================================================
# VEHICLE TRACKER
# =========================================================

class VehicleTracker:

    def __init__(
            self,
            model_path: Optional[str] = None,
            roi: Any = _DEFAULT_ROI,
            approach: Optional[ApproachEnum] = None,
            junction_vector: Optional[List[float]] = None,
            fps: float = 25.0,
            camera_config: Optional[CameraConfig] = None,
    ):

        # =================================================
        # NORMAL VEHICLE MODEL
        # =================================================

        self.model_path = (
                model_path
                or settings.MODEL_PATH
        )

        self.model = YOLO(
            self.model_path
        )

        self.target_classes = (
            settings.TARGET_CLASSES
        )

        self.class_names = (
            settings.CLASS_NAMES
        )

        # =================================================
        # AMBULANCE MODEL
        # =================================================

        self.ambulance_model = YOLO(
            settings.AMBULANCE_MODEL_PATH
        )

        self.ambulance_class_id = int(
            settings.AMBULANCE_CLASS_ID
        )

        self.ambulance_class_name = str(
            settings.AMBULANCE_CLASS_NAME
        )

        # =================================================
        # DETECTION THRESHOLDS
        # =================================================

        self.conf_threshold = float(
            settings.CONFIDENCE_THRESHOLD
        )

        self.ambulance_conf_threshold = float(
            settings.AMBULANCE_CONFIDENCE_THRESHOLD
        )

        self.iou_threshold = float(
            settings.IOU_THRESHOLD
        )

        # =================================================
        # AMBULANCE CONFIRMATION
        # =================================================

        self.ambulance_min_confirmations = 2

        # A strong ambulance-model detection is trusted immediately.
        # Your real ambulance reaches about 0.60-0.66 confidence,
        # while the tested no-ambulance video stayed below about 0.36.
        self.ambulance_strong_confidence = 0.40

        # How long a confirmed ambulance can survive
        # temporary detector misses.
        self.ambulance_memory_frames = 45

        # How long an ambulance candidate remains matchable.
        self.ambulance_max_missing_frames = 45

        # =================================================
        # AMBULANCE TRACK MATCHING
        # =================================================

        self.ambulance_match_iou = 0.02
        self.ambulance_max_center_distance = 300.0

        # =================================================
        # AMBULANCE CONFIDENCE HISTORY
        # =================================================

        self.ambulance_conf_history_size = 20

        # =================================================
        # NORMAL VEHICLE <-> AMBULANCE ASSOCIATION
        # =================================================

        self.ambulance_overlap_iou_threshold = 0.02
        self.ambulance_overlap_center_distance = 140.0

        # Expand ambulance box to compensate for
        # differences between the two models' boxes.
        self.ambulance_box_expansion_x = 0.45
        self.ambulance_box_expansion_y = 0.45

        # =================================================
        # CAMERA CONFIGURATION
        # =================================================

        self.camera_config = camera_config

        if camera_config is not None:

            self.approach = (
                camera_config.approach
            )

            self.fps = (
                    camera_config.fps
                    or fps
                    or 25.0
            )

            self.roi = (
                camera_config.roi
            )

            j_vec = (
                    camera_config.junction_vector
                    or [0.0, 1.0]
            )

        else:

            self.approach = (
                    approach
                    or ApproachEnum.NORTH
            )

            self.fps = (
                    fps
                    or 25.0
            )

            approach_key = (
                self.approach.value
                if hasattr(
                    self.approach,
                    "value",
                )
                else str(
                    self.approach,
                )
            )

            j_vec = (
                    junction_vector
                    or settings.DEFAULT_JUNCTION_VECTORS.get(
                approach_key,
                [0.0, 1.0],
            )
            )

            if roi is _DEFAULT_ROI:

                self.roi = getattr(
                    settings,
                    "DETECTION_ROI",
                    None,
                )

            else:

                self.roi = roi

        # =================================================
        # NORMALIZE JUNCTION VECTOR
        # =================================================

        self.junction_vector = np.array(
            j_vec,
            dtype=np.float32,
        )

        norm = float(
            np.linalg.norm(
                self.junction_vector
            )
        )

        if norm > 0.0:

            self.junction_vector = (
                    self.junction_vector
                    / norm
            )

        else:

            self.junction_vector = np.array(
                [0.0, 1.0],
                dtype=np.float32,
            )

        # =================================================
        # NORMAL VEHICLE TRACKING MEMORY
        # =================================================

        self.track_history: Dict[
            int,
            List[Tuple[float, float]]
        ] = {}

        self.stationary_counts: Dict[
            int,
            int
        ] = {}

        self.crossed_ids: set[int] = set()

        self.class_votes: Dict[
            int,
            Dict[int, int]
        ] = {}

        self.last_moving_direction: Dict[
            int,
            MovementStateEnum
        ] = {}

        self.stopped_frames_count: Dict[
            int,
            int
        ] = {}

        self.parked_status: Dict[
            int,
            bool
        ] = {}

        self.stationary_ref_center: Dict[
            int,
            Tuple[float, float]
        ] = {}

        # =================================================
        # AMBULANCE TRACKING MEMORY
        # =================================================

        self.ambulance_tracks: Dict[
            int,
            Dict[str, Any]
        ] = {}

        self.next_ambulance_track_id = 100000

        # =================================================
        # NORMAL TRACK <-> AMBULANCE ASSOCIATION
        # =================================================

        self.ambulance_normal_track_ids: Dict[
            int,
            int
        ] = {}

        self.ambulance_normal_track_last_seen: Dict[
            int,
            int
        ] = {}

        # =================================================
        # TEMPORARY NORMAL IDS
        # =================================================

        self.next_temp_track_id = 900000

        # =================================================
        # FRAME NUMBER
        # =================================================

        self._frame_number = 0


    # =====================================================
    # CONFIGURATION
    # =====================================================

    def set_roi(
            self,
            roi: Optional[
                Union[List[float], List[int]]
            ],
    ) -> None:

        self.roi = roi


    def set_camera_config(
            self,
            camera_config: CameraConfig,
    ) -> None:

        self.camera_config = camera_config

        self.approach = (
            camera_config.approach
        )

        if camera_config.fps:
            self.fps = camera_config.fps

        self.roi = camera_config.roi

        j_vec = (
                camera_config.junction_vector
                or [0.0, 1.0]
        )

        self.junction_vector = np.array(
            j_vec,
            dtype=np.float32,
        )

        norm = float(
            np.linalg.norm(
                self.junction_vector
            )
        )

        if norm > 0.0:

            self.junction_vector = (
                    self.junction_vector
                    / norm
            )


    def set_approach(
            self,
            approach: ApproachEnum,
            junction_vector: Optional[
                List[float]
            ] = None,
            fps: Optional[float] = None,
            roi: Optional[
                Union[List[float], List[int]]
            ] = None,
    ) -> None:

        self.approach = approach

        approach_key = (
            approach.value
            if hasattr(
                approach,
                "value",
            )
            else str(
                approach
            )
        )

        j_vec = (
                junction_vector
                or settings.DEFAULT_JUNCTION_VECTORS.get(
            approach_key,
            [0.0, 1.0],
        )
        )

        self.junction_vector = np.array(
            j_vec,
            dtype=np.float32,
        )

        norm = float(
            np.linalg.norm(
                self.junction_vector
            )
        )

        if norm > 0.0:

            self.junction_vector = (
                    self.junction_vector
                    / norm
            )

        if fps is not None:
            self.fps = fps

        if roi is not None:
            self.roi = roi


    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:

        self.track_history.clear()
        self.stationary_counts.clear()
        self.crossed_ids.clear()
        self.class_votes.clear()

        self.last_moving_direction.clear()
        self.stopped_frames_count.clear()
        self.parked_status.clear()
        self.stationary_ref_center.clear()

        self.ambulance_tracks.clear()

        self.ambulance_normal_track_ids.clear()
        self.ambulance_normal_track_last_seen.clear()

        self.next_ambulance_track_id = 100000
        self.next_temp_track_id = 900000

        self._frame_number = 0


    # =====================================================
    # CLASS NAME
    # =====================================================

    def _get_class_name(
            self,
            class_id: int,
    ) -> str:

        if isinstance(
                self.class_names,
                dict,
        ):

            return str(
                self.class_names.get(
                    class_id,
                    str(class_id),
                )
            )

        if isinstance(
                self.class_names,
                list,
        ):

            if (
                    0 <= class_id
                    < len(self.class_names)
            ):

                return str(
                    self.class_names[
                        class_id
                    ]
                )

        return str(class_id)


    # =====================================================
    # ROI
    # =====================================================

    def _get_valid_roi(
            self,
            frame: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        int,
        int,
    ]:

        h, w = frame.shape[:2]

        if (
                self.roi is None
                or not isinstance(
            self.roi,
            (list, tuple),
        )
                or len(self.roi) != 4
        ):

            return frame, 0, 0

        try:

            rx1 = float(self.roi[0])
            ry1 = float(self.roi[1])
            rx2 = float(self.roi[2])
            ry2 = float(self.roi[3])

        except (
                TypeError,
                ValueError,
        ):

            return frame, 0, 0

        normalized = (
                0.0 <= rx1 <= 1.0
                and 0.0 <= ry1 <= 1.0
                and 0.0 <= rx2 <= 1.0
                and 0.0 <= ry2 <= 1.0
        )

        if normalized:

            x1 = int(rx1 * w)
            y1 = int(ry1 * h)
            x2 = int(rx2 * w)
            y2 = int(ry2 * h)

        else:

            x1 = int(rx1)
            y1 = int(ry1)
            x2 = int(rx2)
            y2 = int(ry2)

            # A pixel ROI from another resolution is ignored.
            if (
                    x1 < 0
                    or y1 < 0
                    or x2 > w
                    or y2 > h
            ):

                return frame, 0, 0

        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if (
                x2 <= x1
                or y2 <= y1
        ):

            return frame, 0, 0

        roi_frame = frame[
            y1:y2,
            x1:x2
        ]

        if roi_frame.size == 0:

            return frame, 0, 0

        return (
            roi_frame,
            x1,
            y1,
        )


    # =====================================================
    # IOU
    # =====================================================

    @staticmethod
    def _box_iou(
            box1: List[float],
            box2: List[float],
    ) -> float:

        x1 = max(
            box1[0],
            box2[0],
        )

        y1 = max(
            box1[1],
            box2[1],
        )

        x2 = min(
            box1[2],
            box2[2],
        )

        y2 = min(
            box1[3],
            box2[3],
        )

        inter_width = max(
            0.0,
            x2 - x1,
            )

        inter_height = max(
            0.0,
            y2 - y1,
            )

        intersection = (
                inter_width
                * inter_height
        )

        area1 = (
                max(
                    0.0,
                    box1[2] - box1[0],
                    )
                *
                max(
                    0.0,
                    box1[3] - box1[1],
                    )
        )

        area2 = (
                max(
                    0.0,
                    box2[2] - box2[0],
                    )
                *
                max(
                    0.0,
                    box2[3] - box2[1],
                    )
        )

        union = (
                area1
                + area2
                - intersection
        )

        if union <= 0.0:
            return 0.0

        return intersection / union


    # =====================================================
    # CENTER DISTANCE
    # =====================================================

    @staticmethod
    def _center_distance(
            box1: List[float],
            box2: List[float],
    ) -> float:

        cx1 = (
                      box1[0]
                      + box1[2]
              ) / 2.0

        cy1 = (
                      box1[1]
                      + box1[3]
              ) / 2.0

        cx2 = (
                      box2[0]
                      + box2[2]
              ) / 2.0

        cy2 = (
                      box2[1]
                      + box2[3]
              ) / 2.0

        return float(
            np.hypot(
                cx1 - cx2,
                cy1 - cy2,
                )
        )


    # =====================================================
    # CENTER IN EXPANDED AMBULANCE BOX
    # =====================================================

    def _center_inside_expanded_ambulance_box(
            self,
            vehicle_box: List[float],
            ambulance_box: List[float],
    ) -> bool:

        vehicle_cx = (
                             vehicle_box[0]
                             + vehicle_box[2]
                     ) / 2.0

        vehicle_cy = (
                             vehicle_box[1]
                             + vehicle_box[3]
                     ) / 2.0

        ambulance_width = max(
            1.0,
            ambulance_box[2]
            - ambulance_box[0],
            )

        ambulance_height = max(
            1.0,
            ambulance_box[3]
            - ambulance_box[1],
            )

        expand_x = (
                ambulance_width
                * self.ambulance_box_expansion_x
        )

        expand_y = (
                ambulance_height
                * self.ambulance_box_expansion_y
        )

        x1 = (
                ambulance_box[0]
                - expand_x
        )

        y1 = (
                ambulance_box[1]
                - expand_y
        )

        x2 = (
                ambulance_box[2]
                + expand_x
        )

        y2 = (
                ambulance_box[3]
                + expand_y
        )

        return (
                x1 <= vehicle_cx <= x2
                and
                y1 <= vehicle_cy <= y2
        )


    # =====================================================
    # SAME PHYSICAL VEHICLE?
    # =====================================================

    def _is_same_vehicle_as_ambulance(
            self,
            vehicle_box: List[float],
            ambulance_box: List[float],
    ) -> bool:

        # IoU.
        if (
                self._box_iou(
                    vehicle_box,
                    ambulance_box,
                )
                >=
                self.ambulance_overlap_iou_threshold
        ):

            return True

        # Center inside expanded ambulance box.
        if self._center_inside_expanded_ambulance_box(
                vehicle_box,
                ambulance_box,
        ):

            return True

        # Center distance.
        if (
                self._center_distance(
                    vehicle_box,
                    ambulance_box,
                )
                <=
                self.ambulance_overlap_center_distance
        ):

            return True

        return False


    # =====================================================
    # THIS METHOD WAS MISSING IN YOUR CURRENT FILE
    # =====================================================

    def _overlaps_ambulance_detection(
            self,
            vehicle_box: List[float],
            ambulance_boxes: List[List[float]],
    ) -> bool:

        if not ambulance_boxes:
            return False

        for ambulance_box in ambulance_boxes:

            if self._is_same_vehicle_as_ambulance(
                    vehicle_box,
                    ambulance_box,
            ):

                return True

        return False


    # =====================================================
    # MATCH AMBULANCE TRACK
    # =====================================================

    def _match_ambulance_track(
            self,
            xyxy: List[float],
            current_frame: int,
    ) -> int:

        best_track_id: Optional[int] = None
        best_score = -1.0

        current_center = (
            (
                    xyxy[0]
                    + xyxy[2]
            ) / 2.0,

            (
                    xyxy[1]
                    + xyxy[3]
            ) / 2.0,
        )

        for (
                track_id,
                candidate,
        ) in self.ambulance_tracks.items():

            last_seen = int(
                candidate.get(
                    "last_seen",
                    0,
                )
            )

            if (
                    current_frame
                    - last_seen
                    > self.ambulance_max_missing_frames
            ):

                continue

            previous_box = candidate.get(
                "xyxy"
            )

            previous_center = candidate.get(
                "center"
            )

            if not isinstance(
                    previous_box,
                    list,
            ):

                continue

            if not isinstance(
                    previous_center,
                    tuple,
            ):

                continue

            iou = self._box_iou(
                xyxy,
                previous_box,
            )

            distance = float(
                np.hypot(
                    current_center[0]
                    - previous_center[0],

                    current_center[1]
                    - previous_center[1],
                    )
            )

            if not (
                    iou
                    >= self.ambulance_match_iou
                    or
                    distance
                    <= self.ambulance_max_center_distance
            ):

                continue

            distance_score = max(
                0.0,
                1.0
                - (
                        distance
                        / self.ambulance_max_center_distance
                ),
                )

            score = (
                    0.60 * iou
                    + 0.40 * distance_score
            )

            if score > best_score:

                best_score = score
                best_track_id = track_id

        if best_track_id is not None:

            return best_track_id

        new_id = (
            self.next_ambulance_track_id
        )

        self.next_ambulance_track_id += 1

        return new_id


    # =====================================================
    # UPDATE AMBULANCE CANDIDATE
    # =====================================================

    def _update_ambulance_candidate(
            self,
            track_id: int,
            xyxy: List[float],
            confidence: float,
            current_frame: int,
    ) -> Dict[str, Any]:

        candidate = (
            self.ambulance_tracks.get(
                track_id
            )
        )

        if candidate is None:

            candidate = {
                "xyxy": list(xyxy),

                "center": (
                    (
                            xyxy[0]
                            + xyxy[2]
                    ) / 2.0,

                    (
                            xyxy[1]
                            + xyxy[3]
                    ) / 2.0,
                ),

                "first_seen":
                    current_frame,

                "last_seen":
                    current_frame,

                "hits":
                    0,

                "confidence_history":
                    [],

                "confirmed":
                    False,

                "max_confidence":
                    0.0,

                "avg_confidence":
                    0.0,

                "associated_normal_track_id":
                    None,
            }

            self.ambulance_tracks[
                track_id
            ] = candidate

        previous_last_seen = int(
            candidate.get(
                "last_seen",
                current_frame,
            )
        )

        frame_gap = (
                current_frame
                - previous_last_seen
        )

        if (
                frame_gap
                <= self.ambulance_max_missing_frames
        ):

            candidate["hits"] = (
                    int(
                        candidate.get(
                            "hits",
                            0,
                        )
                    )
                    + 1
            )

        else:

            candidate["hits"] = 1

            candidate[
                "confidence_history"
            ] = []

            candidate[
                "confirmed"
            ] = False

            candidate[
                "associated_normal_track_id"
            ] = None

            candidate[
                "first_seen"
            ] = current_frame

        confidence_history = (
            candidate.get(
                "confidence_history",
                [],
            )
        )

        if not isinstance(
                confidence_history,
                list,
        ):

            confidence_history = []

        confidence_history.append(
            float(confidence)
        )

        if (
                len(confidence_history)
                > self.ambulance_conf_history_size
        ):

            confidence_history = (
                confidence_history[
                    -self.ambulance_conf_history_size:
                ]
            )

        candidate[
            "confidence_history"
        ] = confidence_history

        candidate["xyxy"] = list(
            xyxy
        )

        candidate["center"] = (
            (
                    xyxy[0]
                    + xyxy[2]
            ) / 2.0,

            (
                    xyxy[1]
                    + xyxy[3]
            ) / 2.0,
        )

        candidate[
            "last_seen"
        ] = current_frame

        hits = int(
            candidate.get(
                "hits",
                0,
            )
        )

        max_confidence = (
            float(
                max(
                    confidence_history
                )
            )
            if confidence_history
            else 0.0
        )

        avg_confidence = (
            float(
                np.mean(
                    confidence_history
                )
            )
            if confidence_history
            else 0.0
        )

        candidate[
            "max_confidence"
        ] = max_confidence

        candidate[
            "avg_confidence"
        ] = avg_confidence

        # -----------------------------------------------
        # CONFIRM
        # -----------------------------------------------

        # Confirm immediately when the current detection is strong.
        # Otherwise require two sufficiently confident observations.
        strong_single_frame = (
                float(confidence)
                >= self.ambulance_strong_confidence
        )

        persistent_candidate = (
                hits
                >= self.ambulance_min_confirmations
                and
                max_confidence
                >= self.ambulance_conf_threshold
        )

        if strong_single_frame or persistent_candidate:

            candidate[
                "confirmed"
            ] = True

        return candidate


    # =====================================================
    # CLEANUP AMBULANCE TRACKS
    # =====================================================

    def _cleanup_ambulance_tracks(
            self,
            current_frame: int,
    ) -> None:

        expired_ids: List[int] = []

        for (
                track_id,
                candidate,
        ) in self.ambulance_tracks.items():

            last_seen = int(
                candidate.get(
                    "last_seen",
                    0,
                )
            )

            if (
                    current_frame
                    - last_seen
                    > self.ambulance_max_missing_frames
            ):

                expired_ids.append(
                    track_id
                )

        for track_id in expired_ids:

            self.ambulance_tracks.pop(
                track_id,
                None
            )

            self.ambulance_normal_track_ids.pop(
                track_id,
                None,
            )

            self.ambulance_normal_track_last_seen.pop(
                track_id,
                None,
            )


    # =====================================================
    # RECENT CONFIRMED AMBULANCE MATCH
    # =====================================================

    def _matches_recent_confirmed_ambulance(
            self,
            vehicle_box: List[float],
            current_frame: int,
    ) -> Optional[int]:

        best_track_id = None
        best_score = -1.0

        for (
                ambulance_track_id,
                candidate,
        ) in self.ambulance_tracks.items():

            if not candidate.get(
                    "confirmed",
                    False,
            ):

                continue

            last_seen = int(
                candidate.get(
                    "last_seen",
                    0,
                )
            )

            if (
                    current_frame
                    - last_seen
                    > self.ambulance_memory_frames
            ):

                continue

            ambulance_box = (
                candidate.get(
                    "xyxy"
                )
            )

            if not isinstance(
                    ambulance_box,
                    list,
            ):

                continue

            if not self._is_same_vehicle_as_ambulance(
                    vehicle_box,
                    ambulance_box,
            ):

                continue

            iou = self._box_iou(
                vehicle_box,
                ambulance_box,
            )

            distance = self._center_distance(
                vehicle_box,
                ambulance_box,
            )

            score = (
                    iou
                    + max(
                0.0,
                1.0
                - (
                        distance
                        / max(
                    1.0,
                    self.ambulance_overlap_center_distance,
                )
                ),
                )
            )

            if score > best_score:

                best_score = score
                best_track_id = (
                    ambulance_track_id
                )

        return best_track_id


    # =====================================================
    # MOTION
    # =====================================================

    def _update_motion(
            self,
            track_id: int,
            curr_center: Tuple[float, float],
            current_fps: float,
    ) -> Tuple[
        Optional[Tuple[float, float]],
        float,
        MovementStateEnum,
        bool,
        float,
    ]:

        previous_center = None
        speed_px = 0.0

        if (
                track_id
                in self.track_history
                and self.track_history[
            track_id
        ]
        ):

            previous_center = (
                self.track_history[
                    track_id
                ][-1]
            )

            speed_px = float(
                np.hypot(
                    curr_center[0]
                    - previous_center[0],
                    curr_center[1]
                    - previous_center[1],
                    )
            )

        if (
                track_id
                not in self.track_history
        ):

            self.track_history[
                track_id
            ] = []

        self.track_history[
            track_id
        ].append(
            curr_center
        )

        if len(
                self.track_history[
                    track_id
                ]
        ) > 30:

            self.track_history[
                track_id
            ].pop(0)

        history = (
            self.track_history[
                track_id
            ]
        )

        # =================================================
        # QUEUE / STATIONARY
        # =================================================

        if (
                speed_px
                < settings.QUEUE_SPEED_THRESHOLD
        ):

            self.stationary_counts[
                track_id
            ] = (
                    self.stationary_counts.get(
                        track_id,
                        0,
                    )
                    + 1
            )

        else:

            self.stationary_counts[
                track_id
            ] = max(
                0,
                self.stationary_counts.get(
                    track_id,
                    0,
                )
                - 1,
                )

        is_stationary = (
                speed_px
                < settings.MOVEMENT_SPEED_THRESHOLD
        )

        if is_stationary:

            if (
                    track_id
                    not in self.stationary_ref_center
            ):

                self.stationary_ref_center[
                    track_id
                ] = curr_center

        else:

            self.stationary_ref_center.pop(
                track_id,
                None,
            )

        # =================================================
        # STOPPED / PARKED
        # =================================================

        if is_stationary:

            self.stopped_frames_count[
                track_id
            ] = (
                    self.stopped_frames_count.get(
                        track_id,
                        0,
                    )
                    + 1
            )

            stopped_duration = (
                    self.stopped_frames_count[
                        track_id
                    ]
                    / current_fps
            )

            if (
                    stopped_duration
                    > settings.PARKED_DURATION_SECONDS
            ):

                self.parked_status[
                    track_id
                ] = True

                movement_state = (
                    MovementStateEnum.PARKED
                )

            else:

                last_direction = (
                    self.last_moving_direction.get(
                        track_id
                    )
                )

                if (
                        last_direction
                        == MovementStateEnum.INCOMING
                ):

                    movement_state = (
                        MovementStateEnum.STOPPED_INCOMING
                    )

                elif (
                        last_direction
                        == MovementStateEnum.OUTGOING
                ):

                    movement_state = (
                        MovementStateEnum.STOPPED_OUTGOING
                    )

                else:

                    movement_state = (
                        MovementStateEnum.UNKNOWN
                    )

        # =================================================
        # MOVING
        # =================================================

        else:

            self.stopped_frames_count[
                track_id
            ] = 0

            self.parked_status[
                track_id
            ] = False

            if (
                    len(history)
                    < settings.MIN_TRAJECTORY_POINTS
            ):

                movement_state = (
                    self.last_moving_direction.get(
                        track_id,
                        MovementStateEnum.UNKNOWN,
                    )
                )

            else:

                k = min(
                    len(history),
                    8,
                )

                dx = (
                        history[-1][0]
                        - history[-k][0]
                )

                dy = (
                        history[-1][1]
                        - history[-k][1]
                )

                magnitude = float(
                    np.hypot(
                        dx,
                        dy,
                    )
                )

                if magnitude < 2.0:

                    movement_state = (
                        self.last_moving_direction.get(
                            track_id,
                            MovementStateEnum.UNKNOWN,
                        )
                    )

                else:

                    unit_vector = np.array(
                        [
                            dx / magnitude,
                            dy / magnitude,
                            ],
                        dtype=np.float32,
                    )

                    dot_product = float(
                        np.dot(
                            unit_vector,
                            self.junction_vector,
                        )
                    )

                    if dot_product > 0.15:

                        movement_state = (
                            MovementStateEnum.INCOMING
                        )

                        self.last_moving_direction[
                            track_id
                        ] = (
                            MovementStateEnum.INCOMING
                        )

                    elif dot_product < -0.15:

                        movement_state = (
                            MovementStateEnum.OUTGOING
                        )

                        self.last_moving_direction[
                            track_id
                        ] = (
                            MovementStateEnum.OUTGOING
                        )

                    else:

                        movement_state = (
                            self.last_moving_direction.get(
                                track_id,
                                MovementStateEnum.UNKNOWN,
                            )
                        )

        is_parked = (
                movement_state
                == MovementStateEnum.PARKED
        )

        stopped_duration_seconds = (
                self.stopped_frames_count.get(
                    track_id,
                    0,
                )
                / current_fps
        )

        return (
            previous_center,
            speed_px,
            movement_state,
            is_parked,
            stopped_duration_seconds,
        )


    # =====================================================
    # MAIN TRACK
    # =====================================================

    def track(
            self,
            frame: np.ndarray,
            fps: Optional[float] = None,
    ) -> List[TrackedVehicle]:

        current_fps = (
                fps
                or self.fps
                or 25.0
        )

        self._frame_number += 1

        current_frame = (
            self._frame_number
        )

        # =================================================
        # ROI
        # =================================================

        (
            source_img,
            x_offset,
            y_offset,
        ) = self._get_valid_roi(
            frame
        )

        tracked_vehicles: List[
            TrackedVehicle
        ] = []

        # =================================================
        # 1. AMBULANCE DETECTION
        # =================================================

        ambulance_results = (
            self.ambulance_model.predict(
                source=source_img,
                conf=self.ambulance_conf_threshold,
                iou=self.iou_threshold,
                imgsz=settings.INFERENCE_IMAGE_SIZE,
                verbose=False,
            )
        )

        current_ambulance_boxes: List[
            List[float]
        ] = []

        current_ambulance_observations: List[
            Tuple[
                int,
                List[float],
                float,
                Dict[str, Any],
            ]
        ] = []

        if (
                ambulance_results is not None
                and len(ambulance_results) > 0
        ):

            ambulance_result = (
                ambulance_results[0]
            )

            if (
                    ambulance_result.boxes is not None
                    and len(
                ambulance_result.boxes
            ) > 0
            ):

                for box in (
                        ambulance_result.boxes
                ):

                    confidence = float(
                        box.conf[0].item()
                    )

                    raw_box = [
                        float(value)
                        for value
                        in box.xyxy[0].tolist()
                    ]

                    full_box = [
                        raw_box[0] + x_offset,
                        raw_box[1] + y_offset,
                        raw_box[2] + x_offset,
                        raw_box[3] + y_offset,
                        ]

                    current_ambulance_boxes.append(
                        full_box
                    )

                    track_id = (
                        self._match_ambulance_track(
                            full_box,
                            current_frame,
                        )
                    )

                    candidate = (
                        self._update_ambulance_candidate(
                            track_id=track_id,
                            xyxy=full_box,
                            confidence=confidence,
                            current_frame=current_frame,
                        )
                    )

                    current_ambulance_observations.append(
                        (
                            track_id,
                            full_box,
                            confidence,
                            candidate,
                        )
                    )

        # =================================================
        # 2. NORMAL VEHICLE DETECTION
        # =================================================

        results = self.model.track(
            source=source_img,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=settings.INFERENCE_IMAGE_SIZE,
            classes=self.target_classes,
            verbose=False,
        )

        normal_detections: List[
            Tuple[
                List[float],
                int,
                float,
                int,
            ]
        ] = []

        if (
                results is not None
                and len(results) > 0
        ):

            result = results[0]

            if (
                    result.boxes is not None
                    and len(result.boxes) > 0
            ):

                boxes = (
                    result.boxes.xyxy
                    .cpu()
                    .numpy()
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                class_ids = (
                    result.boxes.cls
                    .int()
                    .cpu()
                    .numpy()
                )

                if result.boxes.id is not None:

                    ids = (
                        result.boxes.id
                        .int()
                        .cpu()
                        .numpy()
                    )

                else:

                    generated_ids = []

                    for _ in range(
                            len(boxes)
                    ):

                        generated_ids.append(
                            self.next_temp_track_id
                        )

                        self.next_temp_track_id += 1

                    ids = np.array(
                        generated_ids,
                        dtype=np.int32,
                    )

                for (
                        box,
                        track_id_value,
                        confidence,
                        class_id_value,
                ) in zip(
                    boxes,
                    ids,
                    confidences,
                    class_ids,
                ):

                    track_id = int(
                        track_id_value
                    )

                    class_id = int(
                        class_id_value
                    )

                    if (
                            class_id
                            not in self.target_classes
                    ):
                        continue

                    full_box = [
                        float(
                            box[0]
                            + x_offset
                        ),
                        float(
                            box[1]
                            + y_offset
                        ),
                        float(
                            box[2]
                            + x_offset
                        ),
                        float(
                            box[3]
                            + y_offset
                        ),
                    ]

                    normal_detections.append(
                        (
                            full_box,
                            track_id,
                            float(
                                confidence
                            ),
                            class_id,
                        )
                    )

        # =================================================
        # 3. ASSOCIATE CURRENT AMBULANCE WITH NORMAL TRACK
        # =================================================

        for (
                ambulance_track_id,
                ambulance_box,
                _,
                candidate,
        ) in current_ambulance_observations:

            best_normal_track_id = None
            best_score = -1.0

            for (
                    normal_box,
                    normal_track_id,
                    _,
                    _,
            ) in normal_detections:

                if not self._is_same_vehicle_as_ambulance(
                        normal_box,
                        ambulance_box,
                ):

                    continue

                iou = self._box_iou(
                    normal_box,
                    ambulance_box,
                )

                distance = self._center_distance(
                    normal_box,
                    ambulance_box,
                )

                distance_score = max(
                    0.0,
                    1.0
                    - (
                            distance
                            / max(
                        1.0,
                        self.ambulance_overlap_center_distance,
                    )
                    ),
                    )

                score = (
                        iou
                        + distance_score
                )

                if score > best_score:

                    best_score = score
                    best_normal_track_id = (
                        normal_track_id
                    )

            if best_normal_track_id is not None:

                candidate[
                    "associated_normal_track_id"
                ] = (
                    best_normal_track_id
                )

                self.ambulance_normal_track_ids[
                    ambulance_track_id
                ] = (
                    best_normal_track_id
                )

                self.ambulance_normal_track_last_seen[
                    ambulance_track_id
                ] = current_frame

        # =================================================
        # 4. NORMAL VEHICLE OUTPUT
        # =================================================

        for (
                full_box,
                normal_track_id,
                confidence,
                class_id,
        ) in normal_detections:

            matched_ambulance_track = None

            # -------------------------------------------------
            # Existing normal-track association.
            # -------------------------------------------------

            for (
                    ambulance_track_id,
                    associated_normal_track_id,
            ) in self.ambulance_normal_track_ids.items():

                if (
                        associated_normal_track_id
                        != normal_track_id
                ):
                    continue

                last_seen = (
                    self.ambulance_normal_track_last_seen.get(
                        ambulance_track_id,
                        0,
                    )
                )

                if (
                        current_frame
                        - last_seen
                        <= self.ambulance_memory_frames
                ):

                    candidate = (
                        self.ambulance_tracks.get(
                            ambulance_track_id
                        )
                    )

                    if (
                            candidate is not None
                            and candidate.get(
                        "confirmed",
                        False,
                    )
                    ):

                        matched_ambulance_track = (
                            ambulance_track_id
                        )

                        break

            # -------------------------------------------------
            # Recent confirmed ambulance geometry.
            # -------------------------------------------------

            if (
                    matched_ambulance_track
                    is None
            ):

                matched_ambulance_track = (
                    self._matches_recent_confirmed_ambulance(
                        full_box,
                        current_frame,
                    )
                )

            # -------------------------------------------------
            # Suppress normal vehicle.
            # -------------------------------------------------

            if matched_ambulance_track is not None:

                self.ambulance_normal_track_ids[
                    matched_ambulance_track
                ] = normal_track_id

                self.ambulance_normal_track_last_seen[
                    matched_ambulance_track
                ] = current_frame

                continue

            # -------------------------------------------------
            # Normal class voting.
            # -------------------------------------------------

            if (
                    normal_track_id
                    not in self.class_votes
            ):

                self.class_votes[
                    normal_track_id
                ] = {}

            votes = (
                self.class_votes[
                    normal_track_id
                ]
            )

            votes[class_id] = (
                    votes.get(
                        class_id,
                        0,
                    )
                    + 1
            )

            stable_class_id = max(
                votes,
                key=votes.get,
            )

            stable_class_name = (
                self._get_class_name(
                    stable_class_id
                )
            )

            # -------------------------------------------------
            # CENTER
            # -------------------------------------------------

            cx = (
                         full_box[0]
                         + full_box[2]
                 ) / 2.0

            cy = (
                         full_box[1]
                         + full_box[3]
                 ) / 2.0

            curr_center = (
                cx,
                cy,
            )

            # -------------------------------------------------
            # MOTION
            # -------------------------------------------------

            (
                previous_center,
                speed_px,
                movement_state,
                is_parked,
                stopped_duration_seconds,
            ) = self._update_motion(
                track_id=normal_track_id,
                curr_center=curr_center,
                current_fps=current_fps,
            )

            tracked_vehicles.append(
                TrackedVehicle(
                    track_id=normal_track_id,
                    xyxy=full_box,
                    confidence=float(
                        confidence
                    ),
                    class_id=stable_class_id,
                    class_name=stable_class_name,
                    center=curr_center,
                    previous_center=previous_center,
                    speed_px=speed_px,
                    stationary_frames=(
                        self.stationary_counts.get(
                            normal_track_id,
                            0,
                        )
                    ),
                    crossed_counting_line=(
                            normal_track_id
                            in self.crossed_ids
                    ),
                    raw_class_id=class_id,
                    raw_class_name=(
                        self._get_class_name(
                            class_id
                        )
                    ),
                    direction=movement_state,
                    stopped_duration_seconds=(
                        stopped_duration_seconds
                    ),
                    last_moving_direction=(
                        self.last_moving_direction.get(
                            normal_track_id
                        )
                    ),
                    is_parked=is_parked,
                )
            )

        # =================================================
        # 5. OUTPUT CONFIRMED AMBULANCE
        # =================================================

        output_ambulance_ids = set()

        # -------------------------------------------------
        # Current detector observations.
        # -------------------------------------------------

        for (
                track_id,
                full_box,
                confidence,
                candidate,
        ) in current_ambulance_observations:

            if not bool(
                    candidate.get(
                        "confirmed",
                        False,
                    )
            ):

                continue

            output_ambulance_ids.add(
                track_id
            )

            cx = (
                         full_box[0]
                         + full_box[2]
                 ) / 2.0

            cy = (
                         full_box[1]
                         + full_box[3]
                 ) / 2.0

            curr_center = (
                cx,
                cy,
            )

            (
                previous_center,
                speed_px,
                movement_state,
                is_parked,
                stopped_duration_seconds,
            ) = self._update_motion(
                track_id=track_id,
                curr_center=curr_center,
                current_fps=current_fps,
            )

            confidence_for_output = float(
                candidate.get(
                    "max_confidence",
                    confidence,
                )
            )

            tracked_vehicles.append(
                TrackedVehicle(
                    track_id=track_id,
                    xyxy=list(full_box),
                    confidence=confidence_for_output,
                    class_id=(
                        self.ambulance_class_id
                    ),
                    class_name=(
                        self.ambulance_class_name
                    ),
                    center=curr_center,
                    previous_center=(
                        previous_center
                    ),
                    speed_px=speed_px,
                    stationary_frames=(
                        self.stationary_counts.get(
                            track_id,
                            0,
                        )
                    ),
                    crossed_counting_line=(
                            track_id
                            in self.crossed_ids
                    ),
                    raw_class_id=0,
                    raw_class_name=(
                        self.ambulance_class_name
                    ),
                    direction=movement_state,
                    stopped_duration_seconds=(
                        stopped_duration_seconds
                    ),
                    last_moving_direction=(
                        self.last_moving_direction.get(
                            track_id
                        )
                    ),
                    is_parked=is_parked,
                )
            )

        # -------------------------------------------------
        # Persistent ambulance when current detector misses.
        # -------------------------------------------------

        for (
                track_id,
                candidate,
        ) in self.ambulance_tracks.items():

            if track_id in output_ambulance_ids:
                continue

            if not bool(
                    candidate.get(
                        "confirmed",
                        False,
                    )
            ):

                continue

            last_seen = int(
                candidate.get(
                    "last_seen",
                    0,
                )
            )

            if (
                    current_frame
                    - last_seen
                    > self.ambulance_memory_frames
            ):

                continue

            full_box = candidate.get(
                "xyxy"
            )

            if not isinstance(
                    full_box,
                    list,
            ):

                continue

            cx = (
                         full_box[0]
                         + full_box[2]
                 ) / 2.0

            cy = (
                         full_box[1]
                         + full_box[3]
                 ) / 2.0

            curr_center = (
                cx,
                cy,
            )

            (
                previous_center,
                speed_px,
                movement_state,
                is_parked,
                stopped_duration_seconds,
            ) = self._update_motion(
                track_id=track_id,
                curr_center=curr_center,
                current_fps=current_fps,
            )

            tracked_vehicles.append(
                TrackedVehicle(
                    track_id=track_id,
                    xyxy=list(full_box),
                    confidence=float(
                        candidate.get(
                            "max_confidence",
                            0.0,
                        )
                    ),
                    class_id=(
                        self.ambulance_class_id
                    ),
                    class_name=(
                        self.ambulance_class_name
                    ),
                    center=curr_center,
                    previous_center=(
                        previous_center
                    ),
                    speed_px=speed_px,
                    stationary_frames=(
                        self.stationary_counts.get(
                            track_id,
                            0,
                        )
                    ),
                    crossed_counting_line=(
                            track_id
                            in self.crossed_ids
                    ),
                    raw_class_id=0,
                    raw_class_name=(
                        self.ambulance_class_name
                    ),
                    direction=movement_state,
                    stopped_duration_seconds=(
                        stopped_duration_seconds
                    ),
                    last_moving_direction=(
                        self.last_moving_direction.get(
                            track_id
                        )
                    ),
                    is_parked=(
                        is_parked
                    ),
                )
            )

        # =================================================
        # FINAL SAFETY DEDUPLICATION
        # =================================================

        if self.ambulance_tracks:

            active_ambulance_boxes = []

            for candidate in (
                    self.ambulance_tracks.values()
            ):

                if not candidate.get(
                        "confirmed",
                        False,
                ):

                    continue

                last_seen = int(
                    candidate.get(
                        "last_seen",
                        0,
                    )
                )

                if (
                        current_frame
                        - last_seen
                        > self.ambulance_memory_frames
                ):

                    continue

                box = candidate.get(
                    "xyxy"
                )

                if isinstance(
                        box,
                        list,
                ):

                    active_ambulance_boxes.append(
                        box
                    )

            if active_ambulance_boxes:

                filtered = []

                for vehicle in tracked_vehicles:

                    # Always keep ambulance.
                    if (
                            vehicle.class_name
                            == self.ambulance_class_name
                    ):

                        filtered.append(
                            vehicle
                        )

                        continue

                    # Remove normal vehicle if it is the
                    # same physical object as an ambulance.
                    duplicate = (
                        self._overlaps_ambulance_detection(
                            vehicle.xyxy,
                            active_ambulance_boxes,
                        )
                    )

                    if not duplicate:

                        filtered.append(
                            vehicle
                        )

                tracked_vehicles = filtered

        # =================================================
        # CLEANUP
        # =================================================

        self._cleanup_ambulance_tracks(
            current_frame
        )

        return tracked_vehicles
