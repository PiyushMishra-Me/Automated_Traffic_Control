"""
emergency package
Phase 1, Phase 2, Phase 3, and Phase 4A Emergency Decision Subsystem:
- Emergency models (EmergencyNotice, EmergencyState, EmergencyVehicleType)
- Live ETA manager (EmergencyETAManager, tick_eta, apply_eta_correction)
- Clearance & Urgency calculations (calculate_t_clear, calculate_effective_emergency_g_max, check_emergency_trigger_conditions, is_queue_cleared)
- Multi-Emergency Clustering & Priority (form_eta_clusters, get_clustered_approaches, compute_emergency_priority_score, resolve_emergency_conflict, sort_emergencies_by_eta)
- Episode container & Multi-Emergency Controller (EmergencyEpisode, EmergencyController)
- Camera & Tracking Event Contracts (EmergencyDetectionEvent, EmergencyEtaUpdateEvent, EmergencyPassageEvent, DirectionalHandoffEvent)
- Camera Integration Adapter (CameraIntegrationAdapter)
"""

from backend.decisionbackend.emergency.emergency_models import (
    EmergencyVehicleType,
    EmergencyState,
    EmergencyNotice,
)
from backend.decisionbackend.emergency.emergency_eta import (
    EmergencyETAManager,
    tick_eta,
    apply_eta_correction,
)
from backend.decisionbackend.emergency.emergency_clearance import (
    DEFAULT_EMPTY_QUEUE_THRESHOLD_PCU,
    CLEARANCE_BASE_SECONDS,
    CLEARANCE_PER_VEHICLE_SECONDS,
    EMERGENCY_G_MAX_MARGIN_SECONDS,
    calculate_t_clear,
    calculate_effective_emergency_g_max,
    check_emergency_trigger_conditions,
    is_queue_cleared,
    EmergencyClearanceCalculator,
)
from backend.decisionbackend.emergency.emergency_clustering import (
    form_eta_clusters,
    get_clustered_approaches,
    compute_emergency_priority_score,
    resolve_emergency_conflict,
    sort_emergencies_by_eta,
    CLUSTER_ETA_THRESHOLD_SECONDS,
    BOOSTED_QUEUE_WEIGHT,
    NORMAL_QUEUE_WEIGHT,
    DIRECTION_TIE_BREAK_ORDER,
)
from backend.decisionbackend.emergency.emergency_controller import (
    EmergencyEpisode,
    EmergencyController,
)
from backend.decisionbackend.emergency.camera_events import (
    EmergencyDetectionEvent,
    EmergencyEtaUpdateEvent,
    EmergencyPassageEvent,
    DirectionalHandoffEvent,
)
from backend.decisionbackend.emergency.camera_interface import (
    CameraIntegrationAdapter,
)

__all__ = [
    "EmergencyVehicleType",
    "EmergencyState",
    "EmergencyNotice",
    "EmergencyETAManager",
    "tick_eta",
    "apply_eta_correction",
    "DEFAULT_EMPTY_QUEUE_THRESHOLD_PCU",
    "CLEARANCE_BASE_SECONDS",
    "CLEARANCE_PER_VEHICLE_SECONDS",
    "EMERGENCY_G_MAX_MARGIN_SECONDS",
    "calculate_t_clear",
    "calculate_effective_emergency_g_max",
    "check_emergency_trigger_conditions",
    "is_queue_cleared",
    "EmergencyClearanceCalculator",
    "form_eta_clusters",
    "get_clustered_approaches",
    "compute_emergency_priority_score",
    "resolve_emergency_conflict",
    "sort_emergencies_by_eta",
    "CLUSTER_ETA_THRESHOLD_SECONDS",
    "BOOSTED_QUEUE_WEIGHT",
    "NORMAL_QUEUE_WEIGHT",
    "DIRECTION_TIE_BREAK_ORDER",
    "EmergencyEpisode",
    "EmergencyController",
    "EmergencyDetectionEvent",
    "EmergencyEtaUpdateEvent",
    "EmergencyPassageEvent",
    "DirectionalHandoffEvent",
    "CameraIntegrationAdapter",
]
