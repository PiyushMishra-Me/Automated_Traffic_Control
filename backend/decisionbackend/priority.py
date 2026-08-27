"""
pcu.py
Passenger Car Unit (PCU) calculation module using field-measured Indian PCU values.
"""

from typing import Dict, Optional
from backend.decisionbackend.junction_config import PCUConfig


def calculate_queue_pcu(vehicle_counts: Dict[str, int], config: Optional[PCUConfig] = None) -> float:
    """
    Convert vehicle counts by category into Passenger Car Units (PCU).
    Formula: Queue_PCU(d) = sum(vehicle_count[class] * PCU[class])

    Coefficients (Field-measured Indian PCU):
    - TWO_WHEELER: 0.13
    - CAR: 1.00
    - AUTO_RICKSHAW: 0.75
    - BUS: 5.40
    - TRUCK: 3.70

    Example:
    10 two-wheelers, 8 cars, 2 buses
    Queue_PCU = 10 * 0.13 + 8 * 1.00 + 2 * 5.40 = 20.1 PCU
    """
    if config is None:
        config = PCUConfig()

    pcu_map = config.to_dict()
    total_pcu = 0.0

    for v_class, count in vehicle_counts.items():
        if count <= 0:
            continue
        # Normalize key name
        norm_key = v_class.strip().lower().replace(" ", "_").replace("-", "_")
        coeff = pcu_map.get(norm_key, 1.0) # Default to 1.0 if unrecognized
        total_pcu += count * coeff

    return round(total_pcu, 4)
