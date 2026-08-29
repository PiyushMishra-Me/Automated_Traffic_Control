import heapq
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timezone
from backend.models.traffic_schemas import ApproachEnum
from backend.models.navigation_schemas import (
    VehicleProfileEnum,
    NavigationRequest,
    NavigationResponse,
    NavigationStep,
    CorridorCostDetail
)
from backend.core.control.ambulance_engine import CORRIDOR_MAP
from backend.db.repositories.junction_repo import junction_repo
from backend.db.repositories.incident_repo import incident_repo
from backend.db.repositories.ambulance_repo import ambulance_repo
from backend.core.weather.weather_service import weather_service

# Physical segment base distances in kilometers and base free-flow travel times in seconds
ROAD_SEGMENTS: Dict[Tuple[str, str], Dict] = {
    # -------------------------------------------------------------
    # NEW DELHI / NCR SEGMENTS
    # -------------------------------------------------------------
    ("J-04", "J-01"): {"dist_km": 3.8, "base_sec": 240, "name": "South Radial Expressway", "approach": ApproachEnum.SOUTH},
    ("J-01", "J-04"): {"dist_km": 3.8, "base_sec": 240, "name": "South Radial Expressway", "approach": ApproachEnum.NORTH},
    ("J-01", "J-02"): {"dist_km": 2.6, "base_sec": 160, "name": "East Arterial Corridor B", "approach": ApproachEnum.WEST},
    ("J-02", "J-01"): {"dist_km": 2.6, "base_sec": 160, "name": "East Arterial Corridor B", "approach": ApproachEnum.EAST},
    ("J-01", "J-03"): {"dist_km": 3.1, "base_sec": 200, "name": "North Boulevard", "approach": ApproachEnum.SOUTH},
    ("J-03", "J-01"): {"dist_km": 3.1, "base_sec": 200, "name": "North Boulevard", "approach": ApproachEnum.NORTH},
    ("J-01", "J-05"): {"dist_km": 2.9, "base_sec": 180, "name": "West Commercial Linkway", "approach": ApproachEnum.EAST},
    ("J-05", "J-01"): {"dist_km": 2.9, "base_sec": 180, "name": "West Commercial Linkway", "approach": ApproachEnum.WEST},
    ("J-03", "J-02"): {"dist_km": 4.2, "base_sec": 260, "name": "Campus Flyover", "approach": ApproachEnum.NORTH},
    ("J-02", "J-03"): {"dist_km": 4.2, "base_sec": 260, "name": "Campus Flyover", "approach": ApproachEnum.EAST},
    ("J-04", "J-02"): {"dist_km": 3.5, "base_sec": 220, "name": "Station Terminal Road", "approach": ApproachEnum.SOUTH},
    ("J-02", "J-04"): {"dist_km": 3.5, "base_sec": 220, "name": "Station Terminal Road", "approach": ApproachEnum.NORTH},
    ("J-05", "J-04"): {"dist_km": 2.7, "base_sec": 170, "name": "Hospital Access Way", "approach": ApproachEnum.WEST},
    ("J-04", "J-05"): {"dist_km": 2.7, "base_sec": 170, "name": "Hospital Access Way", "approach": ApproachEnum.SOUTH},
    ("J-03", "J-05"): {"dist_km": 3.9, "base_sec": 250, "name": "Outer Ring Bypass", "approach": ApproachEnum.NORTH},
    ("J-05", "J-03"): {"dist_km": 3.9, "base_sec": 250, "name": "Outer Ring Bypass", "approach": ApproachEnum.WEST},

    # -------------------------------------------------------------
    # MUMBAI METROPOLITAN SEGMENTS
    # -------------------------------------------------------------
    ("J-BOM-01", "J-BOM-02"): {"dist_km": 5.4, "base_sec": 320, "name": "BKC-Dadar SCLR Connector", "approach": ApproachEnum.SOUTH},
    ("J-BOM-02", "J-BOM-01"): {"dist_km": 5.4, "base_sec": 320, "name": "BKC-Dadar SCLR Connector", "approach": ApproachEnum.NORTH},
    ("J-BOM-02", "J-BOM-03"): {"dist_km": 8.2, "base_sec": 480, "name": "Dr. Ambedkar Rd to Marine Drive", "approach": ApproachEnum.SOUTH},
    ("J-BOM-03", "J-BOM-02"): {"dist_km": 8.2, "base_sec": 480, "name": "Dr. Ambedkar Rd to Marine Drive", "approach": ApproachEnum.NORTH},
    ("J-BOM-01", "J-BOM-04"): {"dist_km": 6.8, "base_sec": 380, "name": "Western Express Highway BKC-Andheri", "approach": ApproachEnum.NORTH},
    ("J-BOM-04", "J-BOM-01"): {"dist_km": 6.8, "base_sec": 380, "name": "Western Express Highway BKC-Andheri", "approach": ApproachEnum.SOUTH},
    ("J-BOM-01", "J-BOM-05"): {"dist_km": 14.5, "base_sec": 720, "name": "Sion-Panvel Vashi Creek Expressway", "approach": ApproachEnum.EAST},
    ("J-BOM-05", "J-BOM-01"): {"dist_km": 14.5, "base_sec": 720, "name": "Sion-Panvel Vashi Creek Expressway", "approach": ApproachEnum.WEST},
    ("J-BOM-02", "J-BOM-04"): {"dist_km": 10.2, "base_sec": 550, "name": "Dadar-Andheri Arterial Link", "approach": ApproachEnum.NORTH},
    ("J-BOM-04", "J-BOM-02"): {"dist_km": 10.2, "base_sec": 550, "name": "Dadar-Andheri Arterial Link", "approach": ApproachEnum.SOUTH},
    ("J-BOM-03", "J-BOM-05"): {"dist_km": 21.0, "base_sec": 1100, "name": "Eastern Freeway - Navi Mumbai Line", "approach": ApproachEnum.EAST},
    ("J-BOM-05", "J-BOM-03"): {"dist_km": 21.0, "base_sec": 1100, "name": "Eastern Freeway - Navi Mumbai Line", "approach": ApproachEnum.WEST},

    # -------------------------------------------------------------
    # HYDERABAD METROPOLITAN SEGMENTS
    # -------------------------------------------------------------
    ("J-HYD-01", "J-HYD-02"): {"dist_km": 4.1, "base_sec": 240, "name": "Hitec-Gachibowli BioDiversity Link", "approach": ApproachEnum.WEST},
    ("J-HYD-02", "J-HYD-01"): {"dist_km": 4.1, "base_sec": 240, "name": "Hitec-Gachibowli BioDiversity Link", "approach": ApproachEnum.EAST},
    ("J-HYD-01", "J-HYD-03"): {"dist_km": 3.6, "base_sec": 210, "name": "Durgam Cheruvu Cable Bridge Corridor", "approach": ApproachEnum.EAST},
    ("J-HYD-03", "J-HYD-01"): {"dist_km": 3.6, "base_sec": 210, "name": "Durgam Cheruvu Cable Bridge Corridor", "approach": ApproachEnum.WEST},
    ("J-HYD-03", "J-HYD-04"): {"dist_km": 6.8, "base_sec": 420, "name": "Panjagutta Begumpet Arterial", "approach": ApproachEnum.EAST},
    ("J-HYD-04", "J-HYD-03"): {"dist_km": 6.8, "base_sec": 420, "name": "Panjagutta Begumpet Arterial", "approach": ApproachEnum.WEST},
    ("J-HYD-04", "J-HYD-05"): {"dist_km": 9.5, "base_sec": 580, "name": "Tank Bund - Charminar Heritage Line", "approach": ApproachEnum.SOUTH},
    ("J-HYD-05", "J-HYD-04"): {"dist_km": 9.5, "base_sec": 580, "name": "Tank Bund - Charminar Heritage Line", "approach": ApproachEnum.NORTH},
    ("J-HYD-02", "J-HYD-05"): {"dist_km": 16.2, "base_sec": 890, "name": "PVNR Elevated Airport Expressway", "approach": ApproachEnum.EAST},
    ("J-HYD-05", "J-HYD-02"): {"dist_km": 16.2, "base_sec": 890, "name": "PVNR Elevated Airport Expressway", "approach": ApproachEnum.WEST},

    # -------------------------------------------------------------
    # BENGALURU METROPOLITAN SEGMENTS
    # -------------------------------------------------------------
    ("J-BLR-01", "J-BLR-02"): {"dist_km": 9.8, "base_sec": 520, "name": "Hosur Elevated Expressway (Silk-ECity)", "approach": ApproachEnum.SOUTH},
    ("J-BLR-02", "J-BLR-01"): {"dist_km": 9.8, "base_sec": 520, "name": "Hosur Elevated Expressway (Silk-ECity)", "approach": ApproachEnum.NORTH},
    ("J-BLR-01", "J-BLR-03"): {"dist_km": 2.5, "base_sec": 180, "name": "Koramangala 80ft Linkway", "approach": ApproachEnum.NORTH},
    ("J-BLR-03", "J-BLR-01"): {"dist_km": 2.5, "base_sec": 180, "name": "Koramangala 80ft Linkway", "approach": ApproachEnum.SOUTH},
    ("J-BLR-03", "J-BLR-04"): {"dist_km": 4.6, "base_sec": 310, "name": "Intermediate Ring Road to Indiranagar", "approach": ApproachEnum.NORTH},
    ("J-BLR-04", "J-BLR-03"): {"dist_km": 4.6, "base_sec": 310, "name": "Intermediate Ring Road to Indiranagar", "approach": ApproachEnum.SOUTH},
    ("J-BLR-04", "J-BLR-05"): {"dist_km": 3.2, "base_sec": 220, "name": "Old Madras Road to MG Road", "approach": ApproachEnum.WEST},
    ("J-BLR-05", "J-BLR-04"): {"dist_km": 3.2, "base_sec": 220, "name": "Old Madras Road to MG Road", "approach": ApproachEnum.EAST},
    ("J-BLR-05", "J-BLR-01"): {"dist_km": 6.7, "base_sec": 410, "name": "Hosur Road Richmond Corridor", "approach": ApproachEnum.SOUTH},
    ("J-BLR-01", "J-BLR-05"): {"dist_km": 6.7, "base_sec": 410, "name": "Hosur Road Richmond Corridor", "approach": ApproachEnum.NORTH},
}

class NavigationEngine:
    def __init__(self):
        self.adj_graph: Dict[str, List[str]] = {}
        for (u, v) in ROAD_SEGMENTS.keys():
            if u not in self.adj_graph:
                self.adj_graph[u] = []
            if v not in self.adj_graph[u]:
                self.adj_graph[u].append(v)

    def get_corridor_cost(
        self, 
        u: str, 
        v: str, 
        active_incidents: List, 
        active_ambulances: List, 
        weather_cond: Optional[Dict] = None
    ) -> CorridorCostDetail:
        seg = ROAD_SEGMENTS.get((u, v))
        if not seg:
            # Fallback default
            seg = {
                "dist_km": 3.0, 
                "base_sec": 180, 
                "name": f"Connecting Corridor {u}-{v}", 
                "approach": ApproachEnum.NORTH
            }

        base_sec = seg["base_sec"]
        dist_km = seg["dist_km"]
        road_name = seg["name"]
        approach = seg["approach"]

        # 1. Live Traffic Factor
        traffic_mult = 1.0
        congestion_level = "LOW"
        try:
            target_state = junction_repo.get_state(v)
            if target_state:
                app_state = getattr(target_state, approach.value.lower(), None)
                if app_state:
                    q = app_state.queue_length
                    if q > 35:
                        traffic_mult = 3.2
                        congestion_level = "VERY_HIGH"
                    elif q > 20:
                        traffic_mult = 2.2
                        congestion_level = "HIGH"
                    elif q > 10:
                        traffic_mult = 1.5
                        congestion_level = "MEDIUM"
                    else:
                        traffic_mult = 1.0
                        congestion_level = "LOW"
        except Exception:
            pass

        # 2. Weather Surface Friction Factor
        weather_mult = 1.0
        if weather_cond:
            surface = weather_cond.get("road_surface_condition", "DRY")
            rain = weather_cond.get("rain_mm_per_hour", 0.0)
            if surface in ["WATERLOGGED", "FLOODED"] or rain > 20:
                weather_mult = 1.5
            elif surface == "WET" or rain > 5:
                weather_mult = 1.25

        # 3. Active Incident Blockage
        has_incident = False
        incident_penalty = 0
        for inc in active_incidents:
            if inc.junction_id == v and inc.approach == approach:
                has_incident = True
                incident_penalty = 999  # Heavy penalty for blocked lane
                congestion_level = "SEVERE_INCIDENT_BLOCKAGE"
                break

        # 4. Emergency Priority Preemption
        has_emergency = False
        emergency_mission_id = None
        emergency_priority = 0
        for amb in active_ambulances:
            for node in amb.route_corridor:
                if node.junction_id == v and node.approach == approach:
                    has_emergency = True
                    emergency_mission_id = amb.mission_id
                    emergency_priority = amb.priority_level
                    break
            if has_emergency:
                break

        est_transit = int(base_sec * traffic_mult * weather_mult) + incident_penalty

        return CorridorCostDetail(
            origin_junction_id=u,
            destination_junction_id=v,
            road_name=road_name,
            approach=approach,
            base_distance_km=dist_km,
            free_flow_seconds=base_sec,
            live_traffic_multiplier=traffic_mult,
            weather_multiplier=weather_mult,
            has_active_incident=has_incident,
            has_active_emergency=has_emergency,
            active_emergency_mission_id=emergency_mission_id,
            active_emergency_priority=emergency_priority,
            estimated_transit_seconds=est_transit,
            congestion_level=congestion_level
        )

    def compute_optimal_route(self, req: NavigationRequest) -> NavigationResponse:
        origin = req.origin_junction_id
        dest = req.destination_junction_id

        # Fetch live context
        active_incidents = [i for i in incident_repo.list_incidents() if getattr(i, 'status', None) == 'ACTIVE']
        all_missions = ambulance_repo.list_missions()
        active_ambulances = [
            m for m in all_missions 
            if getattr(m, 'status', None) not in ['MISSION_ACCOMPLISHED', 'CANCELLED']
        ]
        try:
            weather_data = weather_service.get_weather_for_junction(origin).model_dump()
        except Exception:
            weather_data = None

        # Build dynamic edge costs
        costs: Dict[Tuple[str, str], CorridorCostDetail] = {}
        for (u, v) in ROAD_SEGMENTS.keys():
            costs[(u, v)] = self.get_corridor_cost(
                u, v, active_incidents, active_ambulances, weather_data
            )

        # 1. Compute Dynamic Dijkstra for Optimal Fastest Path
        distances = {node: float('inf') for node in self.adj_graph}
        distances[origin] = 0
        pq = [(0, origin, [origin])]
        best_path = [origin]
        best_time = 0

        while pq:
            curr_cost, curr_node, path = heapq.heappop(pq)
            if curr_node == dest:
                best_path = path
                best_time = curr_cost
                break
            if curr_cost > distances.get(curr_node, float('inf')):
                continue

            for neighbor in self.adj_graph.get(curr_node, []):
                cost_detail = costs.get((curr_node, neighbor))
                edge_weight = cost_detail.estimated_transit_seconds if cost_detail else 180
                new_cost = curr_cost + edge_weight

                if new_cost < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_cost
                    heapq.heappush(pq, (new_cost, neighbor, path + [neighbor]))

        # 2. Compute Static Distance-Only Shortest Path for comparison
        dist_only = {node: float('inf') for node in self.adj_graph}
        dist_only[origin] = 0
        pq_static = [(0, origin, [origin])]
        static_path = [origin]

        while pq_static:
            curr_dist, curr_node, path = heapq.heappop(pq_static)
            if curr_node == dest:
                static_path = path
                break
            if curr_dist > dist_only.get(curr_node, float('inf')):
                continue
            for neighbor in self.adj_graph.get(curr_node, []):
                seg = ROAD_SEGMENTS.get((curr_node, neighbor), {"dist_km": 3.0})
                d = seg["dist_km"]
                if curr_dist + d < dist_only.get(neighbor, float('inf')):
                    dist_only[neighbor] = curr_dist + d
                    heapq.heappush(pq_static, (curr_dist + d, neighbor, path + [neighbor]))

        # Format Turn-by-Turn Steps
        steps: List[NavigationStep] = []
        total_dist = 0.0
        total_time = 0
        free_flow_time = 0
        emergency_warnings: List[str] = []

        for i in range(len(best_path) - 1):
            u = best_path[i]
            v = best_path[i + 1]
            c = costs.get((u, v))
            if not c:
                c = self.get_corridor_cost(u, v, active_incidents, active_ambulances, weather_data)

            total_dist += c.base_distance_km
            total_time += c.estimated_transit_seconds
            free_flow_time += c.free_flow_seconds

            warning_text = None
            if c.has_active_emergency:
                msg = f"🚨 EMERGENCY CLEARANCE ACTIVE on {c.road_name} ({c.origin_junction_id} ➔ {c.destination_junction_id}): Priority {c.active_emergency_priority} Emergency Vehicle approaching. Public vehicles must keep left and clear intersection."
                warning_text = msg
                if msg not in emergency_warnings:
                    emergency_warnings.append(msg)

            advisory = None
            if c.has_active_incident:
                advisory = f"⚠️ Hazard Reported on approach. Flow restricted. Detour recommended."
            elif c.congestion_level in ["HIGH", "VERY_HIGH"]:
                advisory = f"⚠️ Heavy traffic queue. Adaptive signal green extension enabled."

            instruction = f"From {u}, proceed along {c.road_name} toward {v} via {c.approach.value} approach."

            steps.append(NavigationStep(
                step_number=i + 1,
                from_junction_id=u,
                to_junction_id=v,
                road_name=c.road_name,
                approach=c.approach,
                instruction=instruction,
                distance_km=c.base_distance_km,
                eta_seconds=c.estimated_transit_seconds,
                congestion_level=c.congestion_level,
                emergency_active=c.has_active_emergency,
                emergency_priority=c.active_emergency_priority,
                emergency_warning=warning_text,
                advisory_notes=advisory
            ))

        # Junction names
        origin_j = junction_repo.get_junction(origin)
        dest_j = junction_repo.get_junction(dest)
        origin_name = origin_j.get("name", f"Junction {origin}") if origin_j else f"Junction {origin}"
        dest_name = dest_j.get("name", f"Junction {dest}") if dest_j else f"Junction {dest}"

        delay_seconds = max(0, total_time - free_flow_time)
        
        # Calculate time saved vs congested direct route if rerouted
        delay_saved = 0
        if best_path != static_path and len(static_path) > 1:
            static_time = 0
            for i in range(len(static_path) - 1):
                su, sv = static_path[i], static_path[i + 1]
                sc = costs.get((su, sv))
                static_time += sc.estimated_transit_seconds if sc else 200
            delay_saved = max(0, static_time - total_time)

        # Formatted time
        mins = total_time // 60
        secs = total_time % 60
        formatted_time = f"{mins} min {secs} sec" if mins > 0 else f"{secs} sec"

        load_advisory = None
        if len(best_path) > 2:
            load_advisory = f"Multi-commuter load balancing active: Distributed transit via {best_path[1]} to prevent arterial congestion."

        weather_adv = None
        if weather_data and weather_data.get("road_surface_condition") != "DRY":
            weather_adv = f"Weather Caution: Road surface is {weather_data.get('road_surface_condition')}. Recommended max speed is {weather_data.get('recommended_speed_kmh', 45)} km/h."

        return NavigationResponse(
            origin_junction_id=origin,
            destination_junction_id=dest,
            origin_name=origin_name,
            destination_name=dest_name,
            vehicle_type=req.vehicle_type,
            total_distance_km=round(total_dist, 2),
            estimated_travel_time_seconds=total_time,
            estimated_travel_time_formatted=formatted_time,
            delay_seconds=delay_seconds,
            delay_saved_seconds=delay_saved,
            optimal_route_junctions=best_path,
            alternative_route_junctions=static_path if static_path != best_path else [],
            steps=steps,
            emergency_corridor_warnings=emergency_warnings,
            load_balancing_advisory=load_advisory,
            weather_impact_advisory=weather_adv
        )

    def list_all_corridor_statuses(self) -> List[CorridorCostDetail]:
        active_incidents = [i for i in incident_repo.list_incidents() if getattr(i, 'status', None) == 'ACTIVE']
        all_missions = ambulance_repo.list_missions()
        active_ambulances = [
            m for m in all_missions 
            if getattr(m, 'status', None) not in ['MISSION_ACCOMPLISHED', 'CANCELLED']
        ]
        try:
            weather_data = weather_service.get_weather_for_junction("J-01").model_dump()
        except Exception:
            weather_data = None

        res = []
        for (u, v) in ROAD_SEGMENTS.keys():
            res.append(self.get_corridor_cost(u, v, active_incidents, active_ambulances, weather_data))
        return res

navigation_engine = NavigationEngine()
