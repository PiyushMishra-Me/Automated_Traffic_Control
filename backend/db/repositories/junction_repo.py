from datetime import datetime, timezone
from typing import List, Optional
from backend.db.mongo_client import db_manager
from backend.models.traffic_schemas import JunctionCreate, JunctionInfo

_memory_junctions: dict[str, dict] = {
    # -------------------------------------------------------------
    # NEW DELHI / NCR METROPOLITAN GRID
    # -------------------------------------------------------------
    "J-01": {
        "junction_id": "J-01",
        "name": "Central Plaza Interchange (Connaught Place)",
        "location": "Connaught Outer Circle & Barakhamba Rd",
        "city": "DELHI",
        "latitude": 28.6315,
        "longitude": 77.2167,
        "road_names": {
            "NORTH": "North Boulevard (Minto Rd)",
            "SOUTH": "South Radial (Janpath Ave)",
            "EAST": "East Arterial (Barakhamba Rd)",
            "WEST": "West Linkway (KG Marg)"
        },
        "connected_junctions": ["J-02", "J-03", "J-04", "J-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-02": {
        "junction_id": "J-02",
        "name": "ITO & Vikas Marg Corridor",
        "location": "ITO Junction & Ring Road East",
        "city": "DELHI",
        "latitude": 28.6280,
        "longitude": 77.2410,
        "road_names": {
            "NORTH": "Delhi Gate Extension",
            "SOUTH": "Pragati Maidan Bypass",
            "EAST": "Vikas Marg Eastway",
            "WEST": "Barakhamba Link Road"
        },
        "connected_junctions": ["J-01", "J-03", "J-04"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-03": {
        "junction_id": "J-03",
        "name": "Civil Lines & University Crossing",
        "location": "North Ring Road & Mall Road",
        "city": "DELHI",
        "latitude": 28.6750,
        "longitude": 77.2180,
        "road_names": {
            "NORTH": "Grand Trunk Outer Arterial",
            "SOUTH": "Minto Road Corridor",
            "EAST": "Campus Flyover East",
            "WEST": "Ridge Road Westway"
        },
        "connected_junctions": ["J-01", "J-02", "J-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-04": {
        "junction_id": "J-04",
        "name": "AIIMS & South Medical Expressway",
        "location": "Ring Road South & Aurobindo Marg",
        "city": "DELHI",
        "latitude": 28.5672,
        "longitude": 77.2100,
        "road_names": {
            "NORTH": "Janpath South Radial",
            "SOUTH": "Mehrauli Medical Link",
            "EAST": "Lajpat Nagar Ring Road",
            "WEST": "Hospital Emergency Access Way"
        },
        "connected_junctions": ["J-01", "J-02", "J-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-05": {
        "junction_id": "J-05",
        "name": "Dhaula Kuan & Airport Arterial",
        "location": "Vande Mataram Marg & Ring Road West",
        "city": "DELHI",
        "latitude": 28.5920,
        "longitude": 77.1610,
        "road_names": {
            "NORTH": "Diplomatic Enclave Lane",
            "SOUTH": "Cantonment Access Road",
            "EAST": "KG Marg West Connector",
            "WEST": "Airport Express Highway"
        },
        "connected_junctions": ["J-01", "J-03", "J-04"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },

    # -------------------------------------------------------------
    # MUMBAI METROPOLITAN GRID
    # -------------------------------------------------------------
    "J-BOM-01": {
        "junction_id": "J-BOM-01",
        "name": "BKC Central Financial Core",
        "location": "Bandra-Kurla Complex & BKC Connector",
        "city": "MUMBAI",
        "latitude": 19.0657,
        "longitude": 72.8688,
        "road_names": {
            "NORTH": "Bandra East Flyover",
            "SOUTH": "SCLR Kurla Expressway",
            "EAST": "Chunabhatti Link Road",
            "WEST": "Kalanagar Junction Approach"
        },
        "connected_junctions": ["J-BOM-02", "J-BOM-03", "J-BOM-04", "J-BOM-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-BOM-02": {
        "junction_id": "J-BOM-02",
        "name": "Dadar TT Central Circle",
        "location": "Dr. Ambedkar Road & Tilak Bridge",
        "city": "MUMBAI",
        "latitude": 19.0178,
        "longitude": 72.8478,
        "road_names": {
            "NORTH": "Sion Matunga Arterial",
            "SOUTH": "Lalbaug Parel Flyover",
            "EAST": "Wadala Harbor Link Road",
            "WEST": "Shivaji Park Beachway"
        },
        "connected_junctions": ["J-BOM-01", "J-BOM-03", "J-BOM-04"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-BOM-03": {
        "junction_id": "J-BOM-03",
        "name": "Marine Drive & Nariman Point",
        "location": "Netaji Subhash Chandra Bose Rd & Free Press Marg",
        "city": "MUMBAI",
        "latitude": 18.9280,
        "longitude": 72.8220,
        "road_names": {
            "NORTH": "Charni Road Queen Necklace",
            "SOUTH": "Nariman Point Loop",
            "EAST": "Churchgate Fort Link",
            "WEST": "Arabian Sea Coastal Road"
        },
        "connected_junctions": ["J-BOM-01", "J-BOM-02", "J-BOM-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-BOM-04": {
        "junction_id": "J-BOM-04",
        "name": "Andheri Western Express Interchange",
        "location": "WEH & Andheri-Kurla Road",
        "city": "MUMBAI",
        "latitude": 19.1197,
        "longitude": 72.8464,
        "road_names": {
            "NORTH": "Borivali Highway North",
            "SOUTH": "Bandra WEH Highway South",
            "EAST": "Airport Terminal 2 Link",
            "WEST": "SV Road JVPD Connector"
        },
        "connected_junctions": ["J-BOM-01", "J-BOM-02", "J-BOM-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-BOM-05": {
        "junction_id": "J-BOM-05",
        "name": "Vashi Toll & Navi Mumbai Gateway",
        "location": "Sion-Panvel Highway & Vashi Bridge",
        "city": "MUMBAI",
        "latitude": 19.0645,
        "longitude": 72.9975,
        "road_names": {
            "NORTH": "Thane Belapur Road",
            "SOUTH": "Panvel Expressway",
            "EAST": "Palm Beach Road Vashi",
            "WEST": "Mankhurd Bridge Link"
        },
        "connected_junctions": ["J-BOM-01", "J-BOM-03", "J-BOM-04"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },

    # -------------------------------------------------------------
    # HYDERABAD METROPOLITAN GRID
    # -------------------------------------------------------------
    "J-HYD-01": {
        "junction_id": "J-HYD-01",
        "name": "Hitec City Cyber Towers Junction",
        "location": "Madhapur Main Rd & Hitec City Flyover",
        "city": "HYDERABAD",
        "latitude": 17.4504,
        "longitude": 78.3808,
        "road_names": {
            "NORTH": "Kondapur IT Arterial",
            "SOUTH": "Durgam Cheruvu Cable Bridge",
            "EAST": "Jubilee Hills 100ft Road",
            "WEST": "Mindspace Tech Loop"
        },
        "connected_junctions": ["J-HYD-02", "J-HYD-03", "J-HYD-04", "J-HYD-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-HYD-02": {
        "junction_id": "J-HYD-02",
        "name": "Gachibowli ORR Interchange",
        "location": "Gachibowli Flyover & Outer Ring Road",
        "city": "HYDERABAD",
        "latitude": 17.4401,
        "longitude": 78.3489,
        "road_names": {
            "NORTH": "Financial District Radial",
            "SOUTH": "Airport ORR Expressway",
            "EAST": "Biodiversity Junction Link",
            "WEST": "Nanakramguda IT Way"
        },
        "connected_junctions": ["J-HYD-01", "J-HYD-03", "J-HYD-04"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-HYD-03": {
        "junction_id": "J-HYD-03",
        "name": "Jubilee Hills Checkpost Crossing",
        "location": "Road No. 36 & Road No. 1",
        "city": "HYDERABAD",
        "latitude": 17.4325,
        "longitude": 78.4072,
        "road_names": {
            "NORTH": "Madhapur Flyover Link",
            "SOUTH": "Banjara Hills Rd No 12",
            "EAST": "Panjagutta Main Corridor",
            "WEST": "KBR Park Radial Way"
        },
        "connected_junctions": ["J-HYD-01", "J-HYD-02", "J-HYD-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-HYD-04": {
        "junction_id": "J-HYD-04",
        "name": "Begumpet Airport Flyover Crossing",
        "location": "Rashtrapati Road & Sardar Patel Rd",
        "city": "HYDERABAD",
        "latitude": 17.4448,
        "longitude": 78.4660,
        "road_names": {
            "NORTH": "Secunderabad Station Arterial",
            "SOUTH": "Raj Bhavan Road",
            "EAST": "Paradise Circle Link",
            "WEST": "Panjagutta Flyover West"
        },
        "connected_junctions": ["J-HYD-01", "J-HYD-02", "J-HYD-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-HYD-05": {
        "junction_id": "J-HYD-05",
        "name": "Charminar & Old City Heritage Corridor",
        "location": "Pathergatti & Madina Crossing",
        "city": "HYDERABAD",
        "latitude": 17.3616,
        "longitude": 78.4747,
        "road_names": {
            "NORTH": "Afzal Gunj Bridge Road",
            "SOUTH": "Falaknuma Palace Radial",
            "EAST": "Mir Alam Mandi Way",
            "WEST": "High Court River Road"
        },
        "connected_junctions": ["J-HYD-01", "J-HYD-03", "J-HYD-04"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },

    # -------------------------------------------------------------
    # BENGALURU METROPOLITAN GRID
    # -------------------------------------------------------------
    "J-BLR-01": {
        "junction_id": "J-BLR-01",
        "name": "Silk Board Central Interchange",
        "location": "Hosur Road & Outer Ring Road South",
        "city": "BENGALURU",
        "latitude": 12.9176,
        "longitude": 77.6238,
        "road_names": {
            "NORTH": "Koramangala 80ft Road",
            "SOUTH": "Electronic City Elevated Highway",
            "EAST": "HSR Layout Ring Road",
            "WEST": "BTM Layout 2nd Stage"
        },
        "connected_junctions": ["J-BLR-02", "J-BLR-03", "J-BLR-04", "J-BLR-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-BLR-02": {
        "junction_id": "J-BLR-02",
        "name": "Electronic City Toll Plaza",
        "location": "Hosur Elevated Expressway & Phase 1 Gate",
        "city": "BENGALURU",
        "latitude": 12.8399,
        "longitude": 77.6770,
        "road_names": {
            "NORTH": "Silk Board Elevated Flyover",
            "SOUTH": "Attibele State Border",
            "EAST": "Wipro Tech Park Avenue",
            "WEST": "Neeladri Road Junction"
        },
        "connected_junctions": ["J-BLR-01", "J-BLR-03", "J-BLR-04"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-BLR-03": {
        "junction_id": "J-BLR-03",
        "name": "Koramangala Sony World Junction",
        "location": "100 Feet Intermediate Ring Road & 80 Feet Road",
        "city": "BENGALURU",
        "latitude": 12.9352,
        "longitude": 77.6245,
        "road_names": {
            "NORTH": "Domlur Flyover Connector",
            "SOUTH": "Silk Board Radial",
            "EAST": "Sarjapur Main Road",
            "WEST": "Forum Mall Hosur Road"
        },
        "connected_junctions": ["J-BLR-01", "J-BLR-02", "J-BLR-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-BLR-04": {
        "junction_id": "J-BLR-04",
        "name": "Indiranagar 100ft Road Junction",
        "location": "CMH Road & 100 Feet Road",
        "city": "BENGALURU",
        "latitude": 12.9719,
        "longitude": 77.6412,
        "road_names": {
            "NORTH": "Old Madras Road Highway",
            "SOUTH": "Domlur EGL Interchange",
            "EAST": "Thippasandra Main Road",
            "WEST": "Halasuru Lake Road"
        },
        "connected_junctions": ["J-BLR-01", "J-BLR-02", "J-BLR-05"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    },
    "J-BLR-05": {
        "junction_id": "J-BLR-05",
        "name": "MG Road & Trinity Circle",
        "location": "MG Road & Old Airport Road Interchange",
        "city": "BENGALURU",
        "latitude": 12.9754,
        "longitude": 77.6186,
        "road_names": {
            "NORTH": "Cubbon Park Avenue",
            "SOUTH": "Hosur Road Richmond Town",
            "EAST": "Indiranagar 100ft Link",
            "WEST": "Brigade Road Commercial Hub"
        },
        "connected_junctions": ["J-BLR-01", "J-BLR-03", "J-BLR-04"],
        "created_at": datetime.now(timezone.utc),
        "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"]
    }
}

class JunctionRepository:
    def __init__(self):
        pass

    @property
    def collection(self):
        db = db_manager.get_database()
        if db is not None:
            return db["junctions"]
        return None

    def create_junction(self, junction: JunctionCreate) -> dict:
        doc = {
            "junction_id": junction.junction_id,
            "name": junction.name,
            "location": junction.location or "",
            "city": junction.city or "DELHI",
            "latitude": junction.latitude,
            "longitude": junction.longitude,
            "road_names": junction.road_names or {},
            "connected_junctions": junction.connected_junctions or [],
            "created_at": datetime.now(timezone.utc),
            "approaches_configured": ["NORTH", "SOUTH", "EAST", "WEST"],
            "custom_counting_lines": junction.custom_counting_lines.dict() if junction.custom_counting_lines else {}
        }
        col = self.collection
        if col is not None:
            try:
                col.update_one({"junction_id": junction.junction_id}, {"$set": doc}, upsert=True)
            except Exception:
                pass
        _memory_junctions[junction.junction_id] = doc
        return doc

    def get_junction(self, junction_id: str) -> Optional[dict]:
        col = self.collection
        if col is not None:
            try:
                res = col.find_one({"junction_id": junction_id})
                if res:
                    res["_id"] = str(res["_id"])
                    return res
            except Exception:
                pass
        return _memory_junctions.get(junction_id)

    def list_junctions(self, city: Optional[str] = None) -> List[dict]:
        col = self.collection
        if col is not None:
            try:
                query = {"city": city.upper()} if city else {}
                items = list(col.find(query))
                if items:
                    for item in items:
                        item["_id"] = str(item["_id"])
                    return items
            except Exception:
                pass
        all_j = list(_memory_junctions.values())
        if city:
            return [j for j in all_j if j.get("city", "DELHI").upper() == city.upper()]
        return all_j

    def update_counting_lines(self, junction_id: str, counting_lines: dict) -> Optional[dict]:
        """Persist calibrated normalized counting lines without overwriting junction details."""
        existing = self.get_junction(junction_id)
        if not existing:
            return None

        serializable_lines = {
            str(approach.value if hasattr(approach, "value") else approach): (
                config.model_dump() if hasattr(config, "model_dump") else config
            )
            for approach, config in counting_lines.items()
        }
        col = self.collection
        if col is not None:
            try:
                col.update_one(
                    {"junction_id": junction_id},
                    {"$set": {"custom_counting_lines": serializable_lines}},
                )
            except Exception:
                pass

        memory_doc = _memory_junctions.get(junction_id, existing)
        memory_doc["custom_counting_lines"] = serializable_lines
        _memory_junctions[junction_id] = memory_doc
        return memory_doc

junction_repo = JunctionRepository()
