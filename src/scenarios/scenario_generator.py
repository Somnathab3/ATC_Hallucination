
# src/scenarios/scenario_generator.py
import os, json, math
from typing import Dict, Any, List

CENTER_LAT = 52.0
CENTER_LON = 4.0
ALT_FT     = 10000.0
SPD_KT     = 250.0

def nm_to_lat(nm: float) -> float:
    return nm / 60.0

def nm_to_lon(nm: float, at_lat_deg: float) -> float:
    return nm / (60.0 * max(1e-6, math.cos(math.radians(at_lat_deg))))

def pos_offset(lat0: float, lon0: float, north_nm: float, east_nm: float):
    return (lat0 + nm_to_lat(north_nm),
            lon0 + nm_to_lon(east_nm, lat0))

def save(name: str, agents: List[Dict[str, Any]], notes: str) -> str:
    obj = {
        "scenario_name": name,
        "seed": 42,
        "notes": notes,
        "center": {"lat": CENTER_LAT, "lon": CENTER_LON, "alt_ft": ALT_FT},
        "sim_dt_s": 1.0,
        "agents": agents,
    }
    os.makedirs("scenarios", exist_ok=True)
    path = os.path.join("scenarios", f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return path

# ---------- 1) HEAD-ON (2 aircraft) ----------
def make_head_on(approach_nm: float = 18.0) -> str:
    # A1 south->north, A2 north->south, both aligned through center
    a1_lat, a1_lon = pos_offset(CENTER_LAT, CENTER_LON, -approach_nm, 0.0)
    a2_lat, a2_lon = pos_offset(CENTER_LAT, CENTER_LON, +approach_nm, 0.0)
    agents = [
        {"id":"A1","type":"A320","lat":a1_lat,"lon":a1_lon,"hdg_deg":  0.0,"spd_kt":SPD_KT,"alt_ft":ALT_FT,
         "waypoint": {"lat": CENTER_LAT+nm_to_lat(+approach_nm), "lon": CENTER_LON}},
        {"id":"A2","type":"A320","lat":a2_lat,"lon":a2_lon,"hdg_deg":180.0,"spd_kt":SPD_KT,"alt_ft":ALT_FT,
         "waypoint": {"lat": CENTER_LAT+nm_to_lat(-approach_nm), "lon": CENTER_LON}},
    ]
    return save("head_on", agents, "Two-aircraft head-on encounter along a straight line through center.")

# ---------- 2) T-FORMATION (3 aircraft) ----------
def make_t_formation(arm_nm: float = 20.0, stem_nm: float = 16.0, stem_x_nm: float = -6.0) -> str:
    """
    Horizontal 'top bar' through center (west->east).
    Vertical 'stem' offset to the west by stem_x_nm, moving north to meet the bar.
    """
    # Bar aircraft: west -> east
    bar_lat, bar_lon = pos_offset(CENTER_LAT, CENTER_LON, 0.0, -arm_nm)
    bar_wp_lat, bar_wp_lon = pos_offset(CENTER_LAT, CENTER_LON, 0.0, +arm_nm)
    # Stem aircraft: south -> north, offset x=-6 NM
    stem1_lat, stem1_lon = pos_offset(CENTER_LAT, CENTER_LON, -stem_nm, stem_x_nm)
    stem2_lat, stem2_lon = pos_offset(CENTER_LAT, CENTER_LON, -stem_nm*0.6, stem_x_nm)  # follower on the stem

    agents = [
        {"id":"A1","type":"A320","lat":bar_lat,"lon":bar_lon,"hdg_deg": 90.0,"spd_kt":SPD_KT,"alt_ft":ALT_FT,
         "waypoint":{"lat": bar_wp_lat, "lon": bar_wp_lon}},
        {"id":"A2","type":"A320","lat":stem1_lat,"lon":stem1_lon,"hdg_deg":  0.0,"spd_kt":SPD_KT,"alt_ft":ALT_FT,
         "waypoint":{"lat": CENTER_LAT+nm_to_lat(+stem_nm), "lon": CENTER_LON+nm_to_lon(stem_x_nm, CENTER_LAT)}},
        {"id":"A3","type":"A320","lat":stem2_lat,"lon":stem2_lon,"hdg_deg":  0.0,"spd_kt":SPD_KT,"alt_ft":ALT_FT,
         "waypoint":{"lat": CENTER_LAT+nm_to_lat(+stem_nm*0.8), "lon": CENTER_LON+nm_to_lon(stem_x_nm, CENTER_LAT)}},
    ]
    return save("t_formation", agents, "Three-aircraft 'T' formation: horizontal bar through center and a northbound stem 6 NM west.")

# ---------- 3) PARALLEL (3 aircraft, same track) ----------
def make_parallel(track_x_nm: float = 0.0, gaps_nm: float = 6.0, south_nm: float = 24.0) -> str:
    """
    Three aircraft flying north in trail on a single track (parallel same-lane).
    Equal speeds produce sustained in-trail separation pressure.
    """
    a1_lat, a1_lon = pos_offset(CENTER_LAT, CENTER_LON, -south_nm,           track_x_nm)
    a2_lat, a2_lon = pos_offset(CENTER_LAT, CENTER_LON, -south_nm + gaps_nm, track_x_nm)
    a3_lat, a3_lon = pos_offset(CENTER_LAT, CENTER_LON, -south_nm + 2*gaps_nm, track_x_nm)

    wp_lat, wp_lon = pos_offset(CENTER_LAT, CENTER_LON, +south_nm, track_x_nm)
    agents = []
    for i,(la,lo) in enumerate([(a1_lat,a1_lon),(a2_lat,a2_lon),(a3_lat,a3_lon)], start=1):
        agents.append({"id":f"A{i}","type":"A320","lat":la,"lon":lo,"hdg_deg":0.0,"spd_kt":SPD_KT,"alt_ft":ALT_FT,
                       "waypoint":{"lat": wp_lat, "lon": wp_lon}})
    return save("parallel", agents, "Three-aircraft parallel/in-trail northbound on one track with 6 NM initial gaps.")

# ---------- 4) CONVERGING (4 aircraft, oblique bearings) ----------
def make_converging(radius_nm: float = 18.0, bearings_deg=(45.0, 135.0, 225.0, 315.0)) -> str:
    """
    Four aircraft converging to center on oblique bearings (not orthogonal).
    Each starts radius_nm away along the reverse bearing and aims to pass through center.
    """
    agents = []
    for i, brg in enumerate(bearings_deg, start=1):
        # start at reverse bearing from center
        back = (brg + 180.0) % 360.0
        north =  radius_nm * math.cos(math.radians(back - 0.0))  # projection onto N
        east  =  radius_nm * math.sin(math.radians(back - 0.0))  # projection onto E
        lat, lon = pos_offset(CENTER_LAT, CENTER_LON, north, east)
        # heading toward the center ≈ brg
        agents.append({
            "id": f"A{i}",
            "type": "A320",
            "lat":  lat,
            "lon":  lon,
            "hdg_deg": float(brg),
            "spd_kt": SPD_KT,
            "alt_ft": ALT_FT,
            "waypoint": {"lat": CENTER_LAT, "lon": CENTER_LON}
        })
    return save("converging", agents, "Four-aircraft converging conflict on oblique bearings (45°,135°,225°,315°).")

if __name__ == "__main__":
    paths = [
        make_head_on(),
        make_t_formation(),
        make_parallel(),
        make_converging(),
    ]
    for p in paths:
        print("Wrote:", p)
