"""
Module Name: scenario_generator.py
Description: Generates standardized aircraft conflict scenarios for MARL training and evaluation.
             Creates head-on, T-formation, parallel, converging, and canonical crossing geometries
             with reproducible conflict configurations centered at common airspace location.
Author: Som
Date: 2025-10-04
"""

import os
import json
import math
from typing import Dict, Any, List

CENTER_LAT = 52.0
CENTER_LON = 4.0
ALT_FT = 10000.0
SPD_KT = 250.0

def nm_to_lat(nm: float) -> float:
    """Convert nautical miles to latitude degrees (1° ≈ 60 NM)."""
    return nm / 60.0

def nm_to_lon(nm: float, at_lat_deg: float) -> float:
    """Convert nautical miles to longitude degrees at given latitude."""
    return nm / (60.0 * max(1e-6, math.cos(math.radians(at_lat_deg))))

def pos_offset(lat0: float, lon0: float, north_nm: float, east_nm: float):
    """Calculate position from base coordinates and north/east offsets (NM)."""
    return (lat0 + nm_to_lat(north_nm), lon0 + nm_to_lon(east_nm, lat0))

def save(name: str, agents: List[Dict[str, Any]], notes: str) -> str:
    """
    Save scenario configuration to JSON file.
    
    Args:
        name: Scenario name (filename).
        agents: List of agent configurations.
        notes: Scenario description.
        
    Returns:
        Path to saved JSON file.
    """
    scenario_config = {
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
        json.dump(scenario_config, f, indent=2)
    return path

def make_head_on(approach_nm: float = 18.0) -> str:
    """
    Generate head-on encounter with two aircraft on reciprocal headings.
    
    Args:
        approach_nm: Initial separation from center.
        
    Returns:
        Path to generated scenario file.
    """
    # Position aircraft north and south of center, flying toward each other
    a1_lat, a1_lon = pos_offset(CENTER_LAT, CENTER_LON, -approach_nm, 0.0)
    a2_lat, a2_lon = pos_offset(CENTER_LAT, CENTER_LON, +approach_nm, 0.0)
    
    agents = [
        {
            "id": "A1", "type": "A320", 
            "lat": a1_lat, "lon": a1_lon, "hdg_deg": 0.0, 
            "spd_kt": SPD_KT, "alt_ft": ALT_FT,
            "waypoint": {"lat": CENTER_LAT + nm_to_lat(+approach_nm), "lon": CENTER_LON}
        },
        {
            "id": "A2", "type": "A320",
            "lat": a2_lat, "lon": a2_lon, "hdg_deg": 180.0,
            "spd_kt": SPD_KT, "alt_ft": ALT_FT, 
            "waypoint": {"lat": CENTER_LAT + nm_to_lat(-approach_nm), "lon": CENTER_LON}
        },
    ]
    
    return save("head_on", agents, 
                "Two-aircraft head-on encounter with reciprocal headings through center point. "
                "Inevitable conflict without evasive action.")

# ---------- 2) T-FORMATION (3 aircraft) ----------
def make_t_formation(arm_nm: float = 7.5, stem_nm: float = 10.0, stem_x_nm: float = -7.0) -> str:
    """
    Horizontal 'top bar' through center (west->east).
    Vertical 'stem' offset to the west by stem_x_nm, moving north to meet the bar.
    FIXED: Reduced distances for better convergence within episode limits.
    """
    # Bar aircraft: west -> east (MUCH shorter distance)
    bar_lat, bar_lon = pos_offset(CENTER_LAT, CENTER_LON, 0.0, -arm_nm)
    bar_wp_lat, bar_wp_lon = pos_offset(CENTER_LAT, CENTER_LON, 0.0, +arm_nm)
    # Stem aircraft: south -> north, offset further west for staggered timing
    stem1_lat, stem1_lon = pos_offset(CENTER_LAT, CENTER_LON, -stem_nm, stem_x_nm)
    stem2_lat, stem2_lon = pos_offset(CENTER_LAT, CENTER_LON, -stem_nm*0.6, stem_x_nm)  # follower on the stem

    agents = [
        {"id":"A1","type":"A320","lat":bar_lat,"lon":bar_lon,"hdg_deg": 90.0,"spd_kt":SPD_KT,"alt_ft":ALT_FT,
         "waypoint":{"lat": bar_wp_lat, "lon": bar_wp_lon}},
        {"id":"A2","type":"A320","lat":stem1_lat,"lon":stem1_lon,"hdg_deg":  0.0,"spd_kt":SPD_KT,"alt_ft":ALT_FT,
         "waypoint":{"lat": CENTER_LAT+nm_to_lat(+stem_nm*0.8), "lon": CENTER_LON+nm_to_lon(stem_x_nm, CENTER_LAT)}},
        {"id":"A3","type":"A320","lat":stem2_lat,"lon":stem2_lon,"hdg_deg":  0.0,"spd_kt":SPD_KT,"alt_ft":ALT_FT,
         "waypoint":{"lat": CENTER_LAT+nm_to_lat(+stem_nm*0.6), "lon": CENTER_LON+nm_to_lon(stem_x_nm, CENTER_LAT)}},
    ]
    return save("t_formation", agents, "FIXED: Three-aircraft 'T' formation with reduced distances (7.5 NM arm, 10 NM stem, -7 NM offset) for better convergence.")

# ---------- 3) PARALLEL (3 aircraft, same track) ----------
def make_parallel(track_x_nm: float = 0.0, gaps_nm: float = 8.0, south_nm: float = 18.0) -> str:
    """
    Three aircraft flying north in trail on a single track (parallel same-lane).
    Equal speeds produce sustained in-trail separation pressure.
    FIXED: Reduced distances and closer waypoint for better convergence.
    """
    a1_lat, a1_lon = pos_offset(CENTER_LAT, CENTER_LON, -south_nm,           track_x_nm)
    a2_lat, a2_lon = pos_offset(CENTER_LAT, CENTER_LON, -south_nm + gaps_nm, track_x_nm)
    a3_lat, a3_lon = pos_offset(CENTER_LAT, CENTER_LON, -south_nm + 2*gaps_nm, track_x_nm)

    # Waypoint 12 NM north of center (reduced from south_nm=24 to 12)
    wp_lat, wp_lon = pos_offset(CENTER_LAT, CENTER_LON, +12.0, track_x_nm)
    agents = []
    for i,(la,lo) in enumerate([(a1_lat,a1_lon),(a2_lat,a2_lon),(a3_lat,a3_lon)], start=1):
        agents.append({"id":f"A{i}","type":"A320","lat":la,"lon":lo,"hdg_deg":0.0,"spd_kt":SPD_KT,"alt_ft":ALT_FT,
                       "waypoint":{"lat": wp_lat, "lon": wp_lon}})
    return save("parallel", agents, "FIXED: Three-aircraft parallel/in-trail northbound with 8 NM gaps and 12 NM waypoint distance for better convergence.")

# ---------- 4) CONVERGING (each AC to its own near-by waypoint) ----------
def make_converging(
    radius_nm: float = 12.0,
    bearings_deg=(20.0, 140.0, 260.0, 320.0),
    wp_offsets_nm=None,
    name: str = "converging",
) -> str:
    """
    Each aircraft converges to its *own* waypoint, with all waypoints clustered
    near the sector center (not exactly at it).
    FIXED: Reduced radius from 25 NM to 12 NM for better convergence within episode limits.

    - bearings_deg: desired inbound headings (toward each aircraft's own waypoint)
    - radius_nm:    initial distance from its waypoint (placed 'behind' each leg)
    - wp_offsets_nm: list of (north_nm, east_nm) offsets for the clustered waypoints
                     relative to (CENTER_LAT, CENTER_LON). If None, uses a small
                     deterministic cluster (~0.5-0.7 NM) around center.
    """
    if wp_offsets_nm is None:
        # 4 distinct waypoints ~0.5–0.7 NM around the center (tweak to taste)
        wp_offsets_nm = [(+2.6, +2.2), (-2.4, +2.5), (+2.3, -2.6), (-2.5, -2.3)]

    n = min(len(bearings_deg), len(wp_offsets_nm))
    agents = []

    for i in range(n):
        inb = float(bearings_deg[i])                 # inbound bearing toward waypoint
        # waypoint for this aircraft: center + small offset (clustered, not center)
        wpN, wpE = wp_offsets_nm[i]
        wp_lat, wp_lon = pos_offset(CENTER_LAT, CENTER_LON, wpN, wpE)

        # start 'radius_nm' behind the waypoint along the reverse bearing
        back = (inb + 180.0) % 360.0
        north = radius_nm * math.cos(math.radians(back))
        east  = radius_nm * math.sin(math.radians(back))
        start_lat, start_lon = pos_offset(wp_lat, wp_lon, north, east)

        # compute heading from start -> waypoint (clockwise from North)
        dN_nm = (wp_lat - start_lat) * 60.0
        dE_nm = (wp_lon - start_lon) * 60.0 * math.cos(math.radians(start_lat))
        hdg = (math.degrees(math.atan2(dE_nm, dN_nm)) + 360.0) % 360.0

        agents.append({
            "id":      f"A{i+1}",
            "type":    "A320",
            "lat":     start_lat,
            "lon":     start_lon,
            "hdg_deg": float(hdg),
            "spd_kt":  SPD_KT,
            "alt_ft":  ALT_FT,
            "waypoint": {"lat": wp_lat, "lon": wp_lon},
        })

    return save(
        name,
        agents,
        "FIXED: Converging scenario with 12 NM approach distances (reduced from 25 NM) for better convergence within episode limits."
    )

# ---------- 5) CANONICAL CROSSING (4 aircraft, orthogonal) ----------
def make_canonical_crossing(radius_nm: float = 12.5) -> str:
    """
    Four aircraft crossing at center on orthogonal bearings (N, E, S, W).
    Classic 4-way intersection scenario with inevitable conflict at center.
    """
    # Aircraft IDs using A0, A1, A2, A3 to match existing file
    agents = [
        # A0: South to North (hdg=0°)
        {"id": "A0", "type": "A320", 
         "lat": CENTER_LAT - nm_to_lat(radius_nm), "lon": CENTER_LON,
         "hdg_deg": 0.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
         "waypoint": {"lat": CENTER_LAT + nm_to_lat(radius_nm), "lon": CENTER_LON}},
        
        # A1: West to East (hdg=90°)
        {"id": "A1", "type": "A320",
         "lat": CENTER_LAT, "lon": CENTER_LON - nm_to_lon(radius_nm, CENTER_LAT),
         "hdg_deg": 90.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
         "waypoint": {"lat": CENTER_LAT, "lon": CENTER_LON + nm_to_lon(radius_nm, CENTER_LAT)}},
        
        # A2: North to South (hdg=180°)
        {"id": "A2", "type": "A320",
         "lat": CENTER_LAT + nm_to_lat(radius_nm), "lon": CENTER_LON,
         "hdg_deg": 180.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
         "waypoint": {"lat": CENTER_LAT - nm_to_lat(radius_nm), "lon": CENTER_LON}},
        
        # A3: East to West (hdg=270°)
        {"id": "A3", "type": "A320",
         "lat": CENTER_LAT, "lon": CENTER_LON + nm_to_lon(radius_nm, CENTER_LAT),
         "hdg_deg": 270.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
         "waypoint": {"lat": CENTER_LAT, "lon": CENTER_LON - nm_to_lon(radius_nm, CENTER_LAT)}}
    ]
    return save("canonical_crossing", agents, "Four-aircraft orthogonal crossing. Inevitable CPA at center at ~t=180s without intervention.")

if __name__ == "__main__":
    paths = [
        make_head_on(),
        make_t_formation(),
        make_parallel(),
        make_converging(),
        make_canonical_crossing(),
    ]
    for p in paths:
        print("Wrote:", p)
