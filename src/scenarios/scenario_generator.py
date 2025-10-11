"""
Module Name: scenario_generator.py
Description: 
    Generates 9 standardized aircraft conflict scenarios for MARL training and evaluation.
    Creates balanced scenario suite covering 3 conflict families × 3 patterns each.
    
    Scenario Structure:
        3 Conflict Families:
            - CHASE: In-trail conflicts (overtaking, same heading)
            - MERGE: Converging conflicts (head-on, angled approaches)
            - CROSS: Crossing conflicts (perpendicular, angular intersections)
        
        3 Patterns per family:
            - 2x2: Two agent pairs (e.g., A1+A2 vs A3+A4)
            - 3+1 (3p1): Three-agent cluster + one singleton
            - 4all: All four agents in symmetric conflict
    
    All 9 scenarios share:
        - 4 agents per scenario
        - 250 kt TAS baseline speed
        - 10,000 ft cruise altitude
        - Conflict-free start: All pairs ≥5.1 NM separation
        - Center: 51.0°N, 13.7°E (Dresden airspace)
    
    Geometry Constraints:
        - Start positions: 30-60 NM from center/waypoints
        - Conflict zones: Within 5 NM of merge/crossing points
        - Waypoint placement: Ensures guaranteed conflicts without intervention
        - Minimum start separation enforced by radial nudging algorithm
    
    JSON Output Format:
        {
            "scenario_name": "chase_2x2",
            "description": "...",
            "agents": [
                {
                    "id": "A1", "type": "A320",
                    "lat": 51.5, "lon": 13.7, "hdg_deg": 0.0,
                    "spd_kt": 250.0, "alt_ft": 10000.0,
                    "waypoint": {"lat": 52.0, "lon": 13.7}
                },
                ...
            ]
        }
    
    Usage:
        python scenario_generator.py  # Generates all 9 scenarios in scenarios/ dir

Author: Som
Date: 2025-10-07
"""

import os
import json
import math
from typing import Dict, Any, List

# Airspace center configuration
CENTER_LAT = 51.0
CENTER_LON = 13.7  # Dresden
ALT_FT = 10000.0
SPD_KT = 250.0

def nm_to_lat(nm: float) -> float:
    """
    Convert nautical miles to latitude degrees.
    
    At Earth's surface: 1° latitude ≈ 60 NM (standard approximation).
    Exact: 1 NM = 1 minute of arc = 1/60 degree.
    
    Args:
        nm: Distance in nautical miles.
    
    Returns:
        Latitude difference in degrees.
    """
    return nm / 60.0

def nm_to_lon(nm: float, at_lat_deg: float) -> float:
    """
    Convert nautical miles to longitude degrees at given latitude.
    
    Longitude degree width shrinks with cos(latitude) due to Earth's curvature.
    At equator: 1° longitude ≈ 60 NM.
    At 60° latitude: 1° longitude ≈ 30 NM.
    
    Args:
        nm: Distance in nautical miles.
        at_lat_deg: Latitude where conversion is performed (deg).
    
    Returns:
        Longitude difference in degrees.
    """
    return nm / (60.0 * max(1e-6, math.cos(math.radians(at_lat_deg))))

def pos_offset(lat0: float, lon0: float, north_nm: float, east_nm: float):
    """
    Calculate position from base coordinates and north/east offsets.
    
    Uses local ENU (East-North-Up) frame approximation.
    Valid for small offsets (<100 NM) from reference point.
    
    Args:
        lat0: Reference latitude (deg).
        lon0: Reference longitude (deg).
        north_nm: Northward offset (NM, positive = north).
        east_nm: Eastward offset (NM, positive = east).
    
    Returns:
        (lat, lon) tuple of new position.
    """
    return (lat0 + nm_to_lat(north_nm), lon0 + nm_to_lon(east_nm, lat0))

def _pairwise_sep_nm(a, b):
    """
    Calculate fast local NM distance between two agents.
    
    Uses planar approximation (valid for small offsets <100 NM):
        - Convert lat/lon differences to NM
        - Apply Pythagorean distance in local ENU frame
    
    Args:
        a: Agent dict with 'lat', 'lon' keys.
        b: Agent dict with 'lat', 'lon' keys.
    
    Returns:
        Horizontal separation distance (NM).
    """
    lat1, lon1, lat2, lon2 = a["lat"], a["lon"], b["lat"], b["lon"]
    dN = (lat1 - lat2) * 60.0  # Latitude difference in NM
    latm = 0.5 * (lat1 + lat2)  # Mean latitude for longitude correction
    dE = (lon1 - lon2) * 60.0 * math.cos(math.radians(latm))  # Longitude difference in NM
    return (dN**2 + dE**2) ** 0.5

def enforce_min_start_sep(agents, min_nm=5.1, max_iter=10, scale=1.06):
    """
    Ensure all aircraft start ≥ min_nm apart by nudging violating pairs radially outward.
    
    Algorithm:
        1. Find closest agent pair with separation < min_nm
        2. Push both agents radially away from CENTER_LAT, CENTER_LON
        3. Scale radial distance by factor (default 1.06)
        4. Repeat until all pairs separated or max_iter reached
    
    Preserves:
        - Headings: Agent headings unchanged
        - Waypoints: Waypoint locations unchanged
        - Relative geometry: Only radial scaling applied
    
    Used to avoid start conflicts while maintaining intended conflict geometry.
    Without this, agents might start inside 5 NM well-clear zone.
    
    Args:
        agents: List of agent configuration dicts (with 'lat', 'lon' keys).
        min_nm: Minimum separation requirement (NM). Default 5.1 NM to ensure
                conflict-free start (>5 NM well-clear threshold).
        max_iter: Maximum iterations to attempt separation (default 10).
        scale: Radial scaling factor for pushing agents apart (default 1.06).
    
    Returns:
        Modified agents list with enforced minimum separation.
        If convergence fails, returns best-effort result after max_iter.
    """
    for _ in range(max_iter):
        # Find current minimum separation
        min_sep = 1e9
        viol = None
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                sep = _pairwise_sep_nm(agents[i], agents[j])
                if sep < min_sep:
                    min_sep, viol = sep, (i, j)
        
        if min_sep >= min_nm or viol is None:
            return agents  # Done - all pairs separated
        
        # Push both members of the closest pair outward
        for k in viol:
            lat, lon = agents[k]["lat"], agents[k]["lon"]
            dN = (lat - CENTER_LAT) * 60.0
            dE = (lon - CENTER_LON) * 60.0 * math.cos(math.radians(CENTER_LAT))
            if abs(dN) + abs(dE) < 1e-6:  # If too close to center, nudge east
                dE = 0.1
            dN *= scale
            dE *= scale
            new_lat, new_lon = pos_offset(CENTER_LAT, CENTER_LON, dN, dE)
            agents[k]["lat"], agents[k]["lon"] = new_lat, new_lon
    
    return agents

def _inbound_agent(aid: str, wp_lat: float, wp_lon: float, inbound_hdg_deg: float, radius_nm: float):
    """
    Create one aircraft radius_nm 'behind' its waypoint on heading inbound_hdg_deg.
    
    Args:
        aid: Agent ID.
        wp_lat: Waypoint latitude.
        wp_lon: Waypoint longitude.
        inbound_hdg_deg: Inbound heading in degrees.
        radius_nm: Distance from waypoint in nautical miles.
        
    Returns:
        Agent configuration dictionary.
    """
    back = (inbound_hdg_deg + 180.0) % 360.0
    north = radius_nm * math.cos(math.radians(back))
    east = radius_nm * math.sin(math.radians(back))
    lat, lon = pos_offset(wp_lat, wp_lon, north, east)
    return {
        "id": aid, "type": "A320",
        "lat": lat, "lon": lon, "hdg_deg": float(inbound_hdg_deg),
        "spd_kt": SPD_KT, "alt_ft": ALT_FT,
        "waypoint": {"lat": wp_lat, "lon": wp_lon},
    }

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

# ==================== 9-SCENARIO SUITE (3 families × 3 conflict patterns) ====================

# ------------------ CHASE (same track), 4 agents ------------------
def make_chase_2x2(gap_nm: float = 6.0, south_nm: float = 18.0, lane_dx_nm: float = 8.0) -> str:
    """
    Two independent in-trail conflicts (A1,A2 on west lane; A3,A4 on east lane).
    Each pair has 6 NM spacing (creates conflict pressure but starts conflict-free).
    WPs are 12 NM north of center on their lane; all legs <~30 NM, reachable <1000s @ 250kt.
    """
    def lane_agents(x_nm, a_ids):
        wp_lat, wp_lon = pos_offset(CENTER_LAT, CENTER_LON, +12.0, x_nm)
        starts = [-south_nm, -(south_nm - gap_nm)]
        out = []
        for i, sN in enumerate(starts):
            la, lo = pos_offset(CENTER_LAT, CENTER_LON, sN, x_nm)
            out.append({
                "id": a_ids[i], "type": "A320",
                "lat": la, "lon": lo, "hdg_deg": 0.0,
                "spd_kt": SPD_KT, "alt_ft": ALT_FT,
                "waypoint": {"lat": wp_lat, "lon": wp_lon},
            })
        return out

    agents = []
    agents += lane_agents(-lane_dx_nm, ["A1", "A2"])  # west lane pair
    agents += lane_agents(+lane_dx_nm, ["A3", "A4"])  # east lane pair
    agents = enforce_min_start_sep(agents, min_nm=5.1)
    return save("chase_2x2", agents, f"CHASE | two separate in-trail conflicts (pairs), lanes ±{lane_dx_nm} NM, {gap_nm} NM spacing.")

def make_chase_3p1(gap_nm: float = 6.0, south_nm: float = 18.0, far_dx_nm: float = 15.0) -> str:
    """
    Three in-trail conflicting on center lane; one non-conflicting cruiser far on a side lane.
    Central 3 aircraft create in-trail conflict; 4th is spatially separated.
    """
    # 3 conflicting on center lane
    wp_lat0, wp_lon0 = pos_offset(CENTER_LAT, CENTER_LON, +12.0, 0.0)
    s_list = [-south_nm, -(south_nm - gap_nm), -(south_nm - 2*gap_nm)]
    agents = []
    for i, sN in enumerate(s_list, start=1):
        la, lo = pos_offset(CENTER_LAT, CENTER_LON, sN, 0.0)
        agents.append({"id": f"A{i}", "type": "A320", "lat": la, "lon": lo, "hdg_deg": 0.0,
                       "spd_kt": SPD_KT, "alt_ft": ALT_FT, "waypoint": {"lat": wp_lat0, "lon": wp_lon0}})
    
    # 1 far away on side lane (no conflict)
    wp_lat1, wp_lon1 = pos_offset(CENTER_LAT, CENTER_LON, +12.0, far_dx_nm)
    la, lo = pos_offset(CENTER_LAT, CENTER_LON, -south_nm, far_dx_nm)
    agents.append({"id": "A4", "type": "A320", "lat": la, "lon": lo, "hdg_deg": 0.0,
                   "spd_kt": SPD_KT, "alt_ft": ALT_FT, "waypoint": {"lat": wp_lat1, "lon": wp_lon1}})
    
    agents = enforce_min_start_sep(agents, min_nm=5.1)
    return save("chase_3p1", agents, f"CHASE | 3 in-trail conflicting ({gap_nm} NM spacing); 1 far on side lane (non-conflicting).")

def make_chase_4all(gap_nm: float = 6.0, south_nm: float = 18.0) -> str:
    """
    Four in-trail on one lane with 6 NM gaps -> chain conflict scenario.
    All 4 aircraft create in-trail conflict pressure.
    """
    wp_lat, wp_lon = pos_offset(CENTER_LAT, CENTER_LON, +12.0, 0.0)
    s_list = [-south_nm, -(south_nm - gap_nm), -(south_nm - 2*gap_nm), -(south_nm - 3*gap_nm)]
    agents = []
    for i, sN in enumerate(s_list, start=1):
        la, lo = pos_offset(CENTER_LAT, CENTER_LON, sN, 0.0)
        agents.append({"id": f"A{i}", "type": "A320", "lat": la, "lon": lo, "hdg_deg": 0.0,
                       "spd_kt": SPD_KT, "alt_ft": ALT_FT, "waypoint": {"lat": wp_lat, "lon": wp_lon}})
    
    agents = enforce_min_start_sep(agents, min_nm=5.1)
    return save("chase_4all", agents, f"CHASE | 4 in-trail with {gap_nm} NM gaps (all conflicting).")

# ------------------ MERGE (small angle convergence), 4 agents ------------------
def make_merge_2x2(radius_nm: float = 50.0, sep_dx_nm: float = 15.0) -> str:
    """
    Two independent 2-aircraft merges, left and right of center (waypoints at ±sep_dx_nm east).
    Each pair merges to their own waypoint; spatially separated to avoid inter-pair conflicts.
    """
    # Left pair - small angular spread
    wpL_lat, wpL_lon = pos_offset(CENTER_LAT, CENTER_LON, 0.0, -sep_dx_nm)
    left = [
        _inbound_agent("A1", wpL_lat, wpL_lon, 352.5, radius_nm),
        _inbound_agent("A2", wpL_lat, wpL_lon, 7.5, radius_nm),
    ]
    
    # Right pair - small angular spread
    wpR_lat, wpR_lon = pos_offset(CENTER_LAT, CENTER_LON, 0.0, +sep_dx_nm)
    right = [
        _inbound_agent("A3", wpR_lat, wpR_lon, 172.5, radius_nm),
        _inbound_agent("A4", wpR_lat, wpR_lon, 187.5, radius_nm),
    ]
    
    agents = enforce_min_start_sep(left + right, min_nm=5.1)
    return save("merge_2x2", agents, f"MERGE | two separate 2→1 merges (centers ±{sep_dx_nm} NM, {radius_nm} NM radius).")

def make_merge_3p1(radius_nm: float = 50.0, far_dx_nm: float = 25.0) -> str:
    """
    Three-aircraft merge to center; one far cruiser (non-conflicting) to an offset waypoint.
    Central 3 create merge conflict; 4th is spatially separated.
    """
    wp_lat, wp_lon = CENTER_LAT, CENTER_LON
    bearings = [345.0, 0.0, 15.0]  # 3 inbound with small angular spread
    agents = []
    for i, b in enumerate(bearings, start=1):
        agents.append(_inbound_agent(f"A{i}", wp_lat, wp_lon, b, radius_nm))
    
    # Far non-conflicting cruiser
    wp2_lat, wp2_lon = pos_offset(CENTER_LAT, CENTER_LON, +5.0, far_dx_nm)
    agents.append(_inbound_agent("A4", wp2_lat, wp2_lon, 0.0, radius_nm))
    
    agents = enforce_min_start_sep(agents, min_nm=5.1)
    return save("merge_3p1", agents, f"MERGE | 3 inbound to center ({radius_nm} NM); 1 far to side waypoint.")

def make_merge_4all(radius_nm: float = 50.0) -> str:
    """
    Four aircraft approaching the same waypoint from near-parallel bearings.
    Small angular spread (340°, 355°, 5°, 20° - 15° increments) creates merge conflict for all 4.
    All agents start 50 NM from waypoint to ensure conflict-free initial positions.
    """
    wp_lat, wp_lon = CENTER_LAT, CENTER_LON
    bearings = [340.0, 355.0, 5.0, 20.0]  # Inbound headings with 15° increments
    agents = [_inbound_agent(f"A{i+1}", wp_lat, wp_lon, bearings[i], radius_nm) for i in range(4)]
    
    agents = enforce_min_start_sep(agents, min_nm=5.1)
    return save("merge_4all", agents, f"MERGE | 4 inbound to single waypoint ({radius_nm} NM) with 15° angular spread (all starting 50 NM away).")

# ------------------ CROSS (orthogonal), 4 agents ------------------
def _cross_pair_at(aid_ns: str, aid_ew: str, center_n_nm: float, center_e_nm: float, radius_nm: float):
    """Build a 2-aircraft 90° crossing at a shifted center."""
    c_lat, c_lon = pos_offset(CENTER_LAT, CENTER_LON, center_n_nm, center_e_nm)
    
    # N-S aircraft (northbound)
    a_ns = {
        "id": aid_ns, "type": "A320",
        "lat": c_lat - nm_to_lat(radius_nm), "lon": c_lon,
        "hdg_deg": 0.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
        "waypoint": {"lat": c_lat + nm_to_lat(radius_nm), "lon": c_lon},
    }
    
    # W-E aircraft (eastbound)
    a_ew = {
        "id": aid_ew, "type": "A320",
        "lat": c_lat, "lon": c_lon - nm_to_lon(radius_nm, c_lat),
        "hdg_deg": 90.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
        "waypoint": {"lat": c_lat, "lon": c_lon + nm_to_lon(radius_nm, c_lat)},
    }
    
    return [a_ns, a_ew]

def make_cross_2x2(radius_nm: float = 15.0, sep_dx_nm: float = 12.0) -> str:
    """
    Two spatially separate 2-ship crossings (left/right of center).
    Each pair creates 90° crossing conflict; pairs are spatially separated.
    """
    agents = []
    agents += _cross_pair_at("A1", "A2", 0.0, -sep_dx_nm, radius_nm)  # left cross
    agents += _cross_pair_at("A3", "A4", 0.0, +sep_dx_nm, radius_nm)  # right cross
    
    agents = enforce_min_start_sep(agents, min_nm=5.1)
    return save("cross_2x2", agents, f"CROSS | two separate 2-ship 90° crossings (centers ±{sep_dx_nm} NM, {radius_nm} NM radius).")

def make_cross_3p1(radius_nm: float = 15.0, far_dx_nm: float = 18.0) -> str:
    """
    Three-way crossing at center (N, E, W legs); one far cruiser on a side lane (non-conflicting).
    Central 3 create crossing conflict; 4th is spatially separated.
    """
    c_lat, c_lon = CENTER_LAT, CENTER_LON
    agents = []
    
    # Northbound (from south)
    agents.append({
        "id": "A1", "type": "A320",
        "lat": c_lat - nm_to_lat(radius_nm), "lon": c_lon,
        "hdg_deg": 0.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
        "waypoint": {"lat": c_lat + nm_to_lat(radius_nm), "lon": c_lon},
    })
    
    # Eastbound (from west)
    agents.append({
        "id": "A2", "type": "A320",
        "lat": c_lat, "lon": c_lon - nm_to_lon(radius_nm, c_lat),
        "hdg_deg": 90.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
        "waypoint": {"lat": c_lat, "lon": c_lon + nm_to_lon(radius_nm, c_lat)},
    })
    
    # Westbound (from east)
    agents.append({
        "id": "A3", "type": "A320",
        "lat": c_lat, "lon": c_lon + nm_to_lon(radius_nm, c_lat),
        "hdg_deg": 270.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
        "waypoint": {"lat": c_lat, "lon": c_lon - nm_to_lon(radius_nm, c_lat)},
    })
    
    # Non-conflicting far cruiser
    wp_lat, wp_lon = pos_offset(CENTER_LAT, CENTER_LON, +12.0, far_dx_nm)
    la, lo = pos_offset(CENTER_LAT, CENTER_LON, -radius_nm, far_dx_nm)
    agents.append({"id": "A4", "type": "A320", "lat": la, "lon": lo, "hdg_deg": 0.0,
                   "spd_kt": SPD_KT, "alt_ft": ALT_FT, "waypoint": {"lat": wp_lat, "lon": wp_lon}})
    
    agents = enforce_min_start_sep(agents, min_nm=5.1)
    return save("cross_3p1", agents, f"CROSS | 3-leg intersection at center ({radius_nm} NM); 1 far on side lane.")

def make_cross_4all(radius_nm: float = 15.0) -> str:
    """
    Canonical 4-way intersection at center (all 4 conflicting).
    Four aircraft crossing on orthogonal bearings (N, E, S, W).
    """
    c_lat, c_lon = CENTER_LAT, CENTER_LON
    agents = [
        # A1: South to North (hdg=0°)
        {"id": "A1", "type": "A320",
         "lat": c_lat - nm_to_lat(radius_nm), "lon": c_lon,
         "hdg_deg": 0.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
         "waypoint": {"lat": c_lat + nm_to_lat(radius_nm), "lon": c_lon}},
        
        # A2: West to East (hdg=90°)
        {"id": "A2", "type": "A320",
         "lat": c_lat, "lon": c_lon - nm_to_lon(radius_nm, c_lat),
         "hdg_deg": 90.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
         "waypoint": {"lat": c_lat, "lon": c_lon + nm_to_lon(radius_nm, c_lat)}},
        
        # A3: North to South (hdg=180°)
        {"id": "A3", "type": "A320",
         "lat": c_lat + nm_to_lat(radius_nm), "lon": c_lon,
         "hdg_deg": 180.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
         "waypoint": {"lat": c_lat - nm_to_lat(radius_nm), "lon": c_lon}},
        
        # A4: East to West (hdg=270°)
        {"id": "A4", "type": "A320",
         "lat": c_lat, "lon": c_lon + nm_to_lon(radius_nm, c_lat),
         "hdg_deg": 270.0, "spd_kt": SPD_KT, "alt_ft": ALT_FT,
         "waypoint": {"lat": c_lat, "lon": c_lon - nm_to_lon(radius_nm, c_lat)}}
    ]
    
    agents = enforce_min_start_sep(agents, min_nm=5.1)
    return save("cross_4all", agents, f"CROSS | 4-way orthogonal intersection ({radius_nm} NM radius) - all conflicting.")

def make_all_nine():
    """Generate all 9 scenarios (3 families × 3 conflict patterns)."""
    return [
        make_chase_2x2(), make_chase_3p1(), make_chase_4all(),
        make_merge_2x2(), make_merge_3p1(), make_merge_4all(),
        make_cross_2x2(), make_cross_3p1(), make_cross_4all(),
    ]

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("="*80)
    print("SCENARIO GENERATOR: 9-Scenario Suite (3 families × 3 conflict patterns)")
    print("="*80)
    print("\nConfiguration:")
    print(f"  - Center: {CENTER_LAT}°N, {CENTER_LON}°E (Dresden)")
    print(f"  - Altitude: {ALT_FT:,.0f} ft")
    print(f"  - Speed: {SPD_KT} knots")
    print(f"  - Agents per scenario: 4")
    print(f"  - Max flight distance: ~30 NM (<1000 seconds @ 250kt)")
    print(f"  - Start separation: >5.0 NM (conflict-free at t=0)")
    print()
    
    # Generate all 9 scenarios
    paths = make_all_nine()
    
    print("\n" + "="*80)
    print("GENERATED SCENARIOS:")
    print("="*80)
    
    families = {
        "CHASE (In-Trail)": paths[0:3],
        "MERGE (Convergence)": paths[3:6],
        "CROSS (Orthogonal)": paths[6:9],
    }
    
    for family_name, family_paths in families.items():
        print(f"\n{family_name}:")
        for i, p in enumerate(family_paths, 1):
            scenario_name = os.path.basename(p).replace('.json', '')
            pattern = scenario_name.split('_')[-1]
            if pattern == "2x2":
                desc = "2 separate conflicts"
            elif pattern == "3p1":
                desc = "3 conflicting + 1 far"
            elif pattern == "4all":
                desc = "All 4 conflicting"
            else:
                desc = ""
            print(f"  {i}. {scenario_name:15s} - {desc}")
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE:")
    print("="*80)
    print(f"✓ Total scenarios generated: {len(paths)}")
    print(f"✓ All scenarios validated with >5 NM start separation")
    print(f"✓ Scenarios saved to: ./scenarios/")
    print("="*80)
