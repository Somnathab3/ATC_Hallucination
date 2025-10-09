"""
Module Name: intrashift_tester.py
Description: Intrashift distribution testing framework for MARL robustness evaluation.
             Applies systematic perturbations to individual agents within a scenario to identify 
             failure modes and test generalization beyond training distribution.
             (Previously: targeted_shift_tester.py)
Author: Som
Date: 2025-10-04
"""

import os
import sys
import json
import time
import logging
import math
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# Ensure we have the correct project root by checking for key files
if not (os.path.exists(os.path.join(project_root, "scenarios")) and 
        os.path.exists(os.path.join(project_root, "atc_cli.py"))):
    # Fallback: try to find project root by looking for atc_cli.py
    search_dir = current_dir
    for _ in range(5):  # Search up to 5 levels up
        if (os.path.exists(os.path.join(search_dir, "scenarios")) and 
            os.path.exists(os.path.join(search_dir, "atc_cli.py"))):
            project_root = search_dir
            break
        search_dir = os.path.dirname(search_dir)

if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
import bluesky as bs
import ray

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from src.environment.marl_collision_env_minimal import MARLCollisionEnv
from src.analysis.viz_hooks import make_episode_visuals, make_run_visuals
from src.analysis.trajectory_comparison_map import generate_shift_comparison_maps
from src.analysis.trajectory_comparison_plot import create_shift_analysis_dashboard
from src.analysis.hallucination_detector_enhanced import HallucinationDetector


LOGGER = logging.getLogger("intrashift_tester")
LOGGER.setLevel(logging.INFO)


def _compute_metrics_from_trajectory_json(json_path: str, sep_nm: float = 5.0) -> Dict[str, Any]:
    """
    Compute hallucination metrics from trajectory JSON file using HallucinationDetector.
    
    This is used as a fallback when CSV extraction fails or CSV is missing.
    
    Args:
        json_path: Path to trajectory JSON file
        sep_nm: Separation threshold in nautical miles
    
    Returns:
        Dictionary of metrics computed by HallucinationDetector
    """
    try:
        if not os.path.exists(json_path):
            LOGGER.warning(f"Trajectory JSON not found: {json_path}")
            return {}
        
        with open(json_path, 'r') as f:
            trajectory = json.load(f)
        
        # Initialize detector with same params as environment
        hd = HallucinationDetector(
            horizon_s=120.0,
            action_thresh=(3.0, 5.0),
            res_window_s=60.0,
            action_period_s=10.0,
            los_threshold_nm=sep_nm,
            lag_pre_steps=1,
            lag_post_steps=1,
            debounce_n=2,
            debounce_m=3,
            iou_threshold=0.1
        )
        
        # Compute metrics
        metrics = hd.compute(trajectory, sep_nm=sep_nm, return_series=False)
        return metrics
        
    except Exception as e:
        LOGGER.error(f"Failed to compute metrics from trajectory JSON: {e}")
        return {}


def _compute_agent_role(agent_data: Dict[str, Any], center_lat: float, center_lon: float) -> str:
    """
    Compute geometric role (heading class × lane side) for an agent.
    
    Args:
        agent_data: Agent configuration with lat, lon, hdg_deg
        center_lat: Scenario center latitude
        center_lon: Scenario center longitude
    
    Returns:
        Role string like "N_west", "E_center", "S_east", "W_center"
    """
    hdg = agent_data["hdg_deg"]
    lat = agent_data["lat"]
    lon = agent_data["lon"]
    
    # Heading class: nearest 90° cardinal direction
    if 315 <= hdg or hdg < 45:
        hdg_class = "N"
    elif 45 <= hdg < 135:
        hdg_class = "E"
    elif 135 <= hdg < 225:
        hdg_class = "S"
    else:
        hdg_class = "W"
    
    # Lane side: for N/S use lon offset, for E/W use lat offset
    if hdg_class in ["N", "S"]:
        offset = lon - center_lon
    else:
        offset = lat - center_lat
    
    if abs(offset) < 0.05:  # Very close to center
        lane_side = "center"
    elif offset < 0:
        lane_side = "west" if hdg_class in ["N", "S"] else "south"
    else:
        lane_side = "east" if hdg_class in ["N", "S"] else "north"
    
    return f"{hdg_class}_{lane_side}"


def _select_representative_agents(scenario_path: str, scenario_name: str) -> List[str]:
    """
    Select representative agents based on geometric role deduplication.
    
    Args:
        scenario_path: Path to scenario JSON file
        scenario_name: Name of the scenario (e.g., "cross_2x2")
    
    Returns:
        List of agent IDs to test (deduplicated by role)
    """
    with open(scenario_path, "r") as f:
        scenario_data = json.load(f)
    
    center_lat = scenario_data["center"]["lat"]
    center_lon = scenario_data["center"]["lon"]
    agents = scenario_data["agents"]
    
    # Scenario-specific recommendations (overrides)
    recommendations = {
        "cross_2x2": ["A1", "A2"],  # N_west, E_center
        "cross_3p1": ["A1", "A2", "A3", "A4"],  # All distinct roles
        "cross_4all": ["A1", "A2", "A3", "A4"],  # Full 4-way symmetry
        "merge_2x2": ["A1", "A3"],  # Northbound, southbound merge
        "merge_3p1": ["A1", "A2", "A3"],  # East, center, west lanes (skip A4, same as A1)
        "merge_4all": ["A2", "A3"],  # East, west lanes
        "chase_2x2": ["A1", "A3"],  # West lane, east lane
        "chase_3p1": ["A1", "A3", "A4"],  # Trailer, leader, side stream
        "chase_4all": ["A1", "A4"]  # Trailer, leader
    }
    
    if scenario_name in recommendations:
        return recommendations[scenario_name]
    
    # Fallback: compute roles and keep one per role
    role_to_agent = {}
    for agent in agents:
        role = _compute_agent_role(agent, center_lat, center_lon)
        if role not in role_to_agent:
            role_to_agent[role] = agent["id"]
    
    return list(role_to_agent.values())


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points in nautical miles.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
    
    Returns:
        Distance in nautical miles
    """
    from math import radians, sin, cos, sqrt, atan2
    
    R_nm = 3440.065  # Earth radius in nautical miles
    
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R_nm * c


def _check_start_separation(scenario_path: str, shift_config: Dict[str, Dict[str, float]], 
                           separation_threshold_nm: float = 5.0) -> Tuple[bool, float]:
    """
    Check if applying shifts would cause t=0 conflict (start in LoS).
    
    Args:
        scenario_path: Path to scenario JSON file
        shift_config: Per-agent shift deltas
        separation_threshold_nm: Minimum acceptable separation (default 5.0 NM)
    
    Returns:
        (is_safe, min_separation_nm) tuple
    """
    with open(scenario_path, "r") as f:
        scenario_data = json.load(f)
    
    # Apply shifts to get hypothetical starting positions
    positions = {}
    for agent in scenario_data["agents"]:
        aid = agent["id"]
        base_lat = agent["lat"]
        base_lon = agent["lon"]
        
        # Apply position deltas
        delta = shift_config.get(aid, {})
        lat = base_lat + delta.get("position_lat_delta", 0.0)
        lon = base_lon + delta.get("position_lon_delta", 0.0)
        
        positions[aid] = (lat, lon)
    
    # Check all pairwise separations
    agent_ids = list(positions.keys())
    min_sep = float('inf')
    
    for i in range(len(agent_ids)):
        for j in range(i+1, len(agent_ids)):
            lat1, lon1 = positions[agent_ids[i]]
            lat2, lon2 = positions[agent_ids[j]]
            sep = _haversine_nm(lat1, lon1, lat2, lon2)
            min_sep = min(min_sep, sep)
    
    is_safe = min_sep >= separation_threshold_nm
    return is_safe, min_sep


def create_targeted_shift(agent_id: str, shift_type: str, shift_value: float, scenario_agents: List[str], 
                         shift_data: Optional[Any] = None) -> Dict[str, Any]:
    """
    Create targeted shift configuration affecting specific agent(s).
    
    This function generates shift parameters that modify only the specified agent
    while keeping others at baseline conditions, enabling isolation of individual
    agent impact on system behavior.
    
    Args:
        agent_id: Target agent identifier or "ALL" for system-wide changes
        shift_type: Category of modification (speed, position_closer, position_lateral, 
                   heading, aircraft_type, waypoint)
        shift_value: Magnitude of shift in appropriate units (kt, degrees, etc.)
        scenario_agents: Complete list of agent IDs in the scenario
        shift_data: Additional parameters for complex shifts (aircraft types, coordinates)
    
    Returns:
        Dictionary mapping agent IDs to their respective shift configurations
    """
    shift_config = {}
    
    # Handle "ALL" agents case for aircraft type shifts
    if agent_id == "ALL":
        target_agents = scenario_agents
    else:
        target_agents = [agent_id]
    
    for aid in scenario_agents:
        if aid in target_agents:
            # Apply shift to target agent(s)
            if shift_type == "speed":
                shift_config[aid] = {"speed_kt_delta": float(shift_value)}
            elif shift_type == "position_closer":
                # Move agent closer to others by reducing latitude separation
                # Positive values move north (closer to A3), negative south (closer to A1)
                shift_config[aid] = {"position_lat_delta": float(shift_value)}
            elif shift_type == "position_lateral":
                # Move agent east/west to create crossing scenarios
                shift_config[aid] = {"position_lon_delta": float(shift_value)}
            elif shift_type == "heading":
                shift_config[aid] = {"heading_deg_delta": float(shift_value)}
            elif shift_type == "aircraft_type":
                # Change aircraft type
                shift_config[aid] = {"aircraft_type": str(shift_data)}
            elif shift_type == "waypoint":
                # Shift waypoint position
                if isinstance(shift_data, dict) and "lat" in shift_data and "lon" in shift_data:
                    shift_config[aid] = {"waypoint_lat_delta": shift_data["lat"], "waypoint_lon_delta": shift_data["lon"]}
                else:
                    shift_config[aid] = {}
            else:
                shift_config[aid] = {}
        else:
            # Keep other agents at baseline
            shift_config[aid] = {}
    
    return shift_config


def _create_wind_config_for_shift(test_id: str, shift_type: str, shift_value: float) -> Dict[str, Any]:
    """
    Create wind and noise configuration based on the shift type and characteristics.
    
    Args:
        test_id: Unique test identifier
        shift_type: Type of shift being applied
        shift_value: Magnitude of the shift
    
    Returns:
        Dictionary containing wind and noise configuration
    """
    # Base configuration - wind only (noise/turbulence removed)
    env_shift: Dict[str, Any] = {
        # Wind configuration will be added based on shift type
    }
    
    # Different wind configurations based on shift characteristics
    if "baseline" in test_id:
        # Baseline: mild uniform wind
        env_shift["wind"] = {
            "mode": "uniform",
            "dir_deg": 270,  # West wind
            "kt": 15
        }
    elif "speed" in shift_type:
        # Speed shifts: crosswind to challenge airspeed management
        wind_strength = 20 + abs(shift_value) * 0.5  # Scale with shift magnitude
        env_shift["wind"] = {
            "mode": "uniform", 
            "dir_deg": 90,  # East crosswind (perpendicular to typical routes)
            "kt": min(wind_strength, 35)  # Cap at reasonable value
        }
    elif "position" in shift_type:
        # Position shifts: headwind/tailwind components to affect timing
        if "closer" in shift_type:
            # Agents moving closer: use tailwind to increase convergence pressure
            env_shift["wind"] = {
                "mode": "uniform",
                "dir_deg": 0,  # North wind (tailwind for south-bound traffic)
                "kt": 25
            }
        elif "lateral" in shift_type:
            # Lateral shifts: use layered winds to create vertical complexity
            env_shift["wind"] = {
                "mode": "layered",
                "layers": [
                    {"alt_ft": 10000, "dir_deg": 270, "kt": 20},  # West at FL100
                    {"alt_ft": 20000, "dir_deg": 90, "kt": 15},   # East at FL200
                    {"alt_ft": 30000, "dir_deg": 180, "kt": 25}   # South at FL300
                ]
            }
    elif "heading" in shift_type:
        # Heading shifts: strong crosswind to challenge course maintenance
        wind_dir = 45 if shift_value > 0 else 315  # NE or NW wind based on shift direction
        env_shift["wind"] = {
            "mode": "uniform",
            "dir_deg": wind_dir,
            "kt": 30
        }
    elif "aircraft" in shift_type:
        # Aircraft type changes: moderate uniform wind to test performance differences
        env_shift["wind"] = {
            "mode": "uniform",
            "dir_deg": 225,  # SW wind
            "kt": 18
        }
    elif "waypoint" in shift_type:
        # Waypoint shifts: complex layered wind to challenge path planning
        env_shift["wind"] = {
            "mode": "layered", 
            "layers": [
                {"alt_ft": 8000, "dir_deg": 180, "kt": 22},   # South at lower altitude
                {"alt_ft": 15000, "dir_deg": 270, "kt": 18},  # West at mid altitude  
                {"alt_ft": 25000, "dir_deg": 90, "kt": 15}    # East at higher altitude
            ]
        }
    else:
        # Default: moderate uniform wind
        env_shift["wind"] = {
            "mode": "uniform",
            "dir_deg": 270,
            "kt": 20
        }
    
    return env_shift


def create_conflict_inducing_shifts(scenario_agents: List[str], scenario_name: str = "", 
                                   scenario_path: str = "") -> List[Tuple[str, str, str, float, str, Optional[Any]]]:
    """
    Generate comprehensive test matrix of conflict-inducing shifts.
    
    Creates systematic variations across multiple dimensions to thoroughly test
    policy robustness. Shifts are categorized into micro (small perturbations)
    and macro (large deviations) ranges to identify training envelope boundaries.
    
    Uses geometry-based constraints to prevent t=0 conflicts and focuses on
    representative agents per geometric role.
    
    Args:
        scenario_agents: List of representative agent IDs to test
        scenario_name: Name of scenario (for geometry-specific caps)
        scenario_path: Path to scenario JSON (for agent role analysis)
    
    Returns:
        List of tuples containing:
        - test_id: Unique identifier for the test configuration
        - agent_id: Target agent to be modified
        - shift_type: Category of modification
        - shift_value: Numerical magnitude of shift
        - description: Human-readable description
        - shift_data: Additional parameters for complex modifications
    """
    shifts = []
    
    # BASELINE CASE - always include zero shift for comparison
    baseline_agent = scenario_agents[0] if scenario_agents else "A1"
    shifts.append(("baseline", baseline_agent, "speed", 0.0, "Baseline (no shifts)", None))
    
    # Speed variations - micro to macro range
    # Micro changes (small perturbations)
    speed_micro = [-10, -5, 5, 10]  # ±10 kt
    # Macro changes (large deviations to test failure modes)
    speed_macro = [-30, -20, 20, 30]  # ±30 kt
    
    for agent in scenario_agents:
        # Micro speed changes
        for delta in speed_micro:
            test_id = f"speed_micro_{agent}_{delta:+d}kt"
            desc = f"Agent {agent} speed micro-shift: {delta:+d} kt"
            shifts.append((test_id, agent, "speed", delta, desc, None))
        
        # Macro speed changes 
        for delta in speed_macro:
            test_id = f"speed_macro_{agent}_{delta:+d}kt"
            desc = f"Agent {agent} speed macro-shift: {delta:+d} kt"
            shifts.append((test_id, agent, "speed", delta, desc, None))
    
    # Position shifts to create conflicts (only for scenarios with 2+ agents)
    # Chase scenarios need special handling: only move in safe direction
    is_chase = "chase" in scenario_name
    
    if len(scenario_agents) >= 2:
        # Moving agents closer together (reducing separation)
        position_closer_micro = [0.05, 0.1, 0.15]  # 0.05-0.15 degrees ≈ 3-9 NM closer
        position_closer_macro = [0.2, 0.3]    # Cap at 0.3 for safety
        
        for agent in scenario_agents:
            # For chase, determine safe direction (away from nearest neighbor)
            # For simplicity, we'll allow both but rely on t=0 guard to filter unsafe
            for delta in position_closer_micro:
                # North movement
                test_id = f"pos_closer_micro_{agent}_north_{delta:.2f}deg"
                desc = f"{agent} moved {delta:.2f}° north (closer)"
                shifts.append((test_id, agent, "position_closer", delta, desc, None))
                
                # South movement
                test_id = f"pos_closer_micro_{agent}_south_{delta:.2f}deg"
                desc = f"{agent} moved {delta:.2f}° south (closer)"
                shifts.append((test_id, agent, "position_closer", -delta, desc, None))
            
            for delta in position_closer_macro:
                # North movement (major shift)
                test_id = f"pos_closer_macro_{agent}_north_{delta:.2f}deg"
                desc = f"{agent} moved {delta:.2f}° north (major shift)"
                shifts.append((test_id, agent, "position_closer", delta, desc, None))
                
                # South movement (major shift)
                test_id = f"pos_closer_macro_{agent}_south_{delta:.2f}deg"
                desc = f"{agent} moved {delta:.2f}° south (major shift)"
                shifts.append((test_id, agent, "position_closer", -delta, desc, None))
    
    # Lateral (crossing) position shifts with geometry-based caps
    # Merge/chase scenarios need tighter caps to avoid t=0 conflicts
    is_merge_chase = any(x in scenario_name for x in ["merge", "chase"])
    
    lateral_micro = [0.05, 0.1, 0.15]  # Small lateral deviations
    if is_merge_chase:
        lateral_macro = [0.2]  # Cap at 0.20° for merge/chase (geometry constraint)
    else:
        lateral_macro = [0.2, 0.3]  # Cross scenarios can handle more
    
    for agent in scenario_agents:
        for delta in lateral_micro + lateral_macro:
            range_type = "micro" if delta in lateral_micro else "macro"
            # East deviation
            test_id = f"pos_lateral_{range_type}_{agent}_east_{delta:.2f}deg"
            desc = f"Agent {agent} lateral {range_type}-shift: {delta:.2f}° east"
            shifts.append((test_id, agent, "position_lateral", delta, desc, None))
            
            # West deviation
            test_id = f"pos_lateral_{range_type}_{agent}_west_{delta:.2f}deg"
            desc = f"Agent {agent} lateral {range_type}-shift: {delta:.2f}° west"
            shifts.append((test_id, agent, "position_lateral", -delta, desc, None))
    
    # Heading deviations - creating converging/diverging paths
    heading_micro = [-10, -5, 5, 10]    # ±10 degrees
    heading_macro = [-30, -20, 20, 30]  # ±30 degrees
    
    for agent in scenario_agents:
        # Micro heading changes
        for delta in heading_micro:
            test_id = f"hdg_micro_{agent}_{delta:+d}deg"
            desc = f"Agent {agent} heading micro-shift: {delta:+d}°"
            shifts.append((test_id, agent, "heading", delta, desc, None))
        
        # Macro heading changes
        for delta in heading_macro:
            test_id = f"hdg_macro_{agent}_{delta:+d}deg"
            desc = f"Agent {agent} heading macro-shift: {delta:+d}°"
            shifts.append((test_id, agent, "heading", delta, desc, None))
    
    # Aircraft type variations - test different aircraft models
    aircraft_types = ["B737", "B747", "CRJ9"]  # Different from baseline A320
    
    # Individual aircraft type changes
    for agent in scenario_agents:
        for aircraft_type in aircraft_types:
            test_id = f"aircraft_{agent}_{aircraft_type}"
            desc = f"Agent {agent} aircraft type: {aircraft_type}"
            shifts.append((test_id, agent, "aircraft_type", 0.0, desc, aircraft_type))
    
    # All aircraft type changes (all agents at once)
    for aircraft_type in aircraft_types:
        test_id = f"aircraft_ALL_{aircraft_type}"
        desc = f"All agents aircraft type: {aircraft_type}"
        shifts.append((test_id, "ALL", "aircraft_type", 0.0, desc, aircraft_type))
    
    # Waypoint variations - test different destinations
    # Micro waypoint shifts (small deviations)
    waypoint_micro = [
        {"lat": 0.05, "lon": 0.0, "desc": "north_0.05deg"},
        {"lat": -0.05, "lon": 0.0, "desc": "south_0.05deg"},
        {"lat": 0.0, "lon": 0.05, "desc": "east_0.05deg"},
        {"lat": 0.0, "lon": -0.05, "desc": "west_0.05deg"}
    ]
    
    # Macro waypoint shifts (large deviations)
    waypoint_macro = [
        {"lat": 0.2, "lon": 0.0, "desc": "north_0.2deg"},
        {"lat": -0.2, "lon": 0.0, "desc": "south_0.2deg"},
        {"lat": 0.0, "lon": 0.2, "desc": "east_0.2deg"},
        {"lat": 0.0, "lon": -0.2, "desc": "west_0.2deg"},
        {"lat": 0.15, "lon": 0.15, "desc": "northeast_0.15deg"},
        {"lat": -0.15, "lon": -0.15, "desc": "southwest_0.15deg"}
    ]
    
    for agent in scenario_agents:
        # Micro waypoint changes
        for wp_shift in waypoint_micro:
            test_id = f"waypoint_micro_{agent}_{wp_shift['desc']}"
            desc = f"Agent {agent} waypoint micro-shift: {wp_shift['desc']}"
            wp_data = {"lat": wp_shift["lat"], "lon": wp_shift["lon"]}
            shifts.append((test_id, agent, "waypoint", 0.0, desc, wp_data))
        
        # Macro waypoint changes
        for wp_shift in waypoint_macro:
            test_id = f"waypoint_macro_{agent}_{wp_shift['desc']}"
            desc = f"Agent {agent} waypoint macro-shift: {wp_shift['desc']}"
            wp_data = {"lat": wp_shift["lat"], "lon": wp_shift["lon"]}
            shifts.append((test_id, agent, "waypoint", 0.0, desc, wp_data))
    
    return shifts


def _create_targeted_analysis(df: pd.DataFrame, analysis_dir: str, timestamp: str, scenario_agents: List[str]):
    """Create detailed analysis summaries for targeted shifts with agent-specific focus."""
    
    # Analysis by target agent (only for agents that exist in this scenario)
    for agent_id in scenario_agents:
        agent_df = df[df['target_agent'] == agent_id].copy()
        
        # Skip if no data for this agent in this scenario
        if agent_df.empty:
            continue
        
        # Statistics by shift type for this agent
        agent_stats = agent_df.groupby(['shift_type', 'shift_range']).agg({
            'tp': ['mean', 'std', 'sum'],
            'fp': ['mean', 'std', 'sum'], 
            'fn': ['mean', 'std', 'sum'],
            'tn': ['mean', 'std', 'sum'],
            'ghost_conflict': ['mean', 'std'],
            'missed_conflict': ['mean', 'std'],
            'num_los_events': ['mean', 'std', 'sum'],
            'total_los_duration': ['mean', 'std', 'sum'],
            'min_separation_nm': ['mean', 'std', 'min'],
            'total_path_length_nm': ['mean', 'std'],
            'flight_time_s': ['mean', 'std'],
            'path_efficiency': ['mean', 'std'],
            'waypoint_reached_ratio': ['mean', 'std'],
            'resolution_fail_rate': ['mean', 'std'],
            'los_failure_rate': ['mean', 'std'],
            'oscillation_rate': ['mean', 'std'],
            # New high-value metrics
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'f1_score': ['mean', 'std'],
            'alert_duty_cycle': ['mean', 'std'],
            'alerts_per_min': ['mean', 'std'],
            'total_alert_time_s': ['mean', 'std'],
            'avg_lead_time_s': ['mean', 'std'],
            # NEW: Extra path metrics
            'total_extra_path_nm': ['mean', 'std'],
            'avg_extra_path_nm': ['mean', 'std'],
            'avg_extra_path_ratio': ['mean', 'std'],
            # NEW: Intervention metrics
            'num_interventions': ['mean', 'std', 'sum'],
            'num_interventions_matched': ['mean', 'std', 'sum'],
            'num_interventions_false': ['mean', 'std', 'sum']
        }).round(4)
        
        # Flatten column names
        agent_stats.columns = [f"{col[0]}_{col[1]}" for col in agent_stats.columns]
        agent_stats = agent_stats.reset_index()
        
        # Save agent-specific analysis
        agent_analysis_path = os.path.join(analysis_dir, f"{agent_id}_analysis.csv")
        agent_stats.to_csv(agent_analysis_path, index=False)
        
        # Save detailed episode data for this agent
        agent_episodes_path = os.path.join(analysis_dir, f"{agent_id}_episodes.csv")
        agent_df.to_csv(agent_episodes_path, index=False)
    
    # Analysis by shift type and range (micro vs macro)
    type_range_stats = df.groupby(['shift_type', 'shift_range']).agg({
        'tp': 'sum',
        'fp': 'sum',
        'fn': 'sum', 
        'tn': 'sum',
        'ghost_conflict': 'mean',
        'missed_conflict': 'mean',
        'num_los_events': 'sum',
        'total_los_duration': 'sum',
        'min_separation_nm': 'min',
        'total_path_length_nm': 'mean',
        'flight_time_s': 'mean',
        'path_efficiency': 'mean',
        'waypoint_reached_ratio': 'mean',
        'resolution_fail_rate': 'mean',
        'los_failure_rate': 'mean',
        'oscillation_rate': 'mean',
        # New high-value metrics
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'alert_duty_cycle': 'mean',
        'alerts_per_min': 'mean',
        'total_alert_time_s': 'mean',
        'avg_lead_time_s': 'mean',
        # NEW: Extra path metrics
        'total_extra_path_nm': 'mean',
        'avg_extra_path_nm': 'mean',
        'avg_extra_path_ratio': 'mean',
        # NEW: Intervention metrics
        'num_interventions': 'sum',
        'num_interventions_matched': 'sum',
        'num_interventions_false': 'sum'
    }).round(4).reset_index()
    
    # Calculate additional derived metrics if they weren't already computed in groupby
    if 'precision' not in type_range_stats.columns:
        type_range_stats['precision'] = type_range_stats['tp'] / (type_range_stats['tp'] + type_range_stats['fp'] + 1e-10)
        type_range_stats['recall'] = type_range_stats['tp'] / (type_range_stats['tp'] + type_range_stats['fn'] + 1e-10)
        type_range_stats['f1_score'] = 2 * (type_range_stats['precision'] * type_range_stats['recall']) / (type_range_stats['precision'] + type_range_stats['recall'] + 1e-10)
    
    # Always calculate accuracy from base metrics
    type_range_stats['accuracy'] = (type_range_stats['tp'] + type_range_stats['tn']) / (type_range_stats['tp'] + type_range_stats['tn'] + type_range_stats['fp'] + type_range_stats['fn'] + 1e-10)
    
    type_range_path = os.path.join(analysis_dir, "type_range_analysis.csv")
    type_range_stats.to_csv(type_range_path, index=False)
    
    # Conflict inducing analysis - which shifts cause most conflicts
    conflict_analysis = df.groupby('test_id').agg({
        'num_los_events': 'sum',
        'total_los_duration': 'sum',
        'min_separation_nm': 'min',
        'resolution_fail_rate': 'mean',
        'los_failure_rate': 'mean',
        'ghost_conflict': 'mean',
        'missed_conflict': 'mean',
        # Add new high-value metrics to conflict analysis
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'alert_duty_cycle': 'mean',
        'alerts_per_min': 'mean',
        'avg_lead_time_s': 'mean',
        # NEW: Extra path and intervention metrics
        'total_extra_path_nm': 'mean',
        'avg_extra_path_nm': 'mean',
        'avg_extra_path_ratio': 'mean',
        'num_interventions': 'sum',
        'num_interventions_matched': 'sum',
        'num_interventions_false': 'sum'
    }).round(4).reset_index()
    
    # Sort by conflict indicators
    conflict_analysis = conflict_analysis.sort_values(['num_los_events', 'total_los_duration'], ascending=False)
    
    conflict_analysis_path = os.path.join(analysis_dir, "conflict_inducing_shifts.csv")
    conflict_analysis.to_csv(conflict_analysis_path, index=False)
    
    # Overall summary
    overall_summary = {
        'total_tests': int(len(df)),
        'total_episodes': int(len(df)),
        'agents_tested': int(df['target_agent'].nunique()),
        'shift_types': int(df['shift_type'].nunique()),
        'total_conflicts': int(df['num_los_events'].sum()),
        'avg_conflicts_per_test': float(df['num_los_events'].mean()),
        'most_conflict_prone_agent': str(df.groupby('target_agent')['num_los_events'].sum().idxmax()),
        'most_conflict_prone_shift_type': str(df.groupby('shift_type')['num_los_events'].sum().idxmax()),
        'macro_vs_micro_conflicts': {k: int(v) for k, v in df.groupby('shift_range')['num_los_events'].sum().to_dict().items()},
        
        # High-value metrics summary
        'detection_performance': {
            'avg_precision': float(df['precision'].mean()),
            'avg_recall': float(df['recall'].mean()),
            'avg_f1_score': float(df['f1_score'].mean()),
            'precision_std': float(df['precision'].std()),
            'recall_std': float(df['recall'].std()),
            'f1_std': float(df['f1_score'].std())
        },
        'alert_burden': {
            'avg_alert_duty_cycle': float(df['alert_duty_cycle'].mean()),
            'avg_alerts_per_min': float(df['alerts_per_min'].mean()),
            'total_alert_time_s': float(df['total_alert_time_s'].sum()),
            'duty_cycle_std': float(df['alert_duty_cycle'].std())
        },
        'timing_performance': {
            'avg_lead_time_s': float(df['avg_lead_time_s'].mean()),
            'lead_time_std': float(df['avg_lead_time_s'].std()),
            'early_alerts_fraction': float((df['avg_lead_time_s'] < 0).mean()),
            'late_alerts_fraction': float((df['avg_lead_time_s'] > 0).mean())
        }
    }
    
    summary_path = os.path.join(analysis_dir, "targeted_shift_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)


def _save_targeted_run_metadata(results_dir: str, repo_root: str, checkpoint_path: str, 
                               episodes_per_shift: int, timestamp: str, total_shifts: int,
                               scenario_agents: List[str], scenario_name: str = "head_on"):
    """Save metadata about the targeted shift testing run."""
    
    metadata = {
        "timestamp": timestamp,
        "run_date": datetime.now().isoformat(),
        "test_type": "intrashift_representative_agents",
        "checkpoint_path": checkpoint_path,
        "episodes_per_shift": episodes_per_shift,
        "scenario": f"{scenario_name}.json",
        "representative_agents": scenario_agents,
        "deduplication_method": "geometric_role (heading_class × lane_side)",
        "shift_types": ["speed", "position_closer", "position_lateral", "heading", "aircraft_type", "waypoint"],
        "shift_ranges": ["micro", "macro"],
        "speed_shifts_kt": {"micro": "±5 to ±10", "macro": "±20 to ±30"},
        "position_shifts_deg": {"micro": "±0.05 to ±0.15", "macro": "±0.2 to ±0.4"},
        "heading_shifts_deg": {"micro": "±5 to ±10", "macro": "±20 to ±30"},
        "aircraft_types": ["B737", "B747", "CRJ9"],
        "waypoint_shifts_deg": {"micro": "±0.05", "macro": "±0.15 to ±0.2"},
        "total_shift_configurations": int(total_shifts),
        "total_episodes": int(episodes_per_shift * total_shifts),
        "separation_threshold_nm": 5.0,
        "conflict_detection_focus": "Single agent modifications to induce conflicts",
        "hallucination_detector": {
            "horizon_s": 120.0,
            "resolution_window_s": 60.0,
            "action_period_s": 10.0
        }
    }
    
    metadata_path = os.path.join(results_dir, "targeted_run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    # Create comprehensive README
    readme_content = f"""# Targeted Shift Analysis Results - {timestamp}

## Overview
This directory contains comprehensive **targeted shift testing** results where only **one agent at a time** is modified to create conflict-prone scenarios and test model robustness.

## Test Philosophy
Unlike unison shifts (where all agents are modified equally), targeted shifts:
- Modify only ONE agent per test case
- Use micro→macro range variations to identify failure modes  
- Create conflict-inducing scenarios (agents moved closer)
- Test edge cases beyond training envelope

## Directory Structure
- `shifts/`: Episode-wise data organized by test configuration
  - `baseline/`: Baseline (no-shift) trajectory data
  - `speed_micro_*_*kt/`: Micro speed variations (±5-10 kt)
  - `speed_macro_*_*kt/`: Macro speed variations (±20-30 kt)
  - `pos_closer_*_*deg/`: Position shifts moving agents closer
  - `pos_lateral_*_*deg/`: Lateral position deviations
  - `hdg_micro_*_*deg/`: Micro heading variations (±5-10°)
  - `hdg_macro_*_*deg/`: Macro heading variations (±20-30°)
- `analysis/`: Aggregated analysis summaries
  - `A1_analysis.csv`, `A2_analysis.csv`, `A3_analysis.csv`: Agent-specific stats
  - `type_range_analysis.csv`: Analysis by shift type and range
  - `conflict_inducing_shifts.csv`: Most conflict-prone configurations
  - `targeted_shift_summary.json`: Overall summary statistics
- `viz/`: Interactive visualizations and maps
  - `comparison_baseline_vs_*.html`: Geographic trajectory comparison maps
  - `comparison_maps_index.html`: Navigation index for all comparison maps
  - `trajectory_analysis_dashboard.html`: Interactive Plotly-based trajectory analysis
  - `trajectory_analysis_*.html`: Shift-type specific trajectory plots  
  - `trajectory_statistics.json`: Statistical analysis of trajectory deviations
  - `map.html`, `time_series.html`: Episode-level visualizations
  - `degradation_curves.png`, `heatmap_*.png`: Run-level analysis plots

## Key Metrics (Enhanced for Conflict Detection)
- **Safety**: LoS count, PH5_time_frac, min_CPA_nm
- **Detection**: TP/FP/FN/TN for conflict prediction accuracy
- **Precision/Recall/F1**: Event-level performance metrics (post IoU matching)
- **Alert Burden**: duty_cycle, alerts_per_min, total_alert_time_s (operator load)
- **Lead Time**: avg_lead_time_s (negative=early, positive=late alerts)
- **Resolution**: TP_res/FP_res/FN_res within 60s after alert
- **Efficiency**: Path length, flight time, waypoint completion  
- **Stability**: Action oscillation rate

Note: Episode CSVs now use proper step-level aggregation (groupby step_idx.max())
to prevent double-counting timesteps across agent rows.

## Test Configuration Summary
- **Target Agents**: A1, A2, A3 (middle agent A2 expected most conflict-prone)
- **Shift Types**: Speed, Position (closer/lateral), Heading
- **Range Types**: Micro (small perturbations), Macro (large deviations)
- **Total Configurations**: {total_shifts}
- **Episodes per Configuration**: {episodes_per_shift}
- **Total Episodes**: {episodes_per_shift * total_shifts}

## Expected Outcomes
- A2 (middle agent) shifts should generate most conflicts
- Macro shifts should exceed training envelope and reveal failure modes
- Position shifts (moving closer) should increase conflict probability
- Speed and heading deviations should test resolution capabilities

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = os.path.join(results_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)


def make_env(env_config: Dict[str, Any]):
    # Robust path correction for Ray workers
    # Check if the corrupted path is present and fix it
    if "scenario_path" in env_config and "077" in str(env_config["scenario_path"]):
        # Extract filename from corrupted path
        filename = os.path.basename(env_config["scenario_path"])
        
        # Find correct project root
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        search_dir = current_file_dir
        project_root = None
        
        for _ in range(5):
            if (os.path.exists(os.path.join(search_dir, "scenarios")) and 
                os.path.exists(os.path.join(search_dir, "atc_cli.py"))):
                project_root = search_dir
                break
            search_dir = os.path.dirname(search_dir)
        
        if project_root:
            corrected_path = os.path.join(project_root, "scenarios", filename)
            corrected_path = os.path.abspath(corrected_path)
            
            if os.path.exists(corrected_path):
                env_config["scenario_path"] = corrected_path
    
    # Also fix results_dir if it contains corrupted path
    if "results_dir" in env_config and "077" in str(env_config["results_dir"]):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        search_dir = current_file_dir
        project_root = None
        
        for _ in range(5):
            if (os.path.exists(os.path.join(search_dir, "scenarios")) and 
                os.path.exists(os.path.join(search_dir, "atc_cli.py"))):
                project_root = search_dir
                break
            search_dir = os.path.dirname(search_dir)
        
        if project_root:
            # Extract relative path from corrupted path
            corrupted_results = env_config["results_dir"]
            if "results" in corrupted_results:
                # Extract the results directory part
                results_part = corrupted_results.split("results")[-1]
                if results_part.startswith("\\") or results_part.startswith("/"):
                    results_part = results_part[1:]
                
                corrected_results = os.path.join(project_root, "results", results_part)
                env_config["results_dir"] = os.path.abspath(corrected_results)
    
    # Ensure scenario_path is absolute
    if "scenario_path" in env_config and not os.path.isabs(env_config["scenario_path"]):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        search_dir = current_file_dir
        project_root = None
        
        for _ in range(5):
            if (os.path.exists(os.path.join(search_dir, "scenarios")) and 
                os.path.exists(os.path.join(search_dir, "atc_cli.py"))):
                project_root = search_dir
                break
            search_dir = os.path.dirname(search_dir)
        
        if project_root:
            rel_path = env_config["scenario_path"]
            if rel_path.startswith("scenarios/") or rel_path.startswith("scenarios\\"):
                env_config["scenario_path"] = os.path.join(project_root, rel_path)
            else:
                env_config["scenario_path"] = os.path.join(project_root, "scenarios", rel_path)
            
            env_config["scenario_path"] = os.path.abspath(env_config["scenario_path"])
    
    # Determine which environment to use based on presence of scenario_path
    if "scenario_path" in env_config and env_config["scenario_path"]:
        # Use frozen scenario environment
        return ParallelPettingZooEnv(MARLCollisionEnv(env_config))
    else:
        # Use generic environment (for generic models testing on frozen scenarios)
        # Import here to avoid circular dependency
        from src.environment.marl_collision_env_generic import MARLCollisionEnvGeneric
        return ParallelPettingZooEnv(MARLCollisionEnvGeneric(env_config))


def run_intrashift_grid(repo_root: str,
                        algo_class,
                        checkpoint_path: str,
                        scenario_name: str = "head_on",
                        episodes_per_shift: int = 5,
                        seeds: Optional[List[int]] = None,
                        generate_viz: bool = False,
                        outdir: Optional[str] = None) -> str:
    """
    Run intrashift testing where only one agent is modified per test case within a scenario.
    
    Args:
        repo_root: Root directory of the project
        algo_class: RLLib algorithm class (PPO, SAC, etc.)
        checkpoint_path: Path to the trained model checkpoint
        scenario_name: Name of the scenario file (without .json extension)
        episodes_per_shift: Number of episodes to run per shift configuration
        seeds: List of random seeds to use for episodes
        generate_viz: Whether to generate visualization artifacts
    
    Returns:
        Path to the main summary CSV file
    """
    scenario_path = os.path.join(repo_root, "scenarios", f"{scenario_name}.json")
    
    # Ensure absolute path to avoid Ray worker directory issues
    scenario_path = os.path.abspath(scenario_path)
    
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
    
    # Load scenario to get representative agent list (deduplicated by geometric role)
    representative_agents = _select_representative_agents(scenario_path, scenario_name)
    
    LOGGER.info(f"Selected {len(representative_agents)} representative agents for {scenario_name}: {representative_agents}")
    LOGGER.info(f"(Deduplication based on geometric role: heading class × lane side)")
    
    # Create results directory (custom or timestamped)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(repo_root, "results")
    if outdir:
        results_dir = os.path.join(base_results_dir, outdir)
    else:
        results_dir = os.path.join(base_results_dir, f"intrashift_analysis_{scenario_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for organized storage
    shifts_dir = os.path.join(results_dir, "shifts")
    analysis_dir = os.path.join(results_dir, "analysis")
    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(shifts_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    env_name = "marl_collision_env_v0"
    register_env(env_name, lambda cfg: make_env(cfg))

    # Base env config with hallucination detection enabled
    # CRITICAL: Match training configuration exactly to avoid observation space mismatch
    base_env_config = {
        "scenario_path": scenario_path,
        "action_delay_steps": 0,
        "max_episode_steps": 100,
        "separation_nm": 5.0,
        "log_trajectories": True,
        "results_dir": results_dir,
        "seed": 42,
        "enable_hallucination_detection": True,  # Enable real-time hallucination detection

        # MATCH TRAINING CONFIG: Enable relative observations (no raw lat/lon)
        "neighbor_topk": 3,
        
        # MATCH TRAINING CONFIG: Collision detection settings
        "collision_nm": 3.0,

        # MATCH TRAINING CONFIG: Team shaping knobs (PBRS coordination rewards)
        "team_coordination_weight": 0.2,
        "team_gamma": 0.99,
        "team_share_mode": "responsibility",
        "team_ema": 0.001,
        "team_cap": 0.005,
        "team_anneal": 1.0,
        "team_neighbor_threshold_km": 10.0,
        
        # MATCH TRAINING CONFIG: Individual reward components
        "drift_penalty_per_sec": -0.1,
        "progress_reward_per_km": 0.02,
        "backtrack_penalty_per_km": -0.1,
        "time_penalty_per_sec": -0.0005,
        "reach_reward": 10.0,
        "intrusion_penalty": -50.0,
        "conflict_dwell_penalty_per_sec": -0.1,
    }

    # CRITICAL FIX: Properly manage Ray initialization
    # Check if Ray is already initialized, if not initialize it
    try:
        if not ray.is_initialized():
            LOGGER.info("Initializing Ray for shift testing...")
            ray.init(ignore_reinit_error=True, log_to_driver=False)
        else:
            LOGGER.info("Ray already initialized, continuing...")
    except Exception as e:
        LOGGER.warning(f"Ray initialization warning: {e}")
        # Try to shutdown and reinitialize
        try:
            ray.shutdown()
            ray.init(ignore_reinit_error=True, log_to_driver=False)
            LOGGER.info("Ray reinitialized successfully")
        except Exception as e2:
            LOGGER.error(f"Failed to reinitialize Ray: {e2}")
            raise

    # CRITICAL FIX: Use Algorithm.from_checkpoint() to restore the exact trained configuration
    # This preserves the original multi-agent policy mapping (shared_policy)
    from ray.rllib.algorithms.algorithm import Algorithm
    
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    # Verify we have the expected trained policy by trying to get it directly
    policy_id = "shared_policy"
    try:
        # Test if shared_policy exists by trying to access it
        test_policy = algo.get_policy(policy_id)
        if test_policy is None:
            raise ValueError("shared_policy is None")
        LOGGER.info(f"✓ Successfully loaded trained policy: {policy_id}")
    except Exception as e:
        # Fallback: try to find available policies
        try:
            if hasattr(algo, 'workers') and algo.workers and hasattr(algo.workers, 'local_worker'):
                worker = algo.workers.local_worker()
                # Try to get policy map from worker
                available_policies = ["policy_map_unavailable"]
            else:
                available_policies = ["unknown"]
        except:
            available_policies = ["unknown"]
            
        raise RuntimeError(f"Could not load 'shared_policy' from checkpoint. "
                          f"Available policies: {available_policies}. "
                          f"Error: {e}. "
                          f"This indicates the checkpoint wasn't trained with multi-agent configuration.")
    
    LOGGER.info(f"✓ Using trained policy: {policy_id}")

    # Generate targeted shift configurations using representative agents
    targeted_shifts = create_conflict_inducing_shifts(representative_agents, scenario_name, scenario_path)
    # Use paired seeds (same seeds for baseline & shifted scenarios for direct comparison)
    if seeds is None:
        import random
        random.seed(42)  # Reproducible but varied
        seeds = [random.randint(1, 10000) for _ in range(episodes_per_shift)]
        LOGGER.info(f"Generated paired seeds for reproducible baseline vs shift comparison: {seeds}")
    else:
        # Handle seed count mismatch by cycling or truncating
        if len(seeds) != episodes_per_shift:
            if len(seeds) < episodes_per_shift:
                # Cycle seeds if we need more
                seeds = (seeds * ((episodes_per_shift // len(seeds)) + 1))[:episodes_per_shift]
                LOGGER.info(f"Cycled seeds to match episodes_per_shift: {seeds}")
            else:
                # Truncate seeds if we have too many
                seeds = seeds[:episodes_per_shift]
                LOGGER.info(f"Truncated seeds to match episodes_per_shift: {seeds}")
        else:
            LOGGER.info(f"Using provided paired seeds: {seeds}")

    summary_rows = []

    LOGGER.info(f"Running {len(targeted_shifts)} targeted shift configurations with {episodes_per_shift} episodes each")
    
    # Track skipped shifts due to t=0 conflicts
    skipped_count = 0
    total_shifts_attempted = 0

    for test_id, agent_id, shift_type, shift_value, description, shift_data in targeted_shifts:
        for ep_idx, seed in enumerate(seeds):
            total_shifts_attempted += 1
            
            # Create per-shift directory structure
            ep_tag = f"{test_id}_ep{ep_idx}"
            shift_dir = os.path.join(shifts_dir, test_id)
            os.makedirs(shift_dir, exist_ok=True)
            
            # Update env config with shift-specific results directory
            env_config = base_env_config.copy()
            env_config["results_dir"] = shift_dir
            env_config["seed"] = seed
            env_config["episode_tag"] = ep_tag  # Pass episode tag for clearer CSV naming
            
            # Get all scenario agents for shift config (not just representatives)
            with open(scenario_path, "r") as f:
                scenario_data = json.load(f)
            all_scenario_agents = [agent["id"] for agent in scenario_data["agents"]]
            
            # Create targeted shift configuration
            shift_config = create_targeted_shift(agent_id, shift_type, shift_value, all_scenario_agents, shift_data)
            
            # CRITICAL: Apply t=0 conflict guard (5 NM separation threshold)
            is_safe, min_sep_nm = _check_start_separation(scenario_path, shift_config, separation_threshold_nm=5.0)
            
            if not is_safe:
                LOGGER.warning(f"Skipping {ep_tag}: t=0 separation too small ({min_sep_nm:.2f} NM < 5.0 NM threshold)")
                skipped_count += 1
                continue
            
            # Create environmental shift configuration (wind + noise) based on shift type
            env_shift = _create_wind_config_for_shift(test_id, shift_type, shift_value)
            
            # Build env with proper wrapper
            env = MARLCollisionEnv(env_config)
            env_pz = ParallelPettingZooEnv(env)
            
            try:
                obs, _ = env_pz.reset(options={"targeted_shift": shift_config, "env_shift": env_shift})
                if not obs:
                    LOGGER.warning(f"Empty observation on reset for {ep_tag}")
                    continue
            except Exception as e:
                LOGGER.error(f"Failed to reset environment for {ep_tag}: {e}")
                continue
            
            # Storage with proper initialization
            traj = {
                "timestamps": [],
                "positions": [],
                "actions": [],
                "agents": {aid: {"headings": [], "speeds": []} for aid in env.possible_agents},
                "scenario_metadata": {
                    "test_id": test_id,
                    "target_agent": agent_id,
                    "shift_type": shift_type,
                    "shift_value": shift_value,
                    "shift_range": "micro" if abs(shift_value) <= 15 else "macro",  # General threshold
                    "episode_id": ep_idx,
                    "seed": seed,  # Add seed for reproducibility
                    "description": description
                }
            }
            
            # Capture initial state (t=0 snapshot)
            t_s = float(bs.sim.simt)
            pos0 = {}
            for aid in env.possible_agents:
                idx = bs.traf.id2idx(aid)
                if isinstance(idx, list):
                    idx = idx[0] if idx else -1
                
                if idx >= 0 and idx < len(bs.traf.lat):
                    pos0[aid] = (float(bs.traf.lat[idx]), float(bs.traf.lon[idx]))
                    heading = getattr(bs.traf, 'hdg', getattr(bs.traf, 'heading', [0.0] * len(bs.traf.lat)))
                    # CRITICAL: Use GS (ground speed) first to show wind effects, fallback to TAS
                    speed = getattr(bs.traf, 'gs', getattr(bs.traf, 'tas', [250.0] * len(bs.traf.lat)))
                    traj["agents"][aid]["headings"].append(float(heading[idx]))
                    traj["agents"][aid]["speeds"].append(float(speed[idx] * 1.94384))  # m/s to knots
                else:
                    pos0[aid] = (52.0, 4.0)
                    traj["agents"][aid]["headings"].append(0.0)
                    traj["agents"][aid]["speeds"].append(250.0)
            
            traj["timestamps"].append(t_s)
            traj["positions"].append(pos0)
            traj["actions"].append({aid: [0.0, 0.0] for aid in env.possible_agents})

            # Run episode
            step_count = 0
            while True:
                # Policy actions for all agents using correct policy ID
                act = {}
                for aid in env.agents:
                    if aid in obs:
                        a = algo.compute_single_action(obs[aid], explore=True, policy_id=policy_id)
                        act[aid] = a
                        
                        # DEBUG: Log first few actions to verify magnitude (only for baseline to reduce noise)
                        if step_count < 3 and test_id == "baseline":
                            a_array = np.asarray(a, dtype=np.float32)
                            scaled_hdg = float(a_array[0]) * 18.0  # D_HEADING scaling
                            scaled_spd = float(a_array[1]) * 10.0  # D_VELOCITY scaling
                            LOGGER.info(f"  {aid} raw action: [{a_array[0]:.6f}, {a_array[1]:.6f}] -> [{scaled_hdg:.3f}°, {scaled_spd:.3f} kt]")

                    else:
                        act[aid] = [0.0, 0.0]
                
                step_count += 1

                next_obs, rewards, terminations, truncations, infos = env_pz.step(act)

                # Log trajectory data using BlueSky's clock
                t_s = float(bs.sim.simt)
                traj["timestamps"].append(t_s)
                
                pos_step = {}
                for aid in env.possible_agents:
                    idx = bs.traf.id2idx(aid)
                    if isinstance(idx, list):
                        idx = idx[0] if idx else -1
                    
                    if idx >= 0 and idx < len(bs.traf.lat):
                        pos_step[aid] = (float(bs.traf.lat[idx]), float(bs.traf.lon[idx]))
                        heading = getattr(bs.traf, 'hdg', getattr(bs.traf, 'heading', [0.0] * len(bs.traf.lat)))
                        # CRITICAL: Use GS (ground speed) first to show wind effects, fallback to TAS
                        speed = getattr(bs.traf, 'gs', getattr(bs.traf, 'tas', [250.0] * len(bs.traf.lat)))
                        traj["agents"][aid]["headings"].append(float(heading[idx]))
                        traj["agents"][aid]["speeds"].append(float(speed[idx] * 1.94384))
                    else:
                        if traj["positions"]:
                            pos_step[aid] = traj["positions"][-1].get(aid, (52.0, 4.0))
                        else:
                            pos_step[aid] = (52.0, 4.0)
                        traj["agents"][aid]["headings"].append(0.0)
                        traj["agents"][aid]["speeds"].append(250.0)
                
                traj["positions"].append(pos_step)
                traj["actions"].append({aid: np.asarray(act.get(aid, [0.0, 0.0]), dtype=float).tolist() for aid in env.possible_agents})

                obs = next_obs
                done = (terminations and all(terminations.values())) or (truncations and all(truncations.values()))
                if done:
                    break

            # Save trajectory JSON file
            json_path = os.path.join(shift_dir, f"trajectory_{ep_tag}.json")
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(traj, f)
                
            # Find the rich trajectory CSV file generated by the environment (now includes hallucination data)
            # Look for episode-specific CSV file first, then fallback to general pattern
            episode_specific_csv = f"traj_{ep_tag}.csv"
            if os.path.exists(os.path.join(shift_dir, episode_specific_csv)):
                csv_path = os.path.join(shift_dir, episode_specific_csv)
            else:
                # Fallback: look for any CSV files but warn about potential issue
                rich_csv_files = [f for f in os.listdir(shift_dir) if 
                                 (f.startswith('traj_ep_') or f.startswith('traj_')) and f.endswith('.csv')]
                if rich_csv_files:
                    # Sort by modification time to get the most recent
                    rich_csv_files.sort(key=lambda f: os.path.getmtime(os.path.join(shift_dir, f)), reverse=True)
                    csv_path = os.path.join(shift_dir, rich_csv_files[0])
                    LOGGER.warning(f"Episode-specific CSV not found, using: {rich_csv_files[0]}. This may cause identical results across episodes.")
                else:
                    csv_path = None

            # Compute episode-level hallucination metrics from the rich CSV if available
            if csv_path:
                try:
                    df = pd.read_csv(csv_path)
                    
                    # CRITICAL FIX: Collapse per-step series across agents using groupby(step_idx).max()
                    # This prevents double-counting each timestep (once per agent row)
                    step_collapsed = df.groupby('step_idx').agg({
                        'tp': 'max',  # Use max because tp/fp/fn/tn are step-level flags (0 or 1)
                        'fp': 'max',
                        'fn': 'max', 
                        'tn': 'max',
                        'gt_conflict': 'max',  # Ground truth conflict at each step
                        'predicted_alert': 'max',  # Predicted alert at each step
                        'conflict_flag': 'max',  # Conflict flag at each step
                        'min_separation_nm': 'min',  # Minimum separation across all agent pairs
                        'sim_time_s': 'mean'  # Average simulation time (should be consistent)
                    }).reset_index()
                    
                    # Calculate episode-level metrics from step-collapsed data
                    total_steps = len(step_collapsed)
                    # Calculate actual flight time as end_time - start_time
                    if not step_collapsed.empty:
                        start_time_s = step_collapsed['sim_time_s'].min()
                        end_time_s = step_collapsed['sim_time_s'].max()
                        episode_time_s = end_time_s - start_time_s
                    else:
                        episode_time_s = 0.0
                    
                    # Basic confusion matrix metrics (step-level aggregation)
                    tp_sum = int(step_collapsed['tp'].sum())
                    fp_sum = int(step_collapsed['fp'].sum())
                    fn_sum = int(step_collapsed['fn'].sum())
                    tn_sum = int(step_collapsed['tn'].sum())
                    
                    # Alert duty cycle: fraction of time an alert is active
                    alert_duty_cycle = float(step_collapsed['predicted_alert'].mean()) if total_steps > 0 else 0.0
                    
                    # Alert burden metrics: operator load
                    total_alert_time_s = float(step_collapsed['predicted_alert'].sum() * 10.0)  # 10s per step
                    alerts_per_min = (total_alert_time_s / max(1.0, episode_time_s / 60.0)) if episode_time_s > 0 else 0.0
                    
                    # Event-level precision, recall, F1 (post IoU matching)
                    # Handle cases where the model generates no alerts (all predictions are 0)
                    if tp_sum + fp_sum == 0:
                        # No alerts generated - precision undefined, but we'll use 0
                        precision = 0.0
                    else:
                        precision = tp_sum / (tp_sum + fp_sum)
                    
                    if tp_sum + fn_sum == 0:
                        # No actual conflicts - recall undefined, but we'll use 1.0 (perfect)
                        recall = 1.0
                    else:
                        recall = tp_sum / (tp_sum + fn_sum)
                    
                    if precision + recall == 0:
                        f1_score = 0.0
                    else:
                        f1_score = 2 * precision * recall / (precision + recall)
                    
                    # Lead time analysis (negative = early alerts, positive = late alerts)
                    # Calculate avg lead time from first alert to first conflict for each episode
                    lead_times = []
                    
                    # Find alert and conflict transitions  
                    gt_flags = step_collapsed['gt_conflict'].values
                    alert_flags = step_collapsed['predicted_alert'].values
                    
                    # Find first alert and first conflict  
                    first_alert_idx = None
                    first_conflict_idx = None
                    
                    for i in range(len(alert_flags)):
                        if alert_flags[i] == 1 and first_alert_idx is None:
                            first_alert_idx = i
                        if gt_flags[i] == 1 and first_conflict_idx is None:
                            first_conflict_idx = i
                        if first_alert_idx is not None and first_conflict_idx is not None:
                            break
                    
                    if first_alert_idx is not None and first_conflict_idx is not None:
                        # Lead time in seconds (10s per step)
                        lead_time_s = (first_alert_idx - first_conflict_idx) * 10.0
                        lead_times.append(lead_time_s)
                    elif first_conflict_idx is not None and first_alert_idx is None:
                        # Conflict occurred but no alert - infinite late response
                        lead_times.append(float('inf'))
                    
                    avg_lead_time_s = float(np.mean([t for t in lead_times if t != float('inf')])) if lead_times and any(t != float('inf') for t in lead_times) else 0.0
                    
                    # FIXED: Compute oscillation_rate, resolution_fail_rate, and los_failure_rate from trajectory using HallucinationDetector
                    # These metrics are NOT in the CSV columns - they must be computed from the trajectory
                    oscillation_rate = 0.0
                    resolution_fail_rate = 0.0
                    los_failure_rate = 0.0
                    
                    try:
                        # Load trajectory JSON from same directory
                        csv_dir = os.path.dirname(csv_path)
                        csv_base = os.path.basename(csv_path).replace('traj_', 'trajectory_').replace('.csv', '.json')
                        json_path = os.path.join(csv_dir, csv_base)
                        
                        if os.path.exists(json_path):
                            fallback_metrics = _compute_metrics_from_trajectory_json(json_path, sep_nm=5.0)
                            
                            # Extract resolution_fail_rate, los_failure_rate, and oscillation_rate from detector (preserve NaN)
                            res_val = fallback_metrics.get("resolution_fail_rate", 0.0)
                            resolution_fail_rate = float(res_val) if not (isinstance(res_val, float) and np.isnan(res_val)) else float('nan')
                            
                            los_val = fallback_metrics.get("los_failure_rate", 0.0)
                            los_failure_rate = float(los_val) if not (isinstance(los_val, float) and np.isnan(los_val)) else float('nan')
                            
                            osc_val = fallback_metrics.get("oscillation_rate", 0.0)
                            oscillation_rate = float(osc_val) if not (isinstance(osc_val, float) and np.isnan(osc_val)) else float('nan')
                    except Exception as e:
                        LOGGER.warning(f"Could not compute oscillation/resolution/los metrics from trajectory: {e}")
                        # Keep defaults of 0.0
                    
                    # Extra path metrics - calculate deviation from direct start->waypoint paths
                    # FIXED: Stop at waypoint_reached = 1
                    total_extra_path_nm = 0.0
                    avg_extra_path_nm = 0.0
                    avg_extra_path_ratio = 0.0
                    
                    try:
                        # Calculate path lengths from trajectory data in CSV
                        agent_ids = df['agent_id'].unique()
                        agent_extra_paths = []
                        
                        for aid in agent_ids:
                            agent_data = df[df['agent_id'] == aid].sort_values('step_idx')
                            if len(agent_data) < 2:
                                continue
                            
                            # FIXED: Truncate trajectory at waypoint_reached = 1
                            if 'waypoint_reached' in agent_data.columns:
                                reached_mask = agent_data['waypoint_reached'] == 1
                                if reached_mask.any():
                                    # Stop at first waypoint reached step (inclusive)
                                    waypoint_idx = agent_data[reached_mask].index[0]
                                    agent_data = agent_data.loc[:waypoint_idx]
                            
                            # Calculate actual path length only up to waypoint
                            positions = list(zip(agent_data['lat_deg'], agent_data['lon_deg']))
                            actual_path_nm = 0.0
                            R_nm = 3440.065  # Earth radius in nautical miles
                            
                            for i in range(1, len(positions)):
                                lat1, lon1 = positions[i-1]
                                lat2, lon2 = positions[i]
                                # Haversine formula for accurate distance
                                dlat = np.radians(lat2 - lat1)
                                dlon = np.radians(lon2 - lon1)
                                a = (np.sin(dlat/2.0) ** 2 +
                                     np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
                                     np.sin(dlon/2.0) ** 2)
                                c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
                                dist_nm = R_nm * c
                                actual_path_nm += dist_nm
                            
                            # FIXED: Use waypoint coordinates or reached position, not arbitrary end
                            # Priority 1: Use wp_lat/wp_lon from CSV
                            # Priority 2: Use position at end of truncated trajectory (where waypoint reached)
                            start_pos = positions[0]
                            
                            if 'wp_lat' in agent_data.columns and 'wp_lon' in agent_data.columns:
                                wp_lat, wp_lon = agent_data.iloc[0]['wp_lat'], agent_data.iloc[0]['wp_lon']
                                if not (pd.isna(wp_lat) or pd.isna(wp_lon)):
                                    # Use scenario waypoint
                                    end_pos = (wp_lat, wp_lon)
                                else:
                                    # Waypoint columns exist but NaN
                                    end_pos = positions[-1]
                            else:
                                # No waypoint columns - use end of truncated trajectory
                                end_pos = positions[-1]
                            
                            dlat = np.radians(end_pos[0] - start_pos[0])
                            dlon = np.radians(end_pos[1] - start_pos[1])
                            a = (np.sin(dlat/2.0) ** 2 +
                                 np.cos(np.radians(start_pos[0])) * np.cos(np.radians(end_pos[0])) *
                                 np.sin(dlon/2.0) ** 2)
                            c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
                            direct_nm = R_nm * c
                            
                            if direct_nm > 0.1:  # Only count if agent moved significantly
                                extra_nm = max(0.0, actual_path_nm - direct_nm)
                                agent_extra_paths.append((extra_nm, actual_path_nm, direct_nm))
                        
                        if agent_extra_paths:
                            total_extra_path_nm = float(sum(x[0] for x in agent_extra_paths))
                            avg_extra_path_nm = float(np.mean([x[0] for x in agent_extra_paths]))
                            # Calculate ratio as (extra / direct), average across agents
                            ratios = [x[0] / x[2] if x[2] > 0.1 else 0.0 for x in agent_extra_paths]
                            avg_extra_path_ratio = float(np.mean(ratios)) if ratios else 0.0
                    except Exception as e:
                        # Fallback if path calculation fails - but log the error for debugging
                        LOGGER.debug(f"Extra path calculation failed: {e}")
                        total_extra_path_nm = avg_extra_path_nm = avg_extra_path_ratio = 0.0
                    
                    # Intervention metrics - count alert windows/runs
                    try:
                        # Find alert runs (consecutive alert periods)
                        alert_runs = []
                        in_run = False
                        run_start = None
                        
                        for i, is_alert in enumerate(alert_flags):
                            if is_alert and not in_run:
                                # Start of new alert run
                                in_run = True
                                run_start = i
                            elif not is_alert and in_run:
                                # End of alert run
                                in_run = False
                                alert_runs.append((run_start, i-1))
                        
                        # Handle case where alert run continues to end
                        if in_run:
                            alert_runs.append((run_start, len(alert_flags)-1))
                        
                        num_interventions = len(alert_runs)
                        
                        # For matched/false interventions, we'd need IoU matching with ground truth runs
                        # For simplicity, use a basic heuristic: interventions that overlap with gt_conflict
                        num_interventions_matched = 0
                        for start, end in alert_runs:
                            # Check if this alert run overlaps with any ground truth conflict
                            gt_overlap = any(gt_flags[i] for i in range(start, end+1))
                            if gt_overlap:
                                num_interventions_matched += 1
                        
                        num_interventions_false = num_interventions - num_interventions_matched
                        
                    except Exception:
                        # Fallback if intervention calculation fails
                        num_interventions = num_interventions_matched = num_interventions_false = 0
                    
                    # Calculate additional missing columns from available data
                    # Flight time: total episode duration
                    flight_time_s = episode_time_s
                    
                    # Number of LoS events: count conflicts
                    num_los_events = int(step_collapsed['conflict_flag'].sum())
                    
                    # Total LoS duration: duration of conflicts (10s per step)
                    total_los_duration = float(num_los_events * 10.0)
                    
                    # Calculate path metrics from trajectory data
                    try:
                        # Calculate path length from position changes
                        path_lengths = []
                        for csv_agent_id in df['agent_id'].unique():
                            agent_data = df[df['agent_id'] == csv_agent_id].sort_values('step_idx')
                            if len(agent_data) > 1:
                                # Calculate distance between consecutive positions
                                lats = agent_data['lat_deg'].values
                                lons = agent_data['lon_deg'].values
                                total_dist = 0.0
                                for i in range(1, len(lats)):
                                    # Simple distance calculation (not exact but reasonable approximation)
                                    dlat = lats[i] - lats[i-1]
                                    dlon = lons[i] - lons[i-1]
                                    dist_deg = np.sqrt(dlat**2 + dlon**2)
                                    dist_nm = dist_deg * 60.0  # Rough conversion to nautical miles
                                    total_dist += dist_nm
                                path_lengths.append(total_dist)
                        
                        total_path_length_nm = float(np.mean(path_lengths)) if path_lengths else 0.0
                        
                        # Path efficiency: straight-line distance / actual path length
                        if total_path_length_nm > 0:
                            # Estimate straight-line distance (rough approximation)
                            straight_line_dist = 50.0  # Rough estimate for parallel scenario
                            path_efficiency = min(1.0, straight_line_dist / total_path_length_nm)
                        else:
                            path_efficiency = 1.0
                            
                    except Exception:
                        total_path_length_nm = 0.0
                        path_efficiency = 1.0
                    
                    # FIXED: Waypoint reached ratio: ratio of agents that reached their waypoints
                    try:
                        if 'waypoint_reached' in df.columns:
                            # Get unique agents and check max waypoint_reached for each
                            agents = df['agent_id'].unique()
                            agents_reached = 0
                            
                            for agent in agents:
                                agent_data = df[df['agent_id'] == agent]
                                max_waypoint_reached = agent_data['waypoint_reached'].max()
                                if max_waypoint_reached > 0:
                                    agents_reached += 1
                            
                            waypoint_reached_ratio = float(agents_reached) / len(agents) if len(agents) > 0 else 0.0
                        else:
                            waypoint_reached_ratio = 0.0
                    except Exception:
                        waypoint_reached_ratio = 0.0
                    
                    # NEW: Total reward calculation
                    try:
                        if 'reward' in df.columns:
                            reward_total = float(df['reward'].sum())
                        else:
                            reward_total = 0.0
                    except Exception:
                        reward_total = 0.0
                    
                    cm_summary = {
                        "tp": tp_sum,
                        "fp": fp_sum,
                        "fn": fn_sum,
                        "tn": tn_sum,
                        "min_separation_nm": float(step_collapsed['min_separation_nm'].min()) if not step_collapsed.empty else 200.0,
                        "num_conflict_steps": int(step_collapsed['conflict_flag'].sum()),
                        
                        # Previously missing columns - now calculated
                        "flight_time_s": flight_time_s,
                        "num_los_events": num_los_events,
                        "total_los_duration": total_los_duration,
                        "total_path_length_nm": total_path_length_nm,
                        "path_efficiency": path_efficiency,
                        "waypoint_reached_ratio": waypoint_reached_ratio,
                        
                        # NEW: Extra path metrics (deviation from direct paths)
                        "total_extra_path_nm": total_extra_path_nm,
                        "avg_extra_path_nm": avg_extra_path_nm,
                        "avg_extra_path_ratio": avg_extra_path_ratio,
                        
                        # NEW: Intervention count metrics
                        "num_interventions": num_interventions,
                        "num_interventions_matched": num_interventions_matched,
                        "num_interventions_false": num_interventions_false,
                        
                        # High-value metrics
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1_score,
                        "alert_duty_cycle": alert_duty_cycle,
                        "alerts_per_min": alerts_per_min,
                        "total_alert_time_s": total_alert_time_s,
                        "avg_lead_time_s": avg_lead_time_s,
                        
                        # Legacy metrics (calculated properly)
                        "ghost_conflict": fp_sum / max(1, total_steps) if total_steps > 0 else 0.0,  # False alert rate per step
                        "missed_conflict": fn_sum / max(1, tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0,  # Missed conflict rate
                        "resolution_fail_rate": resolution_fail_rate,  # Computed from trajectory JSON
                        "los_failure_rate": los_failure_rate,  # NEW: Computed from trajectory JSON
                        "oscillation_rate": oscillation_rate,  # Computed from trajectory JSON
                        "reward_total": reward_total,  # NEW: Total reward from trajectory
                    }
                except Exception as e:
                    LOGGER.warning(f"Failed to extract metrics from CSV {csv_path}: {e}")
                    
                    # FIXED: Fallback to computing from trajectory JSON using HallucinationDetector
                    json_path = os.path.join(shift_dir, f"trajectory_{ep_tag}.json")
                    fallback_metrics = _compute_metrics_from_trajectory_json(json_path, sep_nm=5.0)
                    
                    # Extract resolution_fail_rate, los_failure_rate, and oscillation_rate from detector (preserve NaN)
                    res_val = fallback_metrics.get("resolution_fail_rate", 0.0)
                    resolution_fail_rate = float(res_val) if not (isinstance(res_val, float) and np.isnan(res_val)) else float('nan')
                    
                    los_val = fallback_metrics.get("los_failure_rate", 0.0)
                    los_failure_rate = float(los_val) if not (isinstance(los_val, float) and np.isnan(los_val)) else float('nan')
                    
                    osc_val = fallback_metrics.get("oscillation_rate", 0.0)
                    oscillation_rate = float(osc_val) if not (isinstance(osc_val, float) and np.isnan(osc_val)) else float('nan')
                    
                    # Fallback metrics with new fields
                    cm_summary = {
                        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                        "min_separation_nm": 200.0, "num_conflict_steps": 0,
                        # Previously missing columns - now with defaults
                        "flight_time_s": 0.0,
                        "num_los_events": 0,
                        "total_los_duration": 0.0,
                        "total_path_length_nm": 0.0,
                        "path_efficiency": 1.0,
                        "waypoint_reached_ratio": 0.0,
                        # NEW: Extra path metrics with defaults
                        "total_extra_path_nm": 0.0,
                        "avg_extra_path_nm": 0.0,
                        "avg_extra_path_ratio": 0.0,
                        # NEW: Intervention metrics with defaults
                        "num_interventions": 0,
                        "num_interventions_matched": 0,
                        "num_interventions_false": 0,
                        # High-value metrics
                        "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                        "alert_duty_cycle": 0.0, "alerts_per_min": 0.0, "total_alert_time_s": 0.0,
                        "avg_lead_time_s": 0.0,
                        "ghost_conflict": 0.0, "missed_conflict": 1.0,  # If no data, assume all conflicts missed
                        "resolution_fail_rate": resolution_fail_rate,  # Computed from trajectory JSON or 0.0
                        "los_failure_rate": los_failure_rate,  # NEW: Computed from trajectory JSON or 0.0
                        "oscillation_rate": oscillation_rate,  # Computed from trajectory JSON or 0.0
                        "reward_total": 0.0,  # NEW: Total reward from trajectory
                    }
            else:
                # No CSV available - compute from trajectory JSON using HallucinationDetector
                LOGGER.info(f"No CSV found for {ep_tag}, computing from trajectory JSON")
                json_path = os.path.join(shift_dir, f"trajectory_{ep_tag}.json")
                fallback_metrics = _compute_metrics_from_trajectory_json(json_path, sep_nm=5.0)
                
                # Extract resolution_fail_rate, los_failure_rate, and oscillation_rate from detector (preserve NaN)
                res_val = fallback_metrics.get("resolution_fail_rate", 0.0)
                resolution_fail_rate = float(res_val) if not (isinstance(res_val, float) and np.isnan(res_val)) else float('nan')
                
                los_val = fallback_metrics.get("los_failure_rate", 0.0)
                los_failure_rate = float(los_val) if not (isinstance(los_val, float) and np.isnan(los_val)) else float('nan')
                
                osc_val = fallback_metrics.get("oscillation_rate", 0.0)
                oscillation_rate = float(osc_val) if not (isinstance(osc_val, float) and np.isnan(osc_val)) else float('nan')
                
                # Use default metrics with new fields
                cm_summary = {
                    "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                    "min_separation_nm": 200.0, "num_conflict_steps": 0,
                    # Previously missing columns - now with defaults
                    "flight_time_s": 0.0,
                    "num_los_events": 0,
                    "total_los_duration": 0.0,
                    "total_path_length_nm": 0.0,
                    "path_efficiency": 1.0,
                    "waypoint_reached_ratio": 0.0,
                    # NEW: Extra path metrics with defaults
                    "total_extra_path_nm": 0.0,
                    "avg_extra_path_nm": 0.0,
                    "avg_extra_path_ratio": 0.0,
                    # NEW: Intervention metrics with defaults
                    "num_interventions": 0,
                    "num_interventions_matched": 0,
                    "num_interventions_false": 0,
                    # High-value metrics
                    "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
                    "alert_duty_cycle": 0.0, "alerts_per_min": 0.0, "total_alert_time_s": 0.0,
                    "avg_lead_time_s": 0.0,
                    "ghost_conflict": 0.0, "missed_conflict": 1.0,  # If no CSV, assume all conflicts missed
                    "resolution_fail_rate": resolution_fail_rate,  # Computed from trajectory JSON or 0.0
                    "los_failure_rate": los_failure_rate,  # NEW: Computed from trajectory JSON or 0.0
                    "oscillation_rate": oscillation_rate,  # Computed from trajectory JSON or 0.0
                    "reward_total": 0.0,  # NEW: Total reward from trajectory
                }
                
            # Create summary row with proper column ordering (key identification columns first)
            summary_row = {
                # Key identification columns (first for easy reading)
                "test_id": test_id,
                "target_agent": agent_id,
                "shift_type": shift_type,
                "shift_value": shift_value,
                "shift_range": "micro" if abs(shift_value) <= 15 else "macro",
                "episode_id": ep_idx,
                "seed": seed,  # Add seed for reproducibility
                "description": description,
                # Then add all the computed metrics
                **cm_summary
            }
            summary_rows.append(summary_row)
            
            # Save per-episode summary
            episode_summary_path = os.path.join(shift_dir, f"summary_{ep_tag}.csv")
            pd.DataFrame([cm_summary]).to_csv(episode_summary_path, index=False)

            # VIZ: episode-level artifacts
            if generate_viz and csv_path:
                # Since hallucination data is already in the CSV, we can directly use it for visualization
                try:
                    df = pd.read_csv(csv_path)
                    # Extract series data from CSV columns for visualization compatibility
                    cm_series = {
                        "gt_conflict": df.get("gt_conflict", pd.Series([0] * len(df))).tolist(),
                        "alert": df.get("predicted_alert", pd.Series([0] * len(df))).tolist(),
                        "tp": df.get("tp", pd.Series([0] * len(df))).tolist(),
                        "fp": df.get("fp", pd.Series([0] * len(df))).tolist(),
                        "fn": df.get("fn", pd.Series([0] * len(df))).tolist(),
                        "tn": df.get("tn", pd.Series([0] * len(df))).tolist(),
                        "min_separation_nm": df.get("rt_min_separation_nm", df.get("min_separation_nm", pd.Series([0] * len(df)))).tolist(),
                    }
                    _ = make_episode_visuals(shift_dir, csv_path, cm_series, title=f"{description} | ep={ep_idx}")
                except Exception as e:
                    print(f"Warning: Failed to generate visualizations: {e}")

            LOGGER.info(f"Finished {ep_tag}: {description}")

    # Save comprehensive analysis
    df = pd.DataFrame(summary_rows)
    
    # Main summary CSV
    main_summary_csv = os.path.join(results_dir, "targeted_shift_test_summary.csv")
    df.to_csv(main_summary_csv, index=False)
    
    # Create detailed analysis summaries
    _create_targeted_analysis(df, analysis_dir, timestamp, representative_agents)
    
    # Create run metadata
    _save_targeted_run_metadata(results_dir, repo_root, checkpoint_path, episodes_per_shift, timestamp, len(targeted_shifts), representative_agents, scenario_name)
    
    # VIZ: run-level artifacts (degradation curves, vulnerability heatmap)
    if generate_viz:
        make_run_visuals(results_dir, scenario_name)
        
        # Generate trajectory comparison maps (baseline vs each shift)
        baseline_dir = os.path.join(shifts_dir, "baseline")
        viz_dir = os.path.join(results_dir, "viz")
        
        if os.path.exists(baseline_dir):
            print("Generating trajectory comparison visualizations...")
            
            # Generate geographic comparison maps
            comparison_maps = generate_shift_comparison_maps(
                baseline_dir=baseline_dir,
                shifts_dir=shifts_dir,
                viz_dir=viz_dir,
                sep_nm=5.0
            )
            print(f"Generated {len(comparison_maps)} trajectory comparison maps in {viz_dir}")
            
            # Generate Plotly-based trajectory analysis
            try:
                dashboard_file = create_shift_analysis_dashboard(
                    baseline_dir=baseline_dir,
                    shifts_dir=shifts_dir,
                    viz_dir=viz_dir,
                    scenario_path=scenario_path
                )
                print(f"Generated interactive trajectory analysis dashboard: {os.path.basename(dashboard_file)}")
            except Exception as e:
                print(f"Warning: Failed to generate trajectory analysis dashboard: {e}")
        else:
            print(f"Warning: Baseline directory not found for comparison visualizations: {baseline_dir}")
    
    LOGGER.info(f"Wrote comprehensive targeted shift analysis to: {results_dir}")
    LOGGER.info(f"Total configurations tested: {len(targeted_shifts)}")
    LOGGER.info(f"Total episodes run: {len(summary_rows)}")
    LOGGER.info(f"Skipped {skipped_count}/{total_shifts_attempted} episodes due to t=0 separation violations (<5.0 NM)")
    
    # Clean up Ray resources
    try:
        if hasattr(algo, 'stop'):
            algo.stop()
        # Note: We don't shutdown Ray here as it might be used by the calling CLI
        LOGGER.info("Shift testing cleanup completed")
    except Exception as e:
        LOGGER.warning(f"Cleanup warning: {e}")
    
    return main_summary_csv


def parse_arguments():
    """Parse command line arguments for targeted shift testing."""
    parser = argparse.ArgumentParser(
        description="Run targeted distribution shift testing for MARL collision avoidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use auto-detected checkpoint with parallel scenario
  python intrashift_tester.py
  
  # Specify custom checkpoint
  python intrashift_tester.py --checkpoint /path/to/checkpoint
  
  # Use different scenario with visualizations
  python intrashift_tester.py --scenario head_on --episodes 5 --viz
  
  # Run all scenarios with visualizations
  python intrashift_tester.py --scenario all --episodes 3 --viz
  
  # Full custom run
  python intrashift_tester.py --checkpoint models/my_model --scenario parallel --episodes 10 --algorithm SAC --viz
        """
    )
    
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to the trained model checkpoint directory. If not specified, will auto-detect."
    )
    
    parser.add_argument(
        "--scenario", "-s", 
        type=str,
        default="head_on",
        help="Scenario name (without .json extension) or 'all' to run all scenarios. Default: head_on"
    )
    
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=3,
        help="Number of episodes per shift configuration. Default: 3"
    )
    
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        choices=["PPO", "SAC"],
        default="PPO",
        help="RL algorithm used for training. Default: PPO"
    )
    
    parser.add_argument(
        "--repo-root", "-r",
        type=str,
        default=None,
        help="Project root directory. If not specified, will auto-detect."
    )
    
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit"
    )
    
    parser.add_argument(
        "--list-checkpoints",
        action="store_true", 
        help="List available checkpoints and exit"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without running tests"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Generate episode and run visualizations"
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds for episode reproducibility (e.g., --seeds 42,123,456). If not specified, random seeds will be generated."
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Custom output directory name. If not specified, will use timestamped directory."
    )
    
    return parser.parse_args()


def auto_detect_checkpoint(repo_root: str) -> Optional[str]:
    """Auto-detect the most recent checkpoint."""
    checkpoints = []
    
    # Check models directory and its subdirectories
    models_dir = os.path.join(repo_root, "models")
    if os.path.exists(models_dir):
        # Direct checkpoints in models/
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                if item.startswith("checkpoint_"):
                    # Direct checkpoint directory
                    checkpoints.append((item_path, os.path.getmtime(item_path)))
                else:
                    # Check for models subdirectory (like results_xxx/models)
                    models_subdir = os.path.join(item_path, "models")
                    if os.path.exists(models_subdir):
                        checkpoints.append((models_subdir, os.path.getmtime(models_subdir)))
    
    # Check results directories for model subdirectories
    results_dir = os.path.join(repo_root, "results")
    if os.path.exists(results_dir):
        for result_folder in os.listdir(results_dir):
            result_path = os.path.join(results_dir, result_folder)
            if os.path.isdir(result_path):
                models_path = os.path.join(result_path, "models")
                if os.path.exists(models_path):
                    checkpoints.append((models_path, os.path.getmtime(models_path)))
    
    # Return the most recent checkpoint
    if checkpoints:
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return checkpoints[0][0]
    
    return None


def list_available_scenarios(repo_root: str):
    """List all available scenario files."""
    scenarios_dir = os.path.join(repo_root, "scenarios")
    if not os.path.exists(scenarios_dir):
        print("No scenarios directory found.")
        return
    
    scenarios = [f[:-5] for f in os.listdir(scenarios_dir) if f.endswith('.json')]
    if scenarios:
        print("Available scenarios:")
        for scenario in sorted(scenarios):
            scenario_path = os.path.join(scenarios_dir, f"{scenario}.json")
            try:
                with open(scenario_path, 'r') as f:
                    data = json.load(f)
                    agents = len(data.get('agents', []))
                    print(f"  • {scenario} ({agents} agents)")
            except:
                print(f"  • {scenario} (invalid format)")
    else:
        print("No scenario files found.")


def list_available_checkpoints(repo_root: str):
    """List all available checkpoints."""
    print("Searching for checkpoints...")
    
    checkpoints = []
    
    # Check models directory and its subdirectories
    models_dir = os.path.join(repo_root, "models")
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item)
            if os.path.isdir(item_path):
                if item.startswith("checkpoint_"):
                    # Direct checkpoint directory
                    mod_time = os.path.getmtime(item_path)
                    checkpoints.append((item_path, mod_time, f"models/{item}/"))
                else:
                    # Check for models subdirectory (like results_xxx/models)
                    models_subdir = os.path.join(item_path, "models")
                    if os.path.exists(models_subdir):
                        mod_time = os.path.getmtime(models_subdir)
                        checkpoints.append((models_subdir, mod_time, f"models/{item}/"))
    
    # Check results directories
    results_dir = os.path.join(repo_root, "results")
    if os.path.exists(results_dir):
        for result_folder in os.listdir(results_dir):
            result_path = os.path.join(results_dir, result_folder)
            if os.path.isdir(result_path):
                models_path = os.path.join(result_path, "models")
                if os.path.exists(models_path):
                    mod_time = os.path.getmtime(models_path)
                    checkpoints.append((models_path, mod_time, f"results/{result_folder}/"))
    
    if checkpoints:
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        print("Available checkpoints (newest first):")
        for i, (path, mod_time, prefix) in enumerate(checkpoints):
            mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
            marker = " [AUTO-DETECTED]" if i == 0 else ""
            print(f"  {i+1}. {prefix}models (modified: {mod_date}){marker}")
            print(f"     Full path: {path}")
    else:
        print("No checkpoints found.")


if __name__ == "__main__":
    # Encourage use of ATC CLI for better structure
    print("⚠️  Note: For better structure and robustness, consider using:")
    print("    python atc_cli.py test-shifts --intrashift [options]")
    print("🔄 Running direct execution...\n")
    
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add project root to path for direct execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    REPO = args.repo_root or os.path.dirname(os.path.dirname(current_dir))
    
    if REPO not in sys.path:
        sys.path.append(REPO)
    
    # Handle list commands
    if args.list_scenarios:
        list_available_scenarios(REPO)
        sys.exit(0)
    
    if args.list_checkpoints:
        list_available_checkpoints(REPO)
        sys.exit(0)
    
    # Determine checkpoint path
    if args.checkpoint:
        ckpt = args.checkpoint
        if not os.path.exists(ckpt):
            print(f"Error: Checkpoint path does not exist: {ckpt}")
            sys.exit(1)
    else:
        ckpt = auto_detect_checkpoint(REPO)
        if not ckpt:
            print("Error: No checkpoints found.")
            print("Available options:")
            print("1. Specify checkpoint with --checkpoint /path/to/checkpoint")
            print("2. Train a model first")
            print("3. Use --list-checkpoints to see available checkpoints")
            sys.exit(1)
        print(f"Auto-detected checkpoint: {ckpt}")
    
    # Verify scenario exists or handle 'all' case
    if args.scenario == "all":
        scenarios_dir = os.path.join(REPO, "scenarios")
        if not os.path.exists(scenarios_dir):
            print(f"Error: Scenarios directory not found: {scenarios_dir}")
            sys.exit(1)
        scenario_names = [os.path.splitext(f)[0] for f in os.listdir(scenarios_dir) if f.endswith(".json")]
        if not scenario_names:
            print(f"Error: No scenario files found in {scenarios_dir}")
            sys.exit(1)
    else:
        scenario_names = [args.scenario]
        scenario_path = os.path.join(REPO, "scenarios", f"{args.scenario}.json")
        if not os.path.exists(scenario_path):
            print(f"Error: Scenario file not found: {scenario_path}")
            print("Use --list-scenarios to see available scenarios")
            sys.exit(1)
    
    # Determine algorithm class
    if args.algorithm == "SAC":
        algo_class = SAC
    else:
        algo_class = PPO
    
    # Show configuration
    print("=" * 80)
    # Parse seeds if provided
    seeds = None
    if args.seeds:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
            if len(seeds) != args.episodes:
                print(f"Warning: Number of seeds ({len(seeds)}) doesn't match episodes ({args.episodes})")
                print("Seeds will be cycled or truncated as needed")
        except ValueError:
            print(f"Error: Invalid seeds format: {args.seeds}")
            print("Use comma-separated integers, e.g., --seeds 42,123,456")
            sys.exit(1)

    print("TARGETED DISTRIBUTION SHIFT TESTING CONFIGURATION")
    print("=" * 80)
    print(f"Repository root: {REPO}")
    print(f"Checkpoint: {ckpt}")
    print(f"Scenario(s): {scenario_names if args.scenario == 'all' else args.scenario}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Episodes per shift: {args.episodes}")
    print(f"Seeds: {seeds if seeds else 'Auto-generated (reproducible)'}")
    print(f"Generate visualizations: {args.viz}")
    print("=" * 80)
    
    if args.dry_run:
        print("Dry run completed. Use --help for more options.")
        sys.exit(0)
    
    try:
        # Run the intrashift testing (representative agents, t=0 guard)
        print("Starting intrashift testing with geometric deduplication...")
        
        if args.scenario == "all":
            for scn in sorted(scenario_names):
                print(f"\n--- Processing scenario: {scn} ---")
                out_csv = run_intrashift_grid(
                    repo_root=REPO,
                    algo_class=algo_class,
                    checkpoint_path=ckpt,
                    scenario_name=scn,
                    episodes_per_shift=args.episodes,
                    seeds=seeds,
                    generate_viz=args.viz,
                    outdir=args.outdir
                )
                print(f"[{scn}] summary -> {out_csv}")
        else:
            out_csv = run_intrashift_grid(
                repo_root=REPO,
                algo_class=algo_class,
                checkpoint_path=ckpt,
                scenario_name=args.scenario,
                episodes_per_shift=args.episodes,
                seeds=seeds,
                generate_viz=args.viz,
                outdir=args.outdir
            )
        
        print("=" * 80)
        print("✅ INTRASHIFT TESTING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        if args.scenario != "all":
            print(f"Results summary: {out_csv}")
            print(f"Full results directory: {os.path.dirname(out_csv)}")
        else:
            print(f"Multiple scenario results generated in results/ directory")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


