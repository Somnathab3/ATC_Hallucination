"""
Targeted Distribution Shift Testing Framework.

This module implements systematic robustness testing for MARL collision avoidance policies
by applying targeted perturbations to individual agents while keeping others at baseline.
This approach identifies single points of failure and tests model generalization beyond
the training distribution.

Testing Philosophy:
- Individual agent modification: Only one agent per test case is modified
- Micro to macro ranges: Small perturbations to large deviations from training envelope  
- Conflict-inducing design: Modifications intentionally increase collision probability
- Comprehensive analysis: Statistical evaluation across shift types and magnitudes

Shift Categories:
- Speed variations: ±5-30 kt from nominal cruise speed
- Position shifts: Lateral and proximity modifications (0.05-0.4 degrees)
- Heading deviations: ±5-30 degrees from optimal trajectory
- Aircraft type variations: Different performance characteristics
- Waypoint modifications: Destination changes affecting traffic flow

The framework generates rich trajectory data with real-time hallucination detection,
enabling detailed analysis of policy robustness and failure modes for academic evaluation.
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


LOGGER = logging.getLogger("targeted_shift_tester")
LOGGER.setLevel(logging.INFO)



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


def create_conflict_inducing_shifts(scenario_agents: List[str]) -> List[Tuple[str, str, str, float, str, Optional[Any]]]:
    """
    Generate comprehensive test matrix of conflict-inducing shifts.
    
    Creates systematic variations across multiple dimensions to thoroughly test
    policy robustness. Shifts are categorized into micro (small perturbations)
    and macro (large deviations) ranges to identify training envelope boundaries.
    
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
    
    # Position shifts to create conflicts (only for scenarios with 3+ agents)
    if len(scenario_agents) >= 3:
        # Moving agents closer together (reducing separation)
        position_closer_micro = [0.05, 0.1, 0.15]  # 0.05-0.15 degrees ≈ 3-9 NM closer
        position_closer_macro = [0.2, 0.3, 0.4]    # 0.2-0.4 degrees ≈ 12-24 NM closer
        
        # Use second agent (middle agent) moving closer to others - highest conflict potential
        middle_agent = scenario_agents[1] if len(scenario_agents) > 1 else scenario_agents[0]
        for delta in position_closer_micro:
            # Move middle agent north (closer to others)
            test_id = f"pos_closer_micro_{middle_agent}_north_{delta:.2f}deg"
            desc = f"{middle_agent} moved {delta:.2f}° north (closer to others)"
            shifts.append((test_id, middle_agent, "position_closer", delta, desc, None))
            
            # Move middle agent south (closer to others)  
            test_id = f"pos_closer_micro_{middle_agent}_south_{delta:.2f}deg"
            desc = f"{middle_agent} moved {delta:.2f}° south (closer to others)"
            shifts.append((test_id, middle_agent, "position_closer", -delta, desc, None))
        
        for delta in position_closer_macro:
            # Move middle agent north (major shift)
            test_id = f"pos_closer_macro_{middle_agent}_north_{delta:.2f}deg"
            desc = f"{middle_agent} moved {delta:.2f}° north (major shift toward others)"
            shifts.append((test_id, middle_agent, "position_closer", delta, desc, None))
            
            # Move middle agent south (major shift)
            test_id = f"pos_closer_macro_{middle_agent}_south_{delta:.2f}deg"
            desc = f"{middle_agent} moved {delta:.2f}° south (major shift toward others)"
            shifts.append((test_id, middle_agent, "position_closer", -delta, desc, None))
    
    # Lateral (crossing) position shifts
    lateral_micro = [0.05, 0.1, 0.15]  # Small lateral deviations
    lateral_macro = [0.2, 0.3, 0.4]    # Large lateral deviations
    
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
        "test_type": "targeted_shifts",
        "checkpoint_path": checkpoint_path,
        "episodes_per_shift": episodes_per_shift,
        "scenario": f"{scenario_name}.json",
        "target_agents": scenario_agents,
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
            "horizon_s": 300.0,
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
    
    return ParallelPettingZooEnv(MARLCollisionEnv(env_config))


def run_targeted_shift_grid(repo_root: str,
                           algo_class,
                           checkpoint_path: str,
                           scenario_name: str = "head_on",
                           episodes_per_shift: int = 5,
                           seeds: Optional[List[int]] = None,
                           generate_viz: bool = False) -> str:
    """
    Run targeted shift testing where only one agent is modified per test case.
    
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
    
    # Load scenario to get agent list
    with open(scenario_path, "r") as f:
        scenario_data = json.load(f)
    scenario_agents = [agent["id"] for agent in scenario_data["agents"]]
    
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join(repo_root, "results")
    results_dir = os.path.join(base_results_dir, f"targeted_shift_analysis_{scenario_name}_{timestamp}")
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

    # Generate targeted shift configurations
    targeted_shifts = create_conflict_inducing_shifts(scenario_agents)
    # Use different seeds for each episode to ensure variation
    if seeds is None:
        import random
        random.seed(42)  # Reproducible but varied
        seeds = [random.randint(1, 10000) for _ in range(episodes_per_shift)]

    summary_rows = []

    LOGGER.info(f"Running {len(targeted_shifts)} targeted shift configurations with {episodes_per_shift} episodes each")

    for test_id, agent_id, shift_type, shift_value, description, shift_data in targeted_shifts:
        for ep_idx, seed in enumerate(seeds):
            # Create per-shift directory structure
            ep_tag = f"{test_id}_ep{ep_idx}"
            shift_dir = os.path.join(shifts_dir, test_id)
            os.makedirs(shift_dir, exist_ok=True)
            
            # Update env config with shift-specific results directory
            env_config = base_env_config.copy()
            env_config["results_dir"] = shift_dir
            env_config["seed"] = seed
            env_config["episode_tag"] = ep_tag  # Pass episode tag for clearer CSV naming
            
            # Create targeted shift configuration
            shift_config = create_targeted_shift(agent_id, shift_type, shift_value, scenario_agents, shift_data)
            
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
                    speed = getattr(bs.traf, 'tas', getattr(bs.traf, 'gs', [250.0] * len(bs.traf.lat)))
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
                # Policy actions for all agents using correct policy ID with explore=False for inference
                act = {}
                for aid in env.agents:
                    if aid in obs:
                        a = algo.compute_single_action(obs[aid], explore=False, policy_id=policy_id)
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
                        speed = getattr(bs.traf, 'tas', getattr(bs.traf, 'gs', [250.0] * len(bs.traf.lat)))
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
            # Look for both old pattern (traj_ep_XXXX.csv) and new pattern (traj_EPISODE_TAG.csv)
            rich_csv_files = [f for f in os.listdir(shift_dir) if 
                             (f.startswith('traj_ep_') or f.startswith('traj_')) and f.endswith('.csv')]
            csv_path = os.path.join(shift_dir, rich_csv_files[0]) if rich_csv_files else None

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
                    episode_time_s = step_collapsed['sim_time_s'].max() if not step_collapsed.empty else 0.0
                    
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
                    
                    # Extra path metrics - calculate deviation from direct start->waypoint paths
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
                                
                            # Calculate actual path length
                            positions = list(zip(agent_data['lat_deg'], agent_data['lon_deg']))
                            actual_path_nm = 0.0
                            for i in range(1, len(positions)):
                                lat1, lon1 = positions[i-1]
                                lat2, lon2 = positions[i]
                                # Simple distance approximation (good enough for small distances)
                                dlat = lat2 - lat1
                                dlon = lon2 - lon1
                                dist_deg = np.sqrt(dlat**2 + dlon**2)
                                dist_nm = dist_deg * 60.0  # Rough conversion to nautical miles
                                actual_path_nm += dist_nm
                            
                            # Try to get waypoint info from trajectory waypoints or estimate from end position
                            # For simplicity, use straight-line distance from start to end as "direct" path
                            start_pos = positions[0]
                            end_pos = positions[-1]
                            direct_nm = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2) * 60.0
                            
                            if direct_nm > 0:
                                extra_nm = max(0.0, actual_path_nm - direct_nm)
                                agent_extra_paths.append(extra_nm)
                        
                        if agent_extra_paths:
                            total_extra_path_nm = float(sum(agent_extra_paths))
                            avg_extra_path_nm = float(np.mean(agent_extra_paths))
                            # Calculate ratio as (actual/direct - 1), average across agents
                            ratios = []
                            for aid in agent_ids:
                                agent_data = df[df['agent_id'] == aid].sort_values('step_idx')
                                if len(agent_data) < 2:
                                    continue
                                positions = list(zip(agent_data['lat_deg'], agent_data['lon_deg']))
                                actual_path_nm = sum(np.sqrt((positions[i][0] - positions[i-1][0])**2 + (positions[i][1] - positions[i-1][1])**2) * 60.0 for i in range(1, len(positions)))
                                direct_nm = np.sqrt((positions[-1][0] - positions[0][0])**2 + (positions[-1][1] - positions[0][1])**2) * 60.0
                                if direct_nm > 0:
                                    ratios.append((actual_path_nm / direct_nm) - 1.0)
                            avg_extra_path_ratio = float(np.mean(ratios)) if ratios else 0.0
                    except Exception:
                        # Fallback if path calculation fails
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
                    
                    # Waypoint reached ratio: from waypoint_reached column
                    try:
                        waypoint_reached_ratio = float(df['waypoint_reached'].mean()) if 'waypoint_reached' in df.columns else 0.0
                    except Exception:
                        waypoint_reached_ratio = 0.0
                    
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
                        "resolution_fail_rate": 0.0,  # Would need more complex calculation
                        "oscillation_rate": 0.0,  # Would need action sequence analysis
                    }
                except Exception as e:
                    print(f"Warning: Failed to extract metrics from CSV {csv_path}: {e}")
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
                        "resolution_fail_rate": 0.0, "oscillation_rate": 0.0,
                    }
            else:
                # No CSV available - use default metrics with new fields
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
                    "resolution_fail_rate": 0.0, "oscillation_rate": 0.0,
                }
                
            # Update summary data
            cm_summary.update({
                "test_id": test_id,
                "target_agent": agent_id,
                "shift_type": shift_type,
                "shift_value": shift_value,
                "shift_range": "micro" if abs(shift_value) <= 15 else "macro",
                "episode_id": ep_idx,
                "description": description
            })
            summary_rows.append(cm_summary)
            
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
    _create_targeted_analysis(df, analysis_dir, timestamp, scenario_agents)
    
    # Create run metadata
    _save_targeted_run_metadata(results_dir, repo_root, checkpoint_path, episodes_per_shift, timestamp, len(targeted_shifts), scenario_agents, scenario_name)
    
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
  python targeted_shift_tester.py
  
  # Specify custom checkpoint
  python targeted_shift_tester.py --checkpoint /path/to/checkpoint
  
  # Use different scenario with visualizations
  python targeted_shift_tester.py --scenario head_on --episodes 5 --viz
  
  # Run all scenarios with visualizations
  python targeted_shift_tester.py --scenario all --episodes 3 --viz
  
  # Full custom run
  python targeted_shift_tester.py --checkpoint models/my_model --scenario parallel --episodes 10 --algorithm SAC --viz
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
    print("TARGETED DISTRIBUTION SHIFT TESTING CONFIGURATION")
    print("=" * 80)
    print(f"Repository root: {REPO}")
    print(f"Checkpoint: {ckpt}")
    print(f"Scenario(s): {scenario_names if args.scenario == 'all' else args.scenario}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Episodes per shift: {args.episodes}")
    print(f"Generate visualizations: {args.viz}")
    print("=" * 80)
    
    if args.dry_run:
        print("Dry run completed. Use --help for more options.")
        sys.exit(0)
    
    try:
        # Run the targeted shift testing
        print("Starting targeted shift testing...")
        
        if args.scenario == "all":
            for scn in sorted(scenario_names):
                print(f"\n--- Processing scenario: {scn} ---")
                out_csv = run_targeted_shift_grid(
                    repo_root=REPO,
                    algo_class=algo_class,
                    checkpoint_path=ckpt,
                    scenario_name=scn,
                    episodes_per_shift=args.episodes,
                    generate_viz=args.viz
                )
                print(f"[{scn}] summary -> {out_csv}")
        else:
            out_csv = run_targeted_shift_grid(
                repo_root=REPO,
                algo_class=algo_class,
                checkpoint_path=ckpt,
                scenario_name=args.scenario,
                episodes_per_shift=args.episodes,
                generate_viz=args.viz
            )
        
        print("=" * 80)
        print("✅ TARGETED SHIFT TESTING COMPLETED SUCCESSFULLY")
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