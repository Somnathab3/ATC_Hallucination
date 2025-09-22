"""
Targeted distribution shift tester: varying one agent's parameters while others remain nominal.

This creates more complex scenarios by:
1. Modifying only one agent at a time (speed, position, heading)
2. Using micro to macro range variations to identify training model failures
3. Positioning agents closer to increase conflict probability
4. Testing edge cases that could cause hallucinations and safety violations

Key differences from unison shifts:
- Only one agent is modified per test case
- Larger range variations to push beyond training envelope
- Conflict-inducing position shifts (moving agents closer)
- Enhanced analysis for single-agent impact assessment
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
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
import bluesky as bs

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from src.environment.marl_collision_env_minimal import MARLCollisionEnv
from src.analysis.hallucination_detector_enhanced import HallucinationDetector


LOGGER = logging.getLogger("targeted_shift_tester")
LOGGER.setLevel(logging.INFO)


def write_traj_csv(traj, out_csv):
    """Write trajectory data to CSV format."""
    rows = []
    T = len(traj["positions"])
    aids = list(traj["positions"][0].keys()) if T else []
    for t in range(T):
        for aid in aids:
            lat, lon = traj["positions"][t][aid]
            hdg = traj["agents"][aid]["headings"][t]
            spd = traj["agents"][aid]["speeds"][t]
            a = np.asarray(traj["actions"][t].get(aid, [0, 0]), float)
            rows.append({
                "t": t, "agent": aid, "lat": lat, "lon": lon,
                "hdg_deg": hdg, "spd_kt": spd,
                "dpsi_deg": float(a[0]), "dv_kt": float(a[1])
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def create_targeted_shift(agent_id: str, shift_type: str, shift_value: float, scenario_agents: List[str]) -> Dict[str, Any]:
    """
    Create a targeted shift affecting only one agent.
    
    Args:
        agent_id: Which agent to modify (A1, A2, A3)
        shift_type: Type of modification (speed, position_closer, position_lateral, heading)
        shift_value: Magnitude of the shift
        scenario_agents: List of all agent IDs in scenario
    
    Returns:
        Dictionary with agent-specific shifts
    """
    shift_config = {}
    
    for aid in scenario_agents:
        if aid == agent_id:
            # Apply shift to target agent
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
            else:
                shift_config[aid] = {}
        else:
            # Keep other agents at baseline
            shift_config[aid] = {}
    
    return shift_config


def create_conflict_inducing_shifts() -> List[Tuple[str, str, str, float, str]]:
    """
    Create a comprehensive list of targeted shifts designed to induce conflicts.
    
    Returns:
        List of (test_id, agent_id, shift_type, shift_value, description) tuples
    """
    shifts = []
    
    # Speed variations - micro to macro range
    # Micro changes (small perturbations)
    speed_micro = [-10, -5, 5, 10]  # ±10 kt
    # Macro changes (large deviations to test failure modes)
    speed_macro = [-30, -20, 20, 30]  # ±30 kt
    
    for agent in ["A1", "A2", "A3"]:
        # Micro speed changes
        for delta in speed_micro:
            test_id = f"speed_micro_{agent}_{delta:+d}kt"
            desc = f"Agent {agent} speed micro-shift: {delta:+d} kt"
            shifts.append((test_id, agent, "speed", delta, desc))
        
        # Macro speed changes 
        for delta in speed_macro:
            test_id = f"speed_macro_{agent}_{delta:+d}kt"
            desc = f"Agent {agent} speed macro-shift: {delta:+d} kt"
            shifts.append((test_id, agent, "speed", delta, desc))
    
    # Position shifts to create conflicts
    # Moving agents closer together (reducing separation)
    position_closer_micro = [0.05, 0.1, 0.15]  # 0.05-0.15 degrees ≈ 3-9 NM closer
    position_closer_macro = [0.2, 0.3, 0.4]    # 0.2-0.4 degrees ≈ 12-24 NM closer
    
    # A2 (middle agent) moving closer to others - highest conflict potential
    for delta in position_closer_micro:
        # Move A2 north (toward A3)
        test_id = f"pos_closer_micro_A2_north_{delta:.2f}deg"
        desc = f"A2 moved {delta:.2f}° north (closer to A3)"
        shifts.append((test_id, "A2", "position_closer", delta, desc))
        
        # Move A2 south (toward A1)  
        test_id = f"pos_closer_micro_A2_south_{delta:.2f}deg"
        desc = f"A2 moved {delta:.2f}° south (closer to A1)"
        shifts.append((test_id, "A2", "position_closer", -delta, desc))
    
    for delta in position_closer_macro:
        # Move A2 north (toward A3)
        test_id = f"pos_closer_macro_A2_north_{delta:.2f}deg"
        desc = f"A2 moved {delta:.2f}° north (major shift toward A3)"
        shifts.append((test_id, "A2", "position_closer", delta, desc))
        
        # Move A2 south (toward A1)
        test_id = f"pos_closer_macro_A2_south_{delta:.2f}deg"
        desc = f"A2 moved {delta:.2f}° south (major shift toward A1)"
        shifts.append((test_id, "A2", "position_closer", -delta, desc))
    
    # Lateral (crossing) position shifts
    lateral_micro = [0.05, 0.1, 0.15]  # Small lateral deviations
    lateral_macro = [0.2, 0.3, 0.4]    # Large lateral deviations
    
    for agent in ["A1", "A2", "A3"]:
        for delta in lateral_micro + lateral_macro:
            range_type = "micro" if delta in lateral_micro else "macro"
            # East deviation
            test_id = f"pos_lateral_{range_type}_{agent}_east_{delta:.2f}deg"
            desc = f"Agent {agent} lateral {range_type}-shift: {delta:.2f}° east"
            shifts.append((test_id, agent, "position_lateral", delta, desc))
            
            # West deviation
            test_id = f"pos_lateral_{range_type}_{agent}_west_{delta:.2f}deg"
            desc = f"Agent {agent} lateral {range_type}-shift: {delta:.2f}° west"
            shifts.append((test_id, agent, "position_lateral", -delta, desc))
    
    # Heading deviations - creating converging/diverging paths
    heading_micro = [-10, -5, 5, 10]    # ±10 degrees
    heading_macro = [-30, -20, 20, 30]  # ±30 degrees
    
    for agent in ["A1", "A2", "A3"]:
        # Micro heading changes
        for delta in heading_micro:
            test_id = f"hdg_micro_{agent}_{delta:+d}deg"
            desc = f"Agent {agent} heading micro-shift: {delta:+d}°"
            shifts.append((test_id, agent, "heading", delta, desc))
        
        # Macro heading changes
        for delta in heading_macro:
            test_id = f"hdg_macro_{agent}_{delta:+d}deg"
            desc = f"Agent {agent} heading macro-shift: {delta:+d}°"
            shifts.append((test_id, agent, "heading", delta, desc))
    
    return shifts


def _create_targeted_analysis(df: pd.DataFrame, analysis_dir: str, timestamp: str):
    """Create detailed analysis summaries for targeted shifts with agent-specific focus."""
    
    # Analysis by target agent
    for agent_id in df['target_agent'].unique():
        agent_df = df[df['target_agent'] == agent_id].copy()
        
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
            'oscillation_rate': ['mean', 'std']
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
        'oscillation_rate': 'mean'
    }).round(4).reset_index()
    
    # Calculate derived metrics
    type_range_stats['precision'] = type_range_stats['tp'] / (type_range_stats['tp'] + type_range_stats['fp'] + 1e-10)
    type_range_stats['recall'] = type_range_stats['tp'] / (type_range_stats['tp'] + type_range_stats['fn'] + 1e-10)
    type_range_stats['f1_score'] = 2 * (type_range_stats['precision'] * type_range_stats['recall']) / (type_range_stats['precision'] + type_range_stats['recall'] + 1e-10)
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
        'missed_conflict': 'mean'
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
        'macro_vs_micro_conflicts': {k: int(v) for k, v in df.groupby('shift_range')['num_los_events'].sum().to_dict().items()}
    }
    
    summary_path = os.path.join(analysis_dir, "targeted_shift_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)


def _save_targeted_run_metadata(results_dir: str, repo_root: str, checkpoint_path: str, 
                               episodes_per_shift: int, timestamp: str, total_shifts: int,
                               scenario_name: str = "parallel"):
    """Save metadata about the targeted shift testing run."""
    
    metadata = {
        "timestamp": timestamp,
        "run_date": datetime.now().isoformat(),
        "test_type": "targeted_shifts",
        "checkpoint_path": checkpoint_path,
        "episodes_per_shift": episodes_per_shift,
        "scenario": f"{scenario_name}.json",
        "target_agents": ["A1", "A2", "A3"],
        "shift_types": ["speed", "position_closer", "position_lateral", "heading"],
        "shift_ranges": ["micro", "macro"],
        "speed_shifts_kt": {"micro": "±5 to ±10", "macro": "±20 to ±30"},
        "position_shifts_deg": {"micro": "±0.05 to ±0.15", "macro": "±0.2 to ±0.4"},
        "heading_shifts_deg": {"micro": "±5 to ±10", "macro": "±20 to ±30"},
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

## Key Metrics (Enhanced for Conflict Detection)
- **Safety**: LoS count, PH5_time_frac, min_CPA_nm
- **Detection**: TP/FP/FN/TN for conflict prediction accuracy
- **Resolution**: TP_res/FP_res/FN_res within 60s after alert
- **Efficiency**: Path length, flight time, waypoint completion
- **Stability**: Action oscillation rate

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
    return ParallelPettingZooEnv(MARLCollisionEnv(env_config))


def run_targeted_shift_grid(repo_root: str,
                           algo_class,
                           checkpoint_path: str,
                           scenario_name: str = "parallel",
                           episodes_per_shift: int = 5,
                           seeds: Optional[List[int]] = None) -> str:
    """
    Run targeted shift testing where only one agent is modified per test case.
    
    Args:
        repo_root: Root directory of the project
        algo_class: RLLib algorithm class (PPO, SAC, etc.)
        checkpoint_path: Path to the trained model checkpoint
        scenario_name: Name of the scenario file (without .json extension)
        episodes_per_shift: Number of episodes to run per shift configuration
        seeds: List of random seeds to use for episodes
    
    Returns:
        Path to the main summary CSV file
    """
    scenario_path = os.path.join(repo_root, "scenarios", f"{scenario_name}.json")
    
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

    # Base env config
    base_env_config = {
        "scenario_path": scenario_path,
        "action_delay_steps": 0,
        "max_episode_steps": 100,
        "separation_nm": 5.0,
        "log_trajectories": True,
        "results_dir": results_dir,
        "seed": 42,
    }

    config = (algo_class.get_default_config()
             .environment(env=env_name, env_config=base_env_config)
             .framework("torch")
             .resources(num_gpus=0)
             .env_runners(num_env_runners=0)
             .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False))
    algo = algo_class(config=config)
    algo.restore(checkpoint_path)
    
    # Get correct policy ID from trained model
    try:
        policy_ids = list(algo.get_policy().keys()) if hasattr(algo, 'get_policy') else ["default_policy"]
    except:
        try:
            policy_ids = list(algo.workers.local_worker().policy_map.keys())
        except:
            policy_ids = ["default_policy"]
    
    policy_id = "shared_policy" if "shared_policy" in policy_ids else policy_ids[0]

    # Generate targeted shift configurations
    targeted_shifts = create_conflict_inducing_shifts()
    seeds = seeds or list(range(episodes_per_shift))

    detector = HallucinationDetector(action_period_s=10.0, res_window_s=60.0)
    summary_rows = []

    LOGGER.info(f"Running {len(targeted_shifts)} targeted shift configurations with {episodes_per_shift} episodes each")

    for test_id, agent_id, shift_type, shift_value, description in targeted_shifts:
        for ep_idx, seed in enumerate(seeds):
            # Create per-shift directory structure
            ep_tag = f"{test_id}_ep{ep_idx}"
            shift_dir = os.path.join(shifts_dir, test_id)
            os.makedirs(shift_dir, exist_ok=True)
            
            # Update env config with shift-specific results directory
            env_config = base_env_config.copy()
            env_config["results_dir"] = shift_dir
            env_config["seed"] = seed
            
            # Create targeted shift configuration
            shift_config = create_targeted_shift(agent_id, shift_type, shift_value, scenario_agents)
            
            # Build env with proper wrapper
            env = MARLCollisionEnv(env_config)
            env_pz = ParallelPettingZooEnv(env)
            
            try:
                obs, _ = env_pz.reset(options={"targeted_shift": shift_config})
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
            while True:
                # Policy actions for all agents using correct policy ID
                act = {}
                for aid in env.agents:
                    if aid in obs:
                        a = algo.compute_single_action(obs[aid], explore=False, policy_id=policy_id)
                        act[aid] = a
                    else:
                        act[aid] = [0.0, 0.0]

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

            # Save trajectory files (both JSON and CSV)
            json_path = os.path.join(shift_dir, f"trajectory_{ep_tag}.json")
            csv_path = os.path.join(shift_dir, f"trajectory_{ep_tag}.csv")
            
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(traj, f)
            write_traj_csv(traj, csv_path)

            # Compute hallucination metrics
            cm = detector.compute(traj, sep_nm=base_env_config["separation_nm"])
            cm.update({
                "test_id": test_id,
                "target_agent": agent_id,
                "shift_type": shift_type,
                "shift_value": shift_value,
                "shift_range": "micro" if abs(shift_value) <= 15 else "macro",
                "episode_id": ep_idx,
                "description": description
            })
            summary_rows.append(cm)
            
            # Save per-episode summary
            episode_summary_path = os.path.join(shift_dir, f"summary_{ep_tag}.csv")
            pd.DataFrame([cm]).to_csv(episode_summary_path, index=False)

            LOGGER.info(f"Finished {ep_tag}: {description}")

    # Save comprehensive analysis
    df = pd.DataFrame(summary_rows)
    
    # Main summary CSV
    main_summary_csv = os.path.join(results_dir, "targeted_shift_test_summary.csv")
    df.to_csv(main_summary_csv, index=False)
    
    # Create detailed analysis summaries
    _create_targeted_analysis(df, analysis_dir, timestamp)
    
    # Create run metadata
    _save_targeted_run_metadata(results_dir, repo_root, checkpoint_path, episodes_per_shift, timestamp, len(targeted_shifts), scenario_name)
    
    LOGGER.info(f"Wrote comprehensive targeted shift analysis to: {results_dir}")
    LOGGER.info(f"Total configurations tested: {len(targeted_shifts)}")
    LOGGER.info(f"Total episodes run: {len(summary_rows)}")
    
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
  
  # Use different scenario
  python targeted_shift_tester.py --scenario head_on --episodes 5
  
  # Full custom run
  python targeted_shift_tester.py --checkpoint models/my_model --scenario parallel --episodes 10 --algorithm SAC
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
        default="parallel",
        help="Scenario name (without .json extension). Default: parallel"
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
    
    # Verify scenario exists
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
    print(f"Scenario: {args.scenario}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Episodes per shift: {args.episodes}")
    print(f"Scenario file: {scenario_path}")
    print("=" * 80)
    
    if args.dry_run:
        print("Dry run completed. Use --help for more options.")
        sys.exit(0)
    
    try:
        # Run the targeted shift testing
        print("Starting targeted shift testing...")
        out_csv = run_targeted_shift_grid(
            repo_root=REPO,
            algo_class=algo_class,
            checkpoint_path=ckpt,
            scenario_name=args.scenario,
            episodes_per_shift=args.episodes
        )
        
        print("=" * 80)
        print("✅ TARGETED SHIFT TESTING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Results summary: {out_csv}")
        print(f"Full results directory: {os.path.dirname(out_csv)}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)