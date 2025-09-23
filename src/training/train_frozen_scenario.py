
"""
Training on the single frozen scenario with     # Register env once
    env_name = "marl_collision_env_v0"
    register_env(env_name, lambda cfg: ParallelPettingZooEnv(MARLCollisionEnv(cfg)))

    # Setup timestamped results directory
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(repo_root, f"results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    env_config = { (PPO or SAC).

- Multi-agent configuration with shared policy (parameter sharing).
- Stops at 2M steps OR when 100 consecutive episodes have zero conflicts.
- Saves checkpoints every 100k steps into models/.
"""

import os
import json
import time
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import csv
import ray

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env

# Add project root to Python path for imports
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local imports (within repo)
from src.environment.marl_collision_env_minimal import MARLCollisionEnv


LOGGER = logging.getLogger("train_frozen")
LOGGER.setLevel(logging.INFO)

# Initialize Ray with better configuration to reduce warnings
def init_ray():
    """Initialize Ray with optimized settings to reduce warnings."""
    if not ray.is_initialized():
        ray.init(
            local_mode=True,  # Use local mode to avoid worker conflicts with BlueSky
            log_to_driver=False,
            configure_logging=False,
            ignore_reinit_error=True,  # Allow reinit if needed
        )


def make_env(env_config: Dict[str, Any]):
    """Factory for RLlib to build env instances (use ParallelPettingZooEnv wrapper)."""
    return ParallelPettingZooEnv(MARLCollisionEnv(env_config))


def train_frozen(repo_root: str,
                 algo: str = "PPO",
                 seed: int = 42,
                 scenario_name: str = "head_on",
                 timesteps_total: int = 2_000_000,
                 checkpoint_every: int = 100_000) -> str:
    """
    Returns: path to final checkpoint.
    """
    scenario_path = os.path.join(repo_root, "scenarios", f"{scenario_name}.json")
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario '{scenario_name}' not found at {scenario_path}. Available scenarios: head_on, t_formation, parallel, converging, canonical_crossing")

    # Initialize Ray with optimized settings
    init_ray()

    # Register env once
    env_name = "marl_collision_env_v0"
    register_env(env_name, lambda cfg: ParallelPettingZooEnv(MARLCollisionEnv(cfg)))

    # Setup timestamped results directory
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(repo_root, f"results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    env_config = {
        "scenario_path": scenario_path,
        "action_delay_steps": 0,
        "max_episode_steps": 100,
        "separation_nm": 5.0,
        "log_trajectories": True,
        "seed": seed,
        "results_dir": os.path.abspath(results_dir),  # Pass absolute path of timestamped results directory

        # NEW: Enable relative observations (no raw lat/lon)
        "obs_mode": "relative",
        "neighbor_topk": 3,
        
        # Collision detection settings (used for rewards only, not termination)
        "collision_nm": 3.0,              # collision threshold for reward penalties

        # Team shaping knobs (PBRS coordination rewards)
        "team_coordination_weight": 0.2,
        "team_gamma": 0.99,
        "team_share_mode": "responsibility",   # "even" | "responsibility" | "neighbor"
        "team_ema": 0.001,
        "team_cap": 0.005,
        "team_anneal": 1.0,
        "team_neighbor_threshold_km": 10.0,
        
        # Individual reward components (override defaults if desired)
        "drift_penalty_per_sec": -0.01,
        "progress_reward_per_km": 0.02,
        "backtrack_penalty_per_km": -0.02,
        "time_penalty_per_sec": -0.0005,
        "reach_reward": 50.0,  # Fixed: was 10.0, should match environment
        "heading_align_per_sec": 0.001,
        "intrusion_penalty": -50.0,
        "conflict_dwell_penalty_per_sec": -0.1,
    }

    # Shared policy (parameter-sharing) for all agents
    # Create a temporary env instance to get spaces, then delete it to avoid double BlueSky init
    tmp_env = MARLCollisionEnv({**env_config, "log_trajectories": False})
    
    # Get the first agent ID from the scenario to use for space definition
    first_agent_id = tmp_env.possible_agents[0] if tmp_env.possible_agents else "A0"
    
    policies = {
        "shared_policy": (
            None,  # use default model (MLP)
            tmp_env.observation_space(first_agent_id),
            tmp_env.action_space(first_agent_id),
            {}
        )
    }
    del tmp_env  # Clean up to avoid keeping BlueSky instance in driver
    def policy_mapping_fn(agent_id, episode=None, **kwargs):
        return "shared_policy"

    # Choose algorithm
    if algo.upper() == "PPO":
        config = (PPOConfig()
                  .environment(env=env_name, env_config=env_config)
                  .framework("torch")
                  .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                  .env_runners(
                      num_env_runners=1,        # reduced for speed optimization
                      num_envs_per_env_runner=1,        # keep 1 for BlueSky (heavier per process)
                      rollout_fragment_length=200,  # batches sampled per worker before sending to learner
                      max_requests_in_flight_per_env_runner=1,  # Reduce memory pressure
                  )
                  .training(gamma=0.99,
                            lr=5e-4,
                            train_batch_size=4096,   # Reduced from 16384 for more frequent updates
                            model={"fcnet_hiddens": [256, 256], 
                                   "fcnet_activation": "tanh",
                                   "free_log_std": False},
                            grad_clip=10.0)
                  .evaluation(
                      evaluation_interval=10,     # Less frequent evaluation to reduce conflicts
                      evaluation_duration=1,
                      evaluation_duration_unit="episodes",
                      evaluation_parallel_to_training=False,  # Sequential to avoid conflicts
                      evaluation_num_env_runners=0,  # No separate eval workers
                      evaluation_config={"explore": False}
                  )
                  .multi_agent(
                      policies=policies,
                      policy_mapping_fn=policy_mapping_fn,
                      policies_to_train=["shared_policy"],
                  )
                  .resources(num_gpus=0)
                  )
        # Don't normalize actions as we handle scaling in the environment
        config.clip_actions = True       # Ensure policy output is clipped to space bounds
        config.normalize_actions = False
        config.seed = seed
        algo_obj = config.build_algo()  # Use build_algo() instead of deprecated build()
    elif algo.upper() == "SAC":
        config = (SACConfig()
                  .environment(env=env_name, env_config=env_config)
                  .framework("torch")
                  .env_runners(
                      num_env_runners=1,
                      num_envs_per_env_runner=1,
                      rollout_fragment_length=200,
                  )
                  .training(gamma=0.99, lr=3e-4,
                            train_batch_size=8192,   # Reduced from 32768 for more frequent updates
                            model={"fcnet_hiddens": [256, 256], "fcnet_activation": "relu"})
                  .multi_agent(
                      policies=policies,
                      policy_mapping_fn=policy_mapping_fn,
                      policies_to_train=["shared_policy"],
                  )
                  .resources(num_gpus=0)
                  )
        config.seed = seed
        algo_obj = config.build_algo()  # Use build_algo() instead of deprecated build()
    else:
        raise ValueError("algo must be 'PPO' or 'SAC'")

    os.makedirs(os.path.join(repo_root, "models"), exist_ok=True)
    final_ckpt = ""
    zero_conflict_streak = 0

    # Setup progress CSV logging in the same timestamped directory
    progress_csv = os.path.join(results_dir, "training_progress.csv")
    if not os.path.exists(progress_csv):
        with open(progress_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["ts", "iter", "steps_sampled", "episodes", "reward_mean", "zero_conflict_streak"]
            )

    steps = 0
    it = 0
    total_episodes = 0
    best_reward = float('-inf')  # Track best reward for improved checkpointing
    while steps < timesteps_total:
        it += 1
        result = algo_obj.train()

        # Robust extraction across RLlib versions:
        rew = (result.get("env_runners",{}) or {}).get("episode_return_mean")
        if rew is None:
            rew = result.get("episode_reward_mean")

        steps = (result.get("env_runners",{}) or {}).get("num_env_steps_sampled") \
                or result.get("num_env_steps_sampled") or steps
        episodes_this_iter = (result.get("env_runners",{}) or {}).get("episodes_this_iter", 0)
        total_episodes += episodes_this_iter

        # Was the last episode conflict-free?
        conflict_free_ep = False
        try:
            traj_dir = results_dir  # use the timestamped directory you passed to the env
            traj_files = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir) if f.startswith("traj_ep")]
            if traj_files:
                latest = max(traj_files, key=os.path.getmtime)
                df = pd.read_csv(latest)
                if not df.empty and "episode_id" in df.columns:
                    # Check only the last episode for conflicts
                    last_ep = df["episode_id"].max()
                    last_ep_data = df[df["episode_id"] == last_ep]
                    conflict_free_ep = (last_ep_data["conflict_flag"].sum() == 0)
                else:
                    # Fallback: check entire file if no episode_id column
                    conflict_free_ep = (df["conflict_flag"].sum() == 0)
        except Exception as e:
            if it == 1:  # Only print debug for first iteration
                print(f"Debug: Error checking conflict_free_ep: {e}")
            pass

        zero_conflict_streak = zero_conflict_streak + 1 if conflict_free_ep else 0

        # Debug info for conflict detection
        if it == 1:  # Only print for first iteration
            print(f"Debug: conflict_free_ep={conflict_free_ep}, zero_conflict_streak={zero_conflict_streak}")

        # Extract additional metrics for visibility
        envr = result.get("env_runners", {}) or {}
        eps_this_iter = envr.get("episodes_this_iter", 0)
        steps_sec = result.get("num_env_steps_sampled_throughput_per_sec", 0)
        num_workers = result.get("num_healthy_workers", 0)
        
        # Console line you can actually see
        rew_str = "None" if rew is None else f"{rew:.3f}"
        print(f"[{it:04d}] steps={steps:,} eps_iter={eps_this_iter} episodes={total_episodes} "
              f"reward_mean={rew_str} throughput={steps_sec:.1f} steps/s workers={num_workers} "
              f"zero_conf_streak={zero_conflict_streak}", flush=True)

        # Read trajectory.csv and compute diagnostics from current results directory
        traj_files = []
        try:
            # Look for trajectory files in current timestamped results directory
            if os.path.isdir(results_dir):
                for f in os.listdir(results_dir):
                    if f.startswith("traj_ep") and f.endswith(".csv"):
                        traj_files.append(os.path.join(results_dir, f))
            
            if traj_files:
                # Get the most recent trajectory file
                latest_traj = max(traj_files, key=os.path.getmtime)
                df = pd.read_csv(latest_traj)
                if not df.empty and "episode_id" in df.columns:
                    last_ep = df["episode_id"].max()
                    ep_data = df[df["episode_id"] == last_ep]
                    avg_min_sep = ep_data["min_separation_nm"].mean()
                    
                    # Debug: print available columns
                    if it == 1:  # Only print for first iteration
                        print(f"Debug: CSV columns available: {list(df.columns)}")
                    
                    # Better waypoint hit calculation using waypoint_reached flag
                    if "waypoint_reached" in ep_data.columns:
                        # Count unique agents who reached waypoint in this episode
                        agents_reached = ep_data[ep_data["waypoint_reached"] == 1]["agent_id"].nunique()
                        wp_hits = agents_reached
                        
                        # Additional debug info
                        if it == 1:
                            reached_flags = ep_data["waypoint_reached"].sum()
                            print(f"Debug: waypoint_reached flags: {reached_flags}, unique agents reached: {wp_hits}")
                    else:
                        # Fallback: count unique agents who were within 5 NM at any point
                        agents_near_wp = ep_data[ep_data["wp_dist_nm"] <= 5.0]["agent_id"].nunique()
                        wp_hits = agents_near_wp
                        print(f"Debug: Using fallback wp_hits calculation: {wp_hits}")
                    
                    rt_mean = float(ep_data.get("reward_team", pd.Series([0])).mean())
                    td_mean = float(ep_data.get("team_dphi", pd.Series([0])).mean())
                    print(f"Last episode {last_ep}: avg_min_sep={avg_min_sep:.2f}, wp_hits={wp_hits}, "
                          f"reward_team_mean={rt_mean:.6f}, team_dphi_mean={td_mean:.6f}")
                else:
                    print("Debug: No valid episode data found in trajectory file")
            else:
                print(f"Debug: No trajectory files found in {results_dir}")
        except Exception as e:
            print(f"Warning: could not read trajectory files: {e}")
            import traceback
            traceback.print_exc()

        # Append to CSV
        with open(progress_csv, "a", newline="") as f:
            csv.writer(f).writerow([int(time.time()), it, steps, total_episodes, rew, zero_conflict_streak])

        # Save checkpoint only when reward improves significantly (less negative = better)
        current_reward = float(rew if isinstance(rew, (int, float)) else float('-inf'))
        improvement_threshold = 5.0  # Only save if reward improves by at least 50 points
        if current_reward > (best_reward + improvement_threshold):
            best_reward = current_reward
            models_dir = os.path.join(results_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            ckpt = algo_obj.save(models_dir)
            print(f"🎯 New best reward {current_reward:.1f}! Saved checkpoint: {getattr(ckpt, 'checkpoint', ckpt)}")
        
        # Also save periodic backup every 100k steps
        ckpt_band = steps // checkpoint_every
        models_dir = os.path.join(results_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        have_ckpts = len([f for f in os.listdir(models_dir) if f.startswith("periodic_")])
        if ckpt_band > have_ckpts:
            ckpt = algo_obj.save(os.path.join(models_dir, f"periodic_{steps//1000}k"))
            print(f"📁 Periodic backup: {getattr(ckpt, 'checkpoint', ckpt)}")

        if zero_conflict_streak >= 100:
            print("Early stop: 100 conflict-free episodes.")
            break

    # Final save in timestamped directory
    models_dir = os.path.join(results_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    ckpt = algo_obj.save(models_dir)
    final_ckpt = ckpt if isinstance(ckpt, str) else str(ckpt)
    print("Final checkpoint at:", final_ckpt)
    print(f"All results saved in: {results_dir}")
    return final_ckpt


if __name__ == "__main__":
    import sys
    REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Accept scenario name from command line args or environment variable
    scenario_name = "head_on"  # Default to head_on scenario
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
    elif "SCENARIO" in os.environ:
        scenario_name = os.environ["SCENARIO"]
    
    print(f"Training with scenario: {scenario_name}")
    ckpt = train_frozen(REPO, algo=os.environ.get("ALGO", "PPO"), scenario_name=scenario_name)
    print("Final checkpoint:", ckpt)
