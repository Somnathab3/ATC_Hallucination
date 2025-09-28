"""
Multi-Agent Reinforcement Learning Training for Air Traffic Control.

This module provides comprehensive training capabilities for MARL collision avoidance
policies using frozen scenario configurations. The training framework supports
multiple state-of-the-art algorithms with optimized hyperparameters and early
stopping based on conflict-free performance.

Key Features:
- Parameter sharing: Shared policy architecture across all agents for efficient learning
- Multi-algorithm support: PPO, SAC, IMPALA, CQL, APPO with algorithm-specific tuning
- Adaptive early stopping: Band-based performance evaluation prevents premature termination
- GPU acceleration: Automatic detection and optimized configurations for CUDA devices
- Progress tracking: Comprehensive logging of training metrics and trajectory analysis
- Checkpoint management: Automatic saving of best-performing models during training

Training Philosophy:
The trainer uses a unified reward system that eliminates double-counting penalties through:
- Signed progress rewards (positive for advancement, negative for backtracking)
- Unified well-clear violation penalties (entry + severity-scaled step penalties)
- Drift improvement shaping (rewards heading optimization, not absolute penalties)
- Enhanced team-based potential-based reward shaping (PBRS) with 5 NM sensitivity
Training continues until stable conflict-free performance is achieved or maximum timesteps are reached.

The framework is designed for reproducible academic research with detailed logging
and configurable hyperparameters for systematic ablation studies.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import csv
import ray
import torch

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.impala import IMPALA, IMPALAConfig
from ray.rllib.algorithms.cql import CQL, CQLConfig
from ray.rllib.algorithms.appo import APPO, APPOConfig
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

def check_gpu_availability():
    """
    Detect and report available GPU resources for training acceleration.
    
    Returns:
        bool: True if CUDA-compatible GPUs are available, False otherwise
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU available: {gpu_count} device(s) detected")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU detected, using CPU")
    return torch.cuda.is_available()

# Initialize Ray with better configuration to reduce warnings
def init_ray(use_gpu: bool = False):
    """
    Initialize Ray distributed computing framework with optimized configuration.
    
    Args:
        use_gpu: Whether to enable GPU support for training acceleration
    """
    if not ray.is_initialized():
        init_kwargs = {
            "log_to_driver": False,
            "configure_logging": False,
            "ignore_reinit_error": True,  # Allow reinit if needed
        }
        
        if use_gpu and torch.cuda.is_available():
            print("Initializing Ray with GPU support")
        else:
            init_kwargs["local_mode"] = True
            print("Initializing Ray in local mode (CPU-only)")
        
        ray.init(**init_kwargs)


def make_env(env_config: Dict[str, Any]):
    """
    Factory function for RLlib environment instantiation.
    
    Creates properly wrapped multi-agent environments compatible with RLlib's
    training pipeline using the ParallelPettingZoo wrapper.
    
    Args:
        env_config: Environment configuration dictionary
        
    Returns:
        ParallelPettingZooEnv: Wrapped environment ready for RLlib training
    """
    return ParallelPettingZooEnv(MARLCollisionEnv(env_config))


def train_frozen(repo_root: str,
                 algo: str = "PPO",
                 seed: int = 42,
                 scenario_name: str = "head_on",
                 timesteps_total: int = 2_000_000,
                 checkpoint_every: int = 100_000,
                 use_gpu: Optional[bool] = None,
                 log_trajectories: bool = False) -> str:
    """
    Train multi-agent collision avoidance policy on frozen scenario.
    
    This function orchestrates the complete training pipeline from environment
    setup through model checkpointing. Training continues until either stable
    conflict-free performance is achieved or maximum timesteps are reached.
    
    Args:
        repo_root: Path to project root directory
        algo: RL algorithm to use (PPO, SAC, IMPALA, CQL, APPO)
        seed: Random seed for reproducible training
        scenario_name: Name of scenario file (without .json extension)
        timesteps_total: Maximum training timesteps before termination
        checkpoint_every: Frequency of checkpoint saves (currently unused)
        use_gpu: GPU usage preference (None=auto-detect, True=force, False=disable)
        log_trajectories: Enable detailed trajectory logging (default: False for speed)
        
    Returns:
        Path to final trained model checkpoint directory
        
    Raises:
        FileNotFoundError: If specified scenario file doesn't exist
        ValueError: If unsupported algorithm is specified
    """
    scenario_path = os.path.join(repo_root, "scenarios", f"{scenario_name}.json")
    if not os.path.exists(scenario_path):
        raise FileNotFoundError(f"Scenario '{scenario_name}' not found at {scenario_path}. Available scenarios: head_on, t_formation, parallel, converging, canonical_crossing")

    # Check GPU availability and initialize Ray
    gpu_available = check_gpu_availability()
    
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = gpu_available  # Auto-detect
    elif use_gpu and not gpu_available:
        print("⚠️  GPU requested but not available. Falling back to CPU.")
        use_gpu = False
    
    print(f"🎯 Training mode: {'GPU' if use_gpu else 'CPU'}")
    
    # Initialize Ray with optimized settings
    init_ray(use_gpu=use_gpu)
    
    print(f"Training mode: {'GPU-Accelerated' if use_gpu else 'CPU-Optimized'}")

    # Register env once
    env_name = "marl_collision_env_v0"
    register_env(env_name, lambda cfg: ParallelPettingZooEnv(MARLCollisionEnv(cfg)))

    # Setup timestamped results directory with algo and scenario info under ./training/
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    scenario_base_name = os.path.splitext(os.path.basename(scenario_path))[0]  # Extract scenario name without extension
    training_dir = os.path.join(repo_root, "training")
    os.makedirs(training_dir, exist_ok=True)
    results_dir = os.path.join(training_dir, f"results_{algo}_{scenario_base_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    env_config = {
        "scenario_path": scenario_path,
        "action_delay_steps": 0,
        "max_episode_steps": 100,
        "separation_nm": 5.0,
        "log_trajectories": log_trajectories,  # Enable/disable detailed logging (controllable via CLI)
        "seed": seed,
        "results_dir": os.path.abspath(results_dir),  # Pass absolute path of timestamped results directory

        # Observation configuration
        "neighbor_topk": 3,
        
        # Collision and conflict settings
        "collision_nm": 3.0,

        # === UNIFIED REWARD SYSTEM ===
        
        # Enhanced team coordination (PBRS) with 5 NM sensitivity
        "team_coordination_weight": 0.6,           # Increased coordination signal strength
        "team_gamma": 0.99,
        "team_share_mode": "responsibility",        # Share rewards based on agent responsibility
        "team_ema": 0.05,                          # Faster team phi response
        "team_cap": 0.05,                          # Higher team reward magnitude cap
        "team_anneal": 1.0,
        "team_neighbor_threshold_km": 10.0,
        
        # Signed progress reward (unified forward/backward movement)
        "progress_reward_per_km": 0.04,            # Positive for progress, negative for backtracking
        
        # Unified well-clear violation system (no double-counting)
        "violation_entry_penalty": -25.0,          # One-time penalty on separation violation
        "violation_step_scale": -1.0,              # Per-step penalty scaled by severity
        "deep_breach_nm": 1.0,                     # Steeper scaling for close approaches
        
        # Drift improvement shaping (rewards heading optimization)
        "drift_improve_gain": 0.01,                # Reward per degree of drift reduction
        "drift_deadzone_deg": 8.0,                 # Deadzone prevents oscillation penalties
        
        # Other individual reward components
        "time_penalty_per_sec": -0.0005,           # Efficiency incentive
        "reach_reward": 10.0,                       # Waypoint achievement bonus
        "action_cost_per_unit": -0.01,             # Cost for non-neutral actions
        "terminal_not_reached_penalty": -10.0,     # Penalty for episode termination without goal
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
                  .framework("torch")  # Disable torch compilation to avoid GPU indexing issues
                  .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                  .env_runners(
                      num_env_runners=4 if use_gpu else 1,  # Increased for faster execution
                      num_envs_per_env_runner=1,
                      rollout_fragment_length=200,
                      sample_timeout_s=300.0,
                      max_requests_in_flight_per_env_runner=1,
                      num_cpus_per_env_runner=2,  # Increased for more parallelism
                      num_gpus_per_env_runner=0,
                      batch_mode="truncate_episodes",
                  )
                  .training(
                      gamma=0.995,
                      lr=5e-4 if use_gpu else 3e-4,
                      train_batch_size=8192 if use_gpu else 4096,
                      num_epochs=10 if use_gpu else 8,
                      model={"fcnet_hiddens": [256, 256],
                             "fcnet_activation": "tanh",
                             "free_log_std": False},
                      grad_clip=1.0,
                      num_sgd_iter=4,
                  )
                  .evaluation(
                      evaluation_interval=5,
                      evaluation_duration=5,
                      evaluation_duration_unit="episodes",
                      evaluation_parallel_to_training=False,
                      evaluation_num_env_runners=0,
                      evaluation_config={
                          "explore": False,
                          "seed": 12345,
                      }
                  )
                  .multi_agent(
                      policies=policies,
                      policy_mapping_fn=policy_mapping_fn,
                      policies_to_train=["shared_policy"],
                  )
                  .resources(
                      num_gpus=1 if use_gpu else 0,
                  )
                  .learners(
                      num_learners=0,
                      num_cpus_per_learner=1,
                  )
                  )
                # Add PPO-specific hyperparameters
        config["kl_coeff"] = 0.2  # KL divergence coefficient for stability
        config["entropy_coeff"] = 0.01  # Entropy coefficient for exploration
        config["vf_loss_coeff"] = 1.0  # Value function coefficient
        config["clip_param"] = 0.1  # Smaller clipping for stability
        # Don't normalize actions as we handle scaling in the environment
        # Set additional config parameters
        from ray.rllib.algorithms.ppo import PPO
        algo_obj = PPO(config=config)  # Build the algorithm
    elif algo.upper() == "SAC":
        from ray.rllib.algorithms.sac import SAC, SACConfig

        config = (
            SACConfig()
            .environment(env=env_name, env_config=env_config)
            .framework("torch")
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .training(
                gamma=0.995,
                lr=5e-4 if use_gpu else 3e-4,
                train_batch_size=8192 if use_gpu else 4096,
                model={"fcnet_hiddens": [256, 256], 
                       "fcnet_activation": "tanh", "free_log_std": False},
                grad_clip=5.0,
            )
            .env_runners(
                num_env_runners=4 if use_gpu else 1,  # Increased for faster execution
                num_envs_per_env_runner=1,
                rollout_fragment_length=200,  # Fixed: Changed from 'auto' to integer
                batch_mode="truncate_episodes",
                create_env_on_local_worker=True,
                num_cpus_per_env_runner=2,  # Increased for more parallelism
                num_gpus_per_env_runner=0,
            )
            .evaluation(
                # FIXED: Add evaluation like PPO
                evaluation_interval=1,
                evaluation_duration=10,
                evaluation_duration_unit="episodes",
                evaluation_parallel_to_training=False,
                evaluation_num_env_runners=0,
                evaluation_config={"explore": False}
            )
            .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn,
                         policies_to_train=["shared_policy"])
            .resources(num_gpus=1 if use_gpu else 0)
        )
        
        config["tau"] = 0.01 if use_gpu else 0.005
        config["target_entropy"] = "auto"
        config["initial_alpha"] = 0.1
        config["num_steps_sampled_before_learning_starts"] = 2_000 if use_gpu else 5_000
        config["critic_lr"] = 5e-4 if use_gpu else 3e-4
        config["alpha_lr"] = 5e-4 if use_gpu else 3e-4
        config["training_intensity"] = 1.5 if use_gpu else 1.0
        
        config["replay_buffer_config"] = {
            "type": "MultiAgentReplayBuffer",
            "buffer_size": 1_000_000 if use_gpu else 500_000,
            "storage_unit": "timesteps",
        }
        
        config["exploration_config"] = {"type": "StochasticSampling"}
        # obs are already normalized → disable external filter to avoid no-ops on Dict obs
        config["observation_filter"] = "NoFilter"
        config["synchronize_filters"] = False
        
        algo_obj = SAC(config=config)
    elif algo.upper() == "IMPALA":
        from ray.rllib.algorithms.impala import IMPALA
        
        config = (
            IMPALAConfig()
            .environment(env=env_name, env_config=env_config)
            .framework("torch")
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .env_runners(
                num_env_runners=2 if use_gpu else 1,
                num_envs_per_env_runner=1,
                rollout_fragment_length=50,  # Fixed: Changed from 'auto' to integer
                batch_mode="truncate_episodes",
                create_env_on_local_worker=True,
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0,
            )
            .training(
                gamma=0.995,
                lr=6e-4 if use_gpu else 3e-4,
                train_batch_size=8192 if use_gpu else 4096,
                model={"fcnet_hiddens": [256, 256], 
                       "fcnet_activation": "tanh", "free_log_std": False},
                grad_clip=5.0,
            )
            .evaluation(
                # FIXED: Add evaluation like PPO
                evaluation_interval=1,
                evaluation_duration=10,
                evaluation_duration_unit="episodes",
                evaluation_parallel_to_training=False,
                evaluation_num_env_runners=0,
                evaluation_config={"explore": False}
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=["shared_policy"],
            )
            .resources(
                num_gpus=1 if use_gpu else 0,
            )
            .learners(
                num_learners=1 if use_gpu else 1,
            )
        )
        
        config["vtrace"] = True
        config["vtrace_clip_rho_threshold"] = 1.0
        config["vtrace_clip_pg_rho_threshold"] = 1.0
        config["entropy_coeff"] = 0.01 if use_gpu else 0.01
        config["vf_loss_coeff"] = 0.5
        config["minibatch_size"] = 512 if use_gpu else 256
        config["num_epochs"] = 1 if use_gpu else 1
        config["shuffle_buffer_size"] = 1024 if use_gpu else 512
        
        algo_obj = IMPALA(config=config)
    elif algo.upper() == "CQL":
        config = (
            CQLConfig()
            .environment(env=env_name, env_config=env_config)
            .framework("torch")
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .env_runners(
                num_env_runners=1 if use_gpu else 0,
                num_envs_per_env_runner=1 if use_gpu else 1,
                rollout_fragment_length=200,  # Fixed: Changed from 'auto' to integer
                batch_mode="truncate_episodes",
                create_env_on_local_worker=True,
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0,
            )
            .training(
                gamma=0.995, 
                lr=5e-4 if use_gpu else 3e-4,
                train_batch_size=8192 if use_gpu else 4096,
                model={"fcnet_hiddens": [256, 256], 
                       "fcnet_activation": "tanh", "free_log_std": False},
                grad_clip=5.0,
            )
            .evaluation(
                # FIXED: Add evaluation like PPO
                evaluation_interval=1,
                evaluation_duration=10,
                evaluation_duration_unit="episodes",
                evaluation_parallel_to_training=False,
                evaluation_num_env_runners=0,
                evaluation_config={"explore": False}
            )
            .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn,
                         policies_to_train=["shared_policy"])
            .resources(
                num_gpus=1 if use_gpu else 0,
            )
            .learners(
                num_learners=1 if use_gpu else 1,
            )
        )
        
        config["tau"] = 0.01 if use_gpu else 0.005
        config["n_step"] = 1
        config["target_entropy"] = "auto"
        config["initial_alpha"] = 0.1
        config["critic_lr"] = 6e-4 if use_gpu else 3e-4
        config["min_q_weight"] = 4.0 if use_gpu else 5.0
        config["num_steps_sampled_before_learning_starts"] = 1_000 if use_gpu else 5_000
        config["training_intensity"] = 1.5 if use_gpu else 1.0
        config["replay_buffer_config"] = {
            "type": "MultiAgentReplayBuffer",
            "buffer_size": 1_000_000 if use_gpu else 500_000,
            "storage_unit": "timesteps",
        }

        algo_obj = CQL(config=config)
    elif algo.upper() == "APPO":
        config = (
            APPOConfig()
            .environment(env=env_name, env_config=env_config)
            .framework("torch")
            # keep old API to match the rest of  code
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .env_runners(
                num_env_runners=1,                     # start conservative with BlueSky
                num_envs_per_env_runner=1,
                rollout_fragment_length=100,           # reduce if actors feel too stale
                batch_mode="truncate_episodes",
                create_env_on_local_worker=True,
            )
            .training(
                gamma=0.995,
                lr=3e-4,
                train_batch_size=4096 if use_gpu else 2048,
                grad_clip=5.0,
                model={
                    "fcnet_hiddens": [128, 128],
                    "vf_share_layers": False,
                    "use_lstm": False,
                    "fcnet_activation": "relu", 
                    "free_log_std": False
                },
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=["shared_policy"],
            )
            .resources(num_gpus=1 if use_gpu else 0)
        )
        config["entropy_coeff"] = 0.001
        algo_obj = APPO(config=config)
    else:
        raise ValueError("algo must be 'PPO', 'SAC', 'IMPALA', 'CQL', or 'APPO'")

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
    
    # Band-based early stopping parameters
    BAND_STEPS = int(os.environ.get("BAND_STEPS", 8200))
    MIN_BANDS_BEFORE_ESTOP = int(os.environ.get("MIN_BANDS_BEFORE_ESTOP", 3))
    GOOD_BANDS_TO_STOP = int(os.environ.get("GOOD_BANDS_TO_STOP", 2))
    BAND_ZCS_TARGET = int(os.environ.get("BAND_ZCS_TARGET", 20))
    next_band = BAND_STEPS
    band_idx = 0
    band_zcs = 0
    band_eps = 0
    good_bands = 0
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
                pass  # Silent debug
            pass

        zero_conflict_streak = zero_conflict_streak + 1 if conflict_free_ep else 0
        
        # Update band counters
        band_eps += episodes_this_iter
        band_zcs += int(conflict_free_ep)

        # Debug info for conflict detection (disabled)
        # if it == 1:  # Only print for first iteration
        #     print(f"Debug: conflict_free_ep={conflict_free_ep}, zero_conflict_streak={zero_conflict_streak}")

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
        
        # Band boundary check and early stopping
        if steps >= next_band:
            band_idx += 1
            print(f"=== BAND {band_idx} @ {steps} steps: conflict-free in band = {band_zcs}/{band_eps} ===")
            # Save a band-aligned checkpoint
            temp_models_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(temp_models_dir, exist_ok=True)
            algo_obj.save(os.path.join(temp_models_dir, f"band_{band_idx}_{steps}"))

            # Band "goodness" & reset
            good_bands = good_bands + 1 if band_zcs >= BAND_ZCS_TARGET else 0
            band_zcs = 0
            band_eps = 0
            next_band += BAND_STEPS

            # Banded early-stop (prevents premature stop)
            if band_idx >= MIN_BANDS_BEFORE_ESTOP and good_bands >= GOOD_BANDS_TO_STOP:
                print("Early stop (banded): stable conflict-free performance.")
                break

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
                    
                    # Debug: print available columns (disabled)
                    # if it == 1:  # Only print for first iteration
                    #     print(f"Debug: CSV columns available: {list(df.columns)}")
                    
                    # Better waypoint hit calculation using waypoint_reached flag
                    if "waypoint_reached" in ep_data.columns:
                        # Count unique agents who reached waypoint in this episode
                        agents_reached = ep_data[ep_data["waypoint_reached"] == 1]["agent_id"].nunique()
                        wp_hits = agents_reached
                        
                        # Additional debug info (disabled)
                        # if it == 1:
                        #     reached_flags = ep_data["waypoint_reached"].sum()
                        #     print(f"Debug: waypoint_reached flags: {reached_flags}, unique agents reached: {wp_hits}")
                    else:
                        # Fallback: count unique agents who were within 5 NM at any point
                        agents_near_wp = ep_data[ep_data["wp_dist_nm"] <= 5.0]["agent_id"].nunique()
                        wp_hits = agents_near_wp
                        # print(f"Debug: Using fallback wp_hits calculation: {wp_hits}")
                    
                    rt_mean = float(ep_data.get("reward_team", pd.Series([0])).mean())
                    td_mean = float(ep_data.get("team_dphi", pd.Series([0])).mean())
                    print(f"Last episode {last_ep}: avg_min_sep={avg_min_sep:.2f}, wp_hits={wp_hits}, "
                          f"reward_team_mean={rt_mean:.6f}, team_dphi_mean={td_mean:.6f}")
                else:
                    pass  # print("Debug: No valid episode data found in trajectory file")
            else:
                pass  # print(f"Debug: No trajectory files found in {results_dir}")
        except Exception as e:
            print(f"Warning: could not read trajectory files: {e}")
            import traceback
            traceback.print_exc()

        # Append to CSV
        with open(progress_csv, "a", newline="") as f:
            csv.writer(f).writerow([int(time.time()), it, steps, total_episodes, rew, zero_conflict_streak])

        # Save checkpoint when reward improves significantly
        current_reward = float(rew if isinstance(rew, (int, float)) else float('-inf'))
        improvement_threshold = 2.0
        
        if current_reward > (best_reward + improvement_threshold):
            temp_models_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(temp_models_dir, exist_ok=True)
            
            # Clean up previous best checkpoints
            for old_ckpt in os.listdir(temp_models_dir):
                if old_ckpt.startswith("best_"):
                    import shutil
                    shutil.rmtree(os.path.join(temp_models_dir, old_ckpt), ignore_errors=True)
            
            best_reward = current_reward
            ckpt = algo_obj.save(os.path.join(temp_models_dir, f"best_{steps//1000}k_r{current_reward:.1f}"))
            checkpoint_name = f"{algo}_{scenario_base_name}_best_r{current_reward:.1f}_{timestamp}"
            print(f"NEW BEST REWARD {current_reward:.1f}! Saved checkpoint: {checkpoint_name}")
            print(f"Improvement: {current_reward - (best_reward - improvement_threshold):.1f} points")


    # Final save: Only save if this is the best model we've seen
    final_models_dir = os.path.join(repo_root, "models", f"{algo}_{scenario_base_name}_{timestamp}")
    os.makedirs(final_models_dir, exist_ok=True)
    
    # Copy the best checkpoint to final location if it exists
    temp_models_dir = os.path.join(results_dir, "checkpoints")
    best_checkpoints = [f for f in os.listdir(temp_models_dir) if f.startswith("best_")] if os.path.exists(temp_models_dir) else []
    
    if best_checkpoints:
        best_ckpt_dir = os.path.join(temp_models_dir, best_checkpoints[-1])
        import shutil
        shutil.copytree(best_ckpt_dir, final_models_dir, dirs_exist_ok=True)
        final_ckpt = final_models_dir
        print(f"Best model saved from checkpoint: {best_checkpoints[-1]}")
    else:
        ckpt = algo_obj.save(final_models_dir)
        final_ckpt = ckpt if isinstance(ckpt, str) else str(ckpt)
        print(f"Final model saved (no best checkpoint found)")
    
    # Save training metadata
    metadata = {
        "algorithm": algo,
        "scenario": scenario_base_name,
        "timestamp": timestamp,
        "final_steps": steps,
        "total_episodes": total_episodes,
        "best_reward": best_reward,
        "zero_conflict_streak": zero_conflict_streak,
        "gpu_used": use_gpu,
        "training_duration_iterations": it,
    }
    
    metadata_file = os.path.join(final_models_dir, "training_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    final_checkpoint_name = f"{algo}_{scenario_base_name}_{timestamp}"
    print(f"Final model: ./models/{final_checkpoint_name}")
    print(f"Training results: {results_dir}")
    print(f"Best reward achieved: {best_reward:.2f}")
    print(f"Final conflict-free streak: {zero_conflict_streak}")
    if use_gpu:
        print(f"Training completed using GPU acceleration")
    print(f"Training metadata saved: training_metadata.json")
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
    
    # Check for GPU flag from environment
    use_gpu_env = os.environ.get("USE_GPU", "auto").lower()
    use_gpu = None if use_gpu_env == "auto" else use_gpu_env in ["true", "1", "yes"]
    
    print(f"Training with scenario: {scenario_name}")
    print(f"GPU mode: {use_gpu_env}")
    ckpt = train_frozen(REPO, algo=os.environ.get("ALGO", "PPO"), scenario_name=scenario_name, use_gpu=use_gpu)
    print("Final checkpoint:", ckpt)