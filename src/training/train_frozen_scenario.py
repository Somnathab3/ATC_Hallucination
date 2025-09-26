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
    """Check and report GPU availability."""
    print(f"🔍 GPU Detection:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    return torch.cuda.is_available()

# Initialize Ray with better configuration to reduce warnings
def init_ray(use_gpu: bool = False):
    """Initialize Ray with optimized settings and GPU support."""
    if not ray.is_initialized():
        init_kwargs = {
            "log_to_driver": False,
            "configure_logging": False,
            "ignore_reinit_error": True,  # Allow reinit if needed
        }
        
        if use_gpu and torch.cuda.is_available():
            # Use distributed mode with GPU - note: Ray init doesn't need num_gpus for basic setup
            print(f"🚀 Initializing Ray with GPU support")
        else:
            # Use local mode to avoid worker conflicts with BlueSky (CPU-only)
            init_kwargs["local_mode"] = True
            print("🔧 Initializing Ray in local mode (CPU-only)")
        
        ray.init(**init_kwargs)


def make_env(env_config: Dict[str, Any]):
    """Factory for RLlib to build env instances (use ParallelPettingZooEnv wrapper)."""
    return ParallelPettingZooEnv(MARLCollisionEnv(env_config))


def train_frozen(repo_root: str,
                 algo: str = "PPO",
                 seed: int = 42,
                 scenario_name: str = "head_on",
                 timesteps_total: int = 2_000_000,
                 checkpoint_every: int = 100_000,
                 use_gpu: Optional[bool] = None) -> str:
    """
    Returns: path to final checkpoint.
    
    Args:
        use_gpu: If None, auto-detect GPU. If True, force GPU. If False, force CPU.
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
    
    # PERFORMANCE OPTIMIZATIONS APPLIED:
    # - Multiple workers (4-6 depending on algorithm and GPU availability)
    # - Larger batch sizes (16K-20K with GPU vs 4K-8K without)
    # - Higher learning rates with GPU (5e-4 to 8e-4 vs 3e-4)
    # - Larger neural networks with GPU (512x512 vs 256x256)
    # - Multiple learner workers for parallel gradient updates
    # - Increased buffer sizes and training intensity
    # - Optimized fragment lengths and environment parallelization
    print(f"🚀 AGGRESSIVE PERFORMANCE MODE: {'GPU-Accelerated' if use_gpu else 'CPU-Optimized'}")

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
        "log_trajectories": False,
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
        "drift_penalty_per_sec": -0.1,
        "progress_reward_per_km": 0.02,
        "backtrack_penalty_per_km": -0.1,
        "time_penalty_per_sec": -0.0005,
        "reach_reward": 10.0,  # Fixed: was 50.0, should match environment
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
                  .framework("torch", torch_compile_learner=False)  # Disable compilation for stability
                  .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
                  .env_runners(
                      # AGGRESSIVE: Multiple workers for maximum throughput
                      num_env_runners=4 if use_gpu else 2,  # More workers with GPU
                      num_envs_per_env_runner=2,  # Multiple envs per worker
                      rollout_fragment_length=400 if use_gpu else 200,  # Larger fragments with GPU
                      max_requests_in_flight_per_env_runner=2,  # Higher parallelism
                      num_cpus_per_env_runner=1,
                      num_gpus_per_env_runner=0.1 if use_gpu else 0,  # Share GPU with workers
                      batch_mode="complete_episodes",  # More efficient batching
                  )
                  .training(
                      gamma=0.995,              # Slightly higher gamma for better long-term planning
                      lr=5e-4 if use_gpu else 3e-4,  # Higher LR with GPU for faster learning
                      # AGGRESSIVE: Much larger batches with GPU
                      train_batch_size=16384 if use_gpu else 8192,
                      num_epochs=15 if use_gpu else 10,  # More epochs with GPU
                      model={"fcnet_hiddens": [512, 512] if use_gpu else [256, 256],  # Larger network with GPU
                             "fcnet_activation": "tanh",
                             "free_log_std": False},
                      grad_clip=10.0,
                      # AGGRESSIVE: Faster updates with more frequent training
                      # More frequent training updates
                      num_sgd_iter=15 if use_gpu else 10,
                  )
                  .evaluation(
                      evaluation_interval=1,     # Evaluate every iteration to track true performance
                      evaluation_duration=10,    # More episodes for better evaluation
                      evaluation_duration_unit="episodes",
                      evaluation_parallel_to_training=False,  # Sequential to avoid conflicts
                      evaluation_num_env_runners=0,  # No separate eval workers
                      evaluation_config={"explore": False}  # Deterministic evaluation
                  )
                  .multi_agent(
                      policies=policies,
                      policy_mapping_fn=policy_mapping_fn,
                      policies_to_train=["shared_policy"],
                  )
                  .resources(
                      # AGGRESSIVE: Maximum GPU utilization
                      num_gpus=1 if use_gpu else 0,  # Main GPU for training
                      num_learner_workers=2 if use_gpu else 1,  # Multiple learner workers with GPU
                  )
                  )
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
                # FIXED: Match PPO's gamma for consistency
                gamma=0.995,
                lr=5e-4 if use_gpu else 3e-4,  # Higher LR with GPU
                # AGGRESSIVE: Much larger batches
                train_batch_size=16384 if use_gpu else 8192,
                # AGGRESSIVE: Larger network with GPU
                model={"fcnet_hiddens": [512, 512] if use_gpu else [256, 256], 
                       "fcnet_activation": "tanh", "free_log_std": False},
                grad_clip=10.0,
            )
            .env_runners(
                # AGGRESSIVE: Multiple workers for SAC throughput
                num_env_runners=3 if use_gpu else 1,  # More workers with GPU
                num_envs_per_env_runner=2,  # Multiple envs per worker
                rollout_fragment_length=400 if use_gpu else 200,  # Larger fragments
                batch_mode="truncate_episodes",     # OK for off-policy
                create_env_on_local_worker=True,
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0.1 if use_gpu else 0,  # Share GPU
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
        
        # AGGRESSIVE: Optimized SAC parameters for speed
        config["tau"] = 0.01 if use_gpu else 0.005  # Faster updates with GPU
        config["target_entropy"] = "auto"
        config["initial_alpha"] = 0.1
        # AGGRESSIVE: Minimal warmup for faster startup
        config["num_steps_sampled_before_learning_starts"] = 2_000 if use_gpu else 5_000
        config["critic_lr"] = 5e-4 if use_gpu else 3e-4  # Higher LR with GPU
        config["alpha_lr"] = 5e-4 if use_gpu else 3e-4
        # AGGRESSIVE: More frequent learning
        config["training_intensity"] = 2.0 if use_gpu else 1.0  # Train more often
        
        # AGGRESSIVE: Larger buffer with GPU
        config["replay_buffer_config"] = {
            "type": "MultiAgentReplayBuffer",
            "buffer_size": 2_000_000 if use_gpu else 1_000_000,
            "storage_unit": "timesteps",
        }
        
        config["exploration_config"] = {"type": "StochasticSampling"}
        # Your obs are already normalized → disable external filter to avoid no-ops on Dict obs
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
                # AGGRESSIVE: IMPALA excels with many workers
                num_env_runners=6 if use_gpu else 3,  # Many workers for IMPALA
                num_envs_per_env_runner=2,  # Multiple envs per worker
                rollout_fragment_length=300 if use_gpu else 200,  # Larger fragments
                batch_mode="truncate_episodes",
                create_env_on_local_worker=True,
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0.05 if use_gpu else 0,  # Small GPU share
            )
            .training(
                # FIXED: Match PPO's gamma
                gamma=0.995,
                lr=8e-4 if use_gpu else 3e-4,  # Higher LR for IMPALA with GPU
                # AGGRESSIVE: Large batches for IMPALA
                train_batch_size=20480 if use_gpu else 8192,
                # AGGRESSIVE: Larger network
                model={"fcnet_hiddens": [512, 512] if use_gpu else [256, 256], 
                       "fcnet_activation": "tanh", "free_log_std": False},
                grad_clip=10.0,
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
                num_learner_workers=3 if use_gpu else 1,  # Multiple learner workers
            )
        )
        
        # AGGRESSIVE: IMPALA parameters for maximum throughput
        config["vtrace"] = True
        config["vtrace_clip_rho_threshold"] = 1.0
        config["vtrace_clip_pg_rho_threshold"] = 1.0
        config["entropy_coeff"] = 0.02 if use_gpu else 0.01  # Higher exploration with GPU
        config["vf_loss_coeff"] = 0.5
        # AGGRESSIVE: Larger minibatches
        config["minibatch_size"] = 1024 if use_gpu else 512
        config["num_epochs"] = 2 if use_gpu else 1  # More epochs with GPU
        config["shuffle_buffer_size"] = 2048 if use_gpu else 1024  # Larger shuffle buffer
        
        algo_obj = IMPALA(config=config)
    elif algo.upper() == "CQL":
        config = (
            CQLConfig()
            .environment(env=env_name, env_config=env_config)
            .framework("torch")
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .env_runners(
                # AGGRESSIVE: Multiple workers for CQL
                num_env_runners=2 if use_gpu else 0,  # Some workers with GPU
                num_envs_per_env_runner=2 if use_gpu else 1,
                rollout_fragment_length=400 if use_gpu else 200,  # Larger fragments
                batch_mode="truncate_episodes",
                create_env_on_local_worker=True,
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0.1 if use_gpu else 0,
            )
            .training(
                # FIXED: Match PPO's gamma
                gamma=0.995, 
                lr=5e-4 if use_gpu else 3e-4,  # Higher LR with GPU
                # AGGRESSIVE: Large batches for CQL
                train_batch_size=16384 if use_gpu else 8192,
                # AGGRESSIVE: Larger network
                model={"fcnet_hiddens": [512, 512] if use_gpu else [256, 256], 
                       "fcnet_activation": "tanh", "free_log_std": False},
                grad_clip=10.0,
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
                num_learner_workers=2 if use_gpu else 1,  # Multiple learner workers
            )
        )
        
        # AGGRESSIVE: CQL parameters optimized for speed
        config["tau"] = 0.01 if use_gpu else 0.005  # Faster updates
        config["n_step"] = 1
        config["target_entropy"] = "auto"
        config["initial_alpha"] = 0.1
        config["critic_lr"] = 8e-4 if use_gpu else 3e-4  # Higher LR with GPU
        # AGGRESSIVE: Less conservative, faster learning
        config["min_q_weight"] = 3.0 if use_gpu else 5.0  # Even less conservative with GPU
        config["num_steps_sampled_before_learning_starts"] = 1_000 if use_gpu else 5_000
        # AGGRESSIVE: Larger buffer and more frequent updates
        config["training_intensity"] = 2.0 if use_gpu else 1.0
        config["replay_buffer_config"] = {
            "type": "MultiAgentReplayBuffer",
            "buffer_size": 3_000_000 if use_gpu else 1_000_000,  # Huge buffer with GPU
            "storage_unit": "timesteps",
        }
        
        # Online training (from env) by default; if you create an offline dataset later:
        # config["input"] = "dataset" ; config["input_config"] = {"paths": ["path/to/*.json"]}
        algo_obj = CQL(config=config)
    elif algo.upper() == "APPO":
        config = (
            APPOConfig()
            .environment(env=env_name, env_config=env_config)
            .framework("torch")
            # keep old API to match the rest of your code
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
                train_batch_size=8192 if use_gpu else 4096,
                grad_clip=10.0,
                model={
                    "fcnet_hiddens": [256, 256],
                    "vf_share_layers": False,
                    "use_lstm": True,                 # set False if you don't need memory
                    "lstm_cell_size": 128,
                    "max_seq_len": 20,                # 20 steps = 200 s at your 10 s action step
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
    MIN_BANDS_BEFORE_ESTOP = int(os.environ.get("MIN_BANDS_BEFORE_ESTOP", 3))  # wait ≥3 bands
    GOOD_BANDS_TO_STOP = int(os.environ.get("GOOD_BANDS_TO_STOP", 2))          # need 2 good bands
    BAND_ZCS_TARGET = int(os.environ.get("BAND_ZCS_TARGET", 20))               # ≥20 CF episodes in band
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

        # Save checkpoint only when reward improves significantly (less negative = better)
        current_reward = float(rew if isinstance(rew, (int, float)) else float('-inf'))
        improvement_threshold = 5.0  # Only save if reward improves by at least 5 points
        if current_reward > (best_reward + improvement_threshold):
            best_reward = current_reward
            temp_models_dir = os.path.join(results_dir, "checkpoints")
            os.makedirs(temp_models_dir, exist_ok=True)
            ckpt = algo_obj.save(temp_models_dir)
            checkpoint_name = f"{algo}_{scenario_base_name}_best_{timestamp}"
            print(f"🎯 New best reward {current_reward:.1f}! Saved checkpoint: {checkpoint_name}")
        
        # Also save periodic backup every 100k steps
        ckpt_band = steps // checkpoint_every
        temp_models_dir = os.path.join(results_dir, "checkpoints")
        os.makedirs(temp_models_dir, exist_ok=True)
        have_ckpts = len([f for f in os.listdir(temp_models_dir) if f.startswith("periodic_")])
        if ckpt_band > have_ckpts:
            ckpt = algo_obj.save(os.path.join(temp_models_dir, f"periodic_{algo}_{scenario_base_name}_{steps//1000}k"))
            print(f"📁 Periodic backup: {algo}_{scenario_base_name}_{steps//1000}k")

        # Disabled old immediate early stop to prevent premature exits
        # if zero_conflict_streak >= 100:
        #     print("Early stop: 100 conflict-free episodes.")
        #     break

    # Final save in main models directory with clean structure
    final_models_dir = os.path.join(repo_root, "models", f"{algo}_{scenario_base_name}_{timestamp}")
    os.makedirs(final_models_dir, exist_ok=True)
    ckpt = algo_obj.save(final_models_dir)
    final_ckpt = ckpt if isinstance(ckpt, str) else str(ckpt)
    final_checkpoint_name = f"{algo}_{scenario_base_name}_{timestamp}"
    print(f"🏆 Final model saved at: ./models/{final_checkpoint_name}")
    print(f"📊 Training results saved in: {results_dir}")
    if use_gpu:
        print(f"⚡ Training completed using GPU acceleration")
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
