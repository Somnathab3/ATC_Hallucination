"""
Module Name: resume_training.py
Description: Resume MARL training from an existing checkpoint.
             Supports all algorithms (PPO, SAC, IMPALA, CQL, APPO) with automatic
             configuration restoration and continued training progress tracking.
Author: Som
Date: 2025-10-08
"""

import os
import json
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import csv
import ray
import torch

from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.sac import SAC, SACConfig
from ray.rllib.algorithms.impala import IMPALA, IMPALAConfig
from ray.rllib.algorithms.cql import CQL, CQLConfig
from ray.rllib.algorithms.appo import APPO, APPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env

# Add project root to Python path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.environment.marl_collision_env_minimal import MARLCollisionEnv
from src.environment.marl_collision_env_generic import MARLCollisionEnvGeneric

LOGGER = logging.getLogger("resume_training")
LOGGER.setLevel(logging.INFO)


def check_gpu_availability():
    """Detect and report GPU resources."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU available: {gpu_count} device(s) detected")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU detected, using CPU")
    return torch.cuda.is_available()


def init_ray(use_gpu: bool = False):
    """Initialize Ray framework with optimized configuration."""
    if not ray.is_initialized():
        init_kwargs = {
            "log_to_driver": False,
            "configure_logging": False,
            "ignore_reinit_error": True,
        }
        
        if use_gpu and torch.cuda.is_available():
            print("Initializing Ray with GPU support")
        else:
            init_kwargs["local_mode"] = True
            print("Initializing Ray in local mode (CPU-only)")
        
        ray.init(**init_kwargs)


def make_env(env_config: Dict[str, Any]):
    """Factory function for RLlib environment instantiation."""
    try:
        env_type = env_config.get("env_type", "frozen")
        
        if env_type == "generic":
            worker_index = env_config.get("worker_index", 0)
            is_evaluation = env_config.get("in_evaluation", False)
            
            generic_config = {
                "enable_hallucination_detection": False,
                **env_config
            }
            
            if worker_index > 0 and not is_evaluation:
                generic_config["log_trajectories"] = False
                
            return ParallelPettingZooEnv(MARLCollisionEnvGeneric(generic_config))
        else:
            if "scenario_path" not in env_config:
                raise ValueError(f"scenario_path must be provided for frozen environment")
            return ParallelPettingZooEnv(MARLCollisionEnv(env_config))
            
    except Exception as e:
        print(f"Error creating environment: {e}")
        print(f"Environment config: {env_config}")
        raise


def find_checkpoint_in_dir(checkpoint_dir: str) -> Optional[str]:
    """
    Find the actual checkpoint directory within a checkpoints folder.
    
    Args:
        checkpoint_dir: Path to checkpoints directory
        
    Returns:
        Path to the actual checkpoint subdirectory or None if not found
    """
    checkpoint_path = Path(checkpoint_dir)
    
    # If the path itself is a checkpoint, return it
    if (checkpoint_path / "algorithm_state.pkl").exists():
        return str(checkpoint_path)
    
    # Look for checkpoint subdirectories
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        for item in checkpoint_path.iterdir():
            if item.is_dir():
                # Check if it's a valid checkpoint
                if (item / "algorithm_state.pkl").exists():
                    print(f"Found checkpoint: {item.name}")
                    return str(item)
                # Check for nested checkpoints (RLlib sometimes creates nested dirs)
                for subitem in item.iterdir():
                    if subitem.is_dir() and (subitem / "algorithm_state.pkl").exists():
                        print(f"Found nested checkpoint: {subitem.name}")
                        return str(subitem)
    
    return None


def extract_training_info(results_dir: str) -> Dict[str, Any]:
    """
    Extract training information from the results directory.
    
    Args:
        results_dir: Path to training results directory
        
    Returns:
        Dictionary containing training metadata
    """
    info = {
        "scenario_name": "unknown",
        "algo": "PPO",
        "env_type": "frozen",
        "current_steps": 0,
        "current_episodes": 0,
        "best_reward": float('-inf'),
    }
    
    # Extract from directory name: results_PPO_scenario_timestamp
    dir_name = Path(results_dir).name
    parts = dir_name.split("_")
    if len(parts) >= 3 and parts[0] == "results":
        info["algo"] = parts[1]
        # Scenario name is everything between algo and timestamp
        info["scenario_name"] = "_".join(parts[2:-2]) if len(parts) > 4 else parts[2]
    
    # Try to read progress CSV to get current state
    progress_csv = Path(results_dir) / "training_progress.csv"
    if progress_csv.exists():
        try:
            df = pd.read_csv(progress_csv)
            if not df.empty:
                last_row = df.iloc[-1]
                info["current_steps"] = int(last_row.get("steps_sampled", 0))
                info["current_episodes"] = int(last_row.get("episodes", 0))
                info["best_reward"] = float(last_row.get("reward_mean", float('-inf')))
                print(f"Resuming from: {info['current_steps']:,} steps, {info['current_episodes']} episodes")
                print(f"Best reward so far: {info['best_reward']:.2f}")
        except Exception as e:
            print(f"Warning: Could not read progress CSV: {e}")
    
    # Try to detect environment type from scenario name
    if info["scenario_name"].startswith("generic"):
        info["env_type"] = "generic"
    
    return info


def resume_training(
    checkpoint_dir: str,
    additional_timesteps: int = 1_000_000,
    use_gpu: Optional[bool] = None,
    override_scenario: Optional[str] = None,
    override_results_dir: Optional[str] = None
) -> str:
    """
    Resume training from an existing checkpoint.
    
    Args:
        checkpoint_dir: Path to checkpoint directory (can be the checkpoints folder or specific checkpoint)
        additional_timesteps: Additional timesteps to train
        use_gpu: GPU usage preference (None=auto-detect, True=force, False=disable)
        override_scenario: Override scenario name (if different from checkpoint)
        override_results_dir: Override results directory (if different from checkpoint parent)
        
    Returns:
        Path to final checkpoint
    """
    # Find the actual checkpoint
    checkpoint_path = find_checkpoint_in_dir(checkpoint_dir)
    if not checkpoint_path:
        raise FileNotFoundError(
            f"No valid checkpoint found in {checkpoint_dir}. "
            f"Looking for 'algorithm_state.pkl' file."
        )
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Determine results directory (parent of checkpoints folder)
    if override_results_dir:
        results_dir = override_results_dir
    else:
        results_dir = str(Path(checkpoint_path).parent.parent)
        if Path(checkpoint_path).parent.name == "checkpoints":
            results_dir = str(Path(checkpoint_path).parent.parent)
        else:
            results_dir = str(Path(checkpoint_path).parent)
    
    print(f"Results directory: {results_dir}")
    
    # Extract training info
    training_info = extract_training_info(results_dir)
    scenario_name = override_scenario or training_info["scenario_name"]
    algo = training_info["algo"]
    env_type = training_info["env_type"]
    
    print(f"\n{'='*60}")
    print(f"RESUMING TRAINING")
    print(f"{'='*60}")
    print(f"Algorithm: {algo}")
    print(f"Scenario: {scenario_name}")
    print(f"Environment Type: {env_type}")
    print(f"Current Progress: {training_info['current_steps']:,} steps")
    print(f"Additional Steps: {additional_timesteps:,}")
    print(f"Target Total: {training_info['current_steps'] + additional_timesteps:,} steps")
    print(f"{'='*60}\n")
    
    # Initialize Ray
    gpu_available = check_gpu_availability()
    if use_gpu is None:
        use_gpu = gpu_available
    elif use_gpu and not gpu_available:
        print("⚠️  GPU requested but not available. Falling back to CPU.")
        use_gpu = False
    
    print(f"🎯 Training mode: {'GPU' if use_gpu else 'CPU'}")
    init_ray(use_gpu=use_gpu)
    
    # Register environment
    env_name = "marl_collision_env_v0"
    register_env(env_name, make_env)
    
    # Load the algorithm from checkpoint FIRST to get original config
    print(f"Loading {algo} algorithm from checkpoint...")
    
    algo_classes = {
        "PPO": PPO,
        "SAC": SAC,
        "IMPALA": IMPALA,
        "CQL": CQL,
        "APPO": APPO,
    }
    
    if algo not in algo_classes:
        raise ValueError(f"Unsupported algorithm: {algo}. Supported: {list(algo_classes.keys())}")
    
    AlgoClass = algo_classes[algo]
    
    try:
        # Load algorithm from checkpoint (RLlib automatically restores config)
        algo_obj = AlgoClass.from_checkpoint(checkpoint_path)
        print(f"✅ Successfully loaded {algo} from checkpoint")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise
    
    # Extract the original environment config from loaded checkpoint
    original_env_config = algo_obj.config.get("env_config", {})
    print(f"\n📋 Original environment configuration loaded from checkpoint:")
    print(f"   Environment type: {original_env_config.get('env_type', 'frozen')}")
    print(f"   Max episode steps: {original_env_config.get('max_episode_steps', 'N/A')}")
    print(f"   Team coordination weight: {original_env_config.get('team_coordination_weight', 'N/A')}")
    print(f"   Collision threshold: {original_env_config.get('collision_nm', 'N/A')} NM")
    
    # Update only necessary fields for resumed training (keep all original hyperparameters)
    env_config = original_env_config.copy()
    
    # Update results directory to continue logging to same location
    env_config["results_dir"] = os.path.abspath(results_dir)
    
    # Enable trajectory logging for resumed training (if it was disabled)
    env_config["log_trajectories"] = True
    
    # Ensure scenario path is correct for frozen environments
    if env_config.get("env_type", "frozen") == "frozen":
        repo_root = Path(__file__).parent.parent.parent
        scenario_path = repo_root / "scenarios" / f"{scenario_name}.json"
        if scenario_path.exists():
            env_config["scenario_path"] = str(scenario_path)
            print(f"   Scenario path: {scenario_path}")
        else:
            # Keep original scenario path if new one doesn't exist
            print(f"   Using original scenario path from checkpoint")
    
    # Update the algorithm's environment config
    algo_obj.config["env_config"] = env_config
    
    print(f"\n✅ Environment configuration preserved from original training")
    
    # Setup progress tracking
    progress_csv = Path(results_dir) / "training_progress.csv"
    csv_exists = progress_csv.exists()
    
    # Training loop
    starting_steps = training_info["current_steps"]
    target_steps = starting_steps + additional_timesteps
    best_reward = training_info["best_reward"]
    
    # Get total agents
    temp_config = {**env_config, "log_trajectories": False, "enable_hallucination_detection": False}
    if env_type == "generic":
        tmp_env = MARLCollisionEnvGeneric(temp_config)
    else:
        tmp_env = MARLCollisionEnv(temp_config)
    total_num_agents = len(tmp_env.possible_agents)
    del tmp_env
    
    print(f"Total agents: {total_num_agents}")
    print(f"\nStarting training loop...")
    print(f"Target: {target_steps:,} total steps ({additional_timesteps:,} additional)\n")
    
    steps = starting_steps
    it = 0
    total_episodes = training_info["current_episodes"]
    zero_conflict_streak = 0
    perfect_episode_streak = 0
    PERFECT_STREAK_TARGET = 5
    
    while steps < target_steps:
        it += 1
        result = algo_obj.train()
        
        # Extract metrics
        rew = (result.get("env_runners", {}) or {}).get("episode_return_mean")
        if rew is None:
            rew = result.get("episode_reward_mean")
        
        steps = (result.get("env_runners", {}) or {}).get("num_env_steps_sampled") or \
                result.get("num_env_steps_sampled") or steps
        episodes_this_iter = (result.get("env_runners", {}) or {}).get("episodes_this_iter", 0)
        total_episodes += episodes_this_iter
        
        # Check for conflict-free and perfect episodes
        conflict_free_ep = False
        all_agents_reached_wp = False
        wp_hits = 0
        
        try:
            traj_files = [f for f in Path(results_dir).iterdir() if f.name.startswith("traj_ep")]
            if traj_files:
                latest = max(traj_files, key=lambda f: f.stat().st_mtime)
                df = pd.read_csv(latest)
                if not df.empty and "episode_id" in df.columns:
                    last_ep = df["episode_id"].max()
                    last_ep_data = df[df["episode_id"] == last_ep]
                    conflict_free_ep = (last_ep_data["conflict_flag"].sum() == 0)
                    
                    if "waypoint_reached" in last_ep_data.columns:
                        wp_hits = last_ep_data[last_ep_data["waypoint_reached"] == 1]["agent_id"].nunique()
                    else:
                        wp_hits = last_ep_data[last_ep_data["wp_dist_nm"] <= 5.0]["agent_id"].nunique()
                    
                    all_agents_reached_wp = (wp_hits == total_num_agents)
        except:
            pass
        
        zero_conflict_streak = zero_conflict_streak + 1 if conflict_free_ep else 0
        is_perfect_episode = conflict_free_ep and all_agents_reached_wp
        perfect_episode_streak = perfect_episode_streak + 1 if is_perfect_episode else 0
        
        # Display progress
        rew_str = "None" if rew is None else f"{rew:.3f}"
        perfect_str = f"✓ {perfect_episode_streak}/{PERFECT_STREAK_TARGET}" if is_perfect_episode else f"✗ 0/{PERFECT_STREAK_TARGET}"
        
        steps_sec = result.get("num_env_steps_sampled_throughput_per_sec", 0)
        progress_pct = ((steps - starting_steps) / additional_timesteps) * 100
        
        print(f"[{it:04d}] steps={steps:,}/{target_steps:,} ({progress_pct:.1f}%) "
              f"eps_iter={episodes_this_iter} total_eps={total_episodes} "
              f"reward={rew_str} throughput={steps_sec:.1f} steps/s "
              f"zero_conf={zero_conflict_streak} wp={wp_hits}/{total_num_agents} "
              f"perfect={perfect_str}", flush=True)
        
        # Log to CSV
        with open(progress_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                int(time.time()), it, steps, total_episodes, rew, 
                zero_conflict_streak, perfect_episode_streak
            ])
        
        # Early stopping for perfect episodes
        if perfect_episode_streak >= PERFECT_STREAK_TARGET:
            print(f"\n🎉 PERFECT TRAINING ACHIEVED!")
            print(f"   All {total_num_agents} agents reached waypoints safely for {PERFECT_STREAK_TARGET} consecutive iterations.")
            checkpoint_dir_resume = Path(results_dir) / "checkpoints"
            checkpoint_dir_resume.mkdir(exist_ok=True)
            algo_obj.save(str(checkpoint_dir_resume / f"perfect_resumed_{steps}"))
            break
        
        # Save improved checkpoints
        current_reward = float(rew if isinstance(rew, (int, float)) else float('-inf'))
        improvement_threshold = 2.0
        
        if current_reward > (best_reward + improvement_threshold):
            checkpoint_dir_resume = Path(results_dir) / "checkpoints"
            checkpoint_dir_resume.mkdir(exist_ok=True)
            
            # Clean up old best checkpoints
            for old_ckpt in checkpoint_dir_resume.glob("best_resumed_*"):
                shutil.rmtree(old_ckpt, ignore_errors=True)
            
            best_reward = current_reward
            algo_obj.save(str(checkpoint_dir_resume / f"best_resumed_{steps//1000}k_r{current_reward:.1f}"))
            print(f"NEW BEST REWARD {current_reward:.1f}! Checkpoint saved.")
    
    # Final save
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Final steps: {steps:,}")
    print(f"Final episodes: {total_episodes}")
    print(f"Best reward: {best_reward:.2f}")
    print(f"Final conflict-free streak: {zero_conflict_streak}")
    print(f"{'='*60}\n")
    
    final_checkpoint_dir = Path(results_dir) / "checkpoints"
    final_checkpoint_dir.mkdir(exist_ok=True)
    final_path = algo_obj.save(str(final_checkpoint_dir / f"resumed_final_{steps}"))
    
    print(f"Final checkpoint saved: {final_path}")
    
    return final_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Resume MARL training from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., path/to/checkpoints or path/to/checkpoints/checkpoint_000123)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1_000_000,
        help="Additional timesteps to train (default: 1,000,000)"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Force GPU usage (auto-detect by default)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU-only training"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Override scenario name (optional)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override results directory (optional)"
    )
    
    args = parser.parse_args()
    
    # Determine GPU usage
    use_gpu = None
    if args.use_gpu:
        use_gpu = True
    elif args.no_gpu:
        use_gpu = False
    
    # Resume training
    final_checkpoint = resume_training(
        checkpoint_dir=args.checkpoint,
        additional_timesteps=args.steps,
        use_gpu=use_gpu,
        override_scenario=args.scenario,
        override_results_dir=args.results_dir
    )
    
    print(f"\n✅ Training resumed and completed successfully!")
    print(f"Final checkpoint: {final_checkpoint}")
