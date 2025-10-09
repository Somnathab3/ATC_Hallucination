"""
Module Name: train_generic.py
Description: Simplified wrapper for training on generic structured conflict scenarios.
             Generates balanced mix of MERGE, CHASE, and CROSS scenarios with fixed
             4-agent configurations for cross-scenario generalization.
Author: Som
Date: 2025-10-04 (Updated: 2025-10-08)
"""

import os
import sys
from typing import Optional

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.train_frozen_scenario import train_frozen


def train_generic(repo_root: str,
                  algo: str = "PPO",
                  timesteps_total: int = 2_000_000,
                  use_gpu: Optional[bool] = None,
                  log_trajectories: bool = False,
                  **kwargs) -> str:
    """
    Train model on generic structured conflict scenarios.
    
    Wrapper around train_frozen that configures generic environment with balanced
    scenario generation across MERGE, CHASE, and CROSS conflict types. All scenarios
    use fixed 4-agent configurations (A1-A4) with consistent parameters:
    - Fixed 4 agents per scenario
    - Standard speed: 250 kt
    - Standard altitude: 10000 ft
    - Balanced distribution: merge_2x2, merge_3p1, chase_2x2, chase_3p1, cross_2x2, cross_3p1
    
    This enables the model to generalize across all three major conflict geometries
    encountered in real air traffic scenarios.
    
    Args:
        repo_root: Project root directory path.
        algo: RL algorithm (PPO, SAC, IMPALA, CQL, APPO).
        timesteps_total: Maximum training timesteps.
        use_gpu: GPU usage (None=auto, True=force, False=disable).
        log_trajectories: Enable trajectory logging.
        **kwargs: Additional training parameters.
        
    Returns:
        Path to final model checkpoint.
    """
    return train_frozen(
        repo_root=repo_root,
        algo=algo,
        seed=42,
        scenario_name="generic",
        timesteps_total=timesteps_total,
        checkpoint_every=100_000,
        use_gpu=use_gpu,
        log_trajectories=log_trajectories,
        env_type="generic",
        **kwargs
    )


if __name__ == "__main__":
    """Direct execution for quick generic training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train on generic dynamic conflicts")
    parser.add_argument("--algo", "-a", type=str, default="PPO",
                       choices=["PPO", "SAC", "IMPALA", "CQL", "APPO"],
                       help="Algorithm to use")
    parser.add_argument("--timesteps", "-t", type=int, default=1_000_000,
                       help="Number of training timesteps")
    parser.add_argument("--gpu", action="store_true",
                       help="Enable GPU training")
    parser.add_argument("--no-gpu", action="store_true",
                       help="Force CPU-only training")
    parser.add_argument("--log-trajectories", action="store_true",
                       help="Enable detailed trajectory logging")
    
    args = parser.parse_args()
    
    # Determine GPU usage
    use_gpu = None
    if args.gpu:
        use_gpu = True
    elif args.no_gpu:
        use_gpu = False
    
    print(f"🚀 Training {args.algo} on generic environment for {args.timesteps:,} timesteps")
    print(f"🎯 GPU mode: {'Enabled' if use_gpu else 'CPU-only' if use_gpu is False else 'Auto-detect'}")
    
    try:
        checkpoint = train_generic(
            repo_root=project_root,
            algo=args.algo,
            timesteps_total=args.timesteps,
            use_gpu=use_gpu,
            log_trajectories=args.log_trajectories
        )
        
        print(f"✅ Training completed successfully!")
        print(f"📁 Model checkpoint: {checkpoint}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)