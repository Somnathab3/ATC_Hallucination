"""
Top-level test script for distribution shifts with enhanced hallucination detection.

1) Load final trained checkpoint.
2) Sweep over speed/position/heading/delay shifts with gradual changes.
3) Run multiple episodes per shift setting, collect trajectories.
4) Compute enhanced hallucination metrics including LOS events and efficiency and save summary CSV.
"""

import os
import json
import logging

from ray.rllib.algorithms.ppo import PPO
from src.testing.shift_tester import run_shift_grid

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    REPO = os.path.dirname(os.path.abspath(__file__))
    
    # Look for parallel scenario models first
    parallel_model_path = os.path.join(REPO, "results_20250922_181059_Parallel", "models")
    
    if os.path.exists(parallel_model_path):
        ckpt = parallel_model_path
        print(f"Using parallel scenario model: {ckpt}")
    else:
        # Fallback to models directory
        models_dir = os.path.join(REPO, "models")
        ckpts = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.startswith("checkpoint_")] if os.path.exists(models_dir) else []
        if not ckpts:
            raise SystemExit("No checkpoints found. Train the parallel scenario first.")
        ckpt = max(ckpts, key=os.path.getmtime)

    # Default to PPO class; shift_tester will try to infer SAC if the metadata says so.
    from ray.rllib.algorithms.ppo import PPO
    out_csv = run_shift_grid(REPO, PPO, ckpt, episodes_per_shift=5)  # Reduced for faster testing
    print("Wrote enhanced shift test summary with LOS events and efficiency metrics:", out_csv)
