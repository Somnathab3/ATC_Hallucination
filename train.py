
"""
Top-level training script.

1) Generate frozen scenario if missing.
2) Initialize RLlib trainer (PPO by default).
3) Train until convergence or 2M steps.
4) Save final checkpoint.
"""

# Quiet BlueSky logs BEFORE any imports
import logging
logging.basicConfig(level=logging.INFO)
for name in ("bluesky", "bluesky.navdatabase", "bluesky.simulation", "bluesky.traffic"):
    logging.getLogger(name).setLevel(logging.CRITICAL)
logging.getLogger().setLevel(logging.INFO)  # keep our training prints visible

import os

from src.scenarios.scenario_generator import make_head_on, make_t_formation, make_parallel, make_converging
from src.training.train_frozen_scenario import train_frozen

def ensure_scenarios_exist(repo_root: str):
    """Ensure basic scenarios exist."""
    scenarios_dir = os.path.join(repo_root, "scenarios")
    os.makedirs(scenarios_dir, exist_ok=True)
    
    # Check if canonical_crossing exists, if not create converging as fallback
    canonical_path = os.path.join(scenarios_dir, "canonical_crossing.json")
    if not os.path.exists(canonical_path):
        print("canonical_crossing.json not found, generating scenarios...")
        # Generate all basic scenarios
        make_head_on()
        make_t_formation() 
        make_parallel()
        make_converging()
        
        # Use converging as canonical_crossing
        converging_path = os.path.join(scenarios_dir, "converging.json")
        if os.path.exists(converging_path):
            import shutil
            shutil.copy2(converging_path, canonical_path)
            print(f"Created canonical_crossing.json from converging.json")
        else:
            print("Warning: Could not create canonical_crossing.json")

if __name__ == "__main__":
    REPO = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure scenarios exist
    ensure_scenarios_exist(REPO)
    
    # Get scenario name from environment or use default
    scenario_name = os.environ.get("SCENARIO", "canonical_crossing")
    algo = os.environ.get("ALGO", "PPO")
    
    print(f"Training with scenario: {scenario_name}, algorithm: {algo}")
    
    try:
        ckpt = train_frozen(REPO, algo=algo, scenario_name=scenario_name)
        print("Final checkpoint:", ckpt)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Available scenarios:")
        scenarios_dir = os.path.join(REPO, "scenarios")
        if os.path.exists(scenarios_dir):
            for f in os.listdir(scenarios_dir):
                if f.endswith('.json'):
                    print(f"  - {f[:-5]}")  # Remove .json extension
