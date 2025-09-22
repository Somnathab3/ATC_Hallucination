
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

from src.scenarios.scenario_generator import generate_and_save_canonical
from src.training.train_frozen_scenario import train_frozen

if __name__ == "__main__":
    REPO = os.path.dirname(os.path.abspath(__file__))
    scen_path = os.path.join(REPO, "scenarios", "canonical_crossing.json")
    if not os.path.exists(scen_path):
        generate_and_save_canonical(REPO)

    ckpt = train_frozen(REPO, algo=os.environ.get("ALGO", "PPO"))
    print("Final checkpoint:", ckpt)
