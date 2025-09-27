"""
ATC Hallucination Detection Framework - Core Implementation Package

This package implements a comprehensive research framework for training, evaluating,
and analyzing Multi-Agent Reinforcement Learning (MARL) policies for air traffic
control collision avoidance, with particular focus on detecting and quantifying
hallucination phenomena (false alerts and missed conflicts).

SYSTEM ARCHITECTURE AND DATA FLOW:

1. SCENARIO GENERATION (src.scenarios)
   ├── scenario_generator.py: Creates standardized conflict scenarios
   ├── Outputs: JSON configuration files (head_on.json, t_formation.json, etc.)
   └── Purpose: Provides reproducible initial conditions for training/testing

2. ENVIRONMENT SIMULATION (src.environment) 
   ├── marl_collision_env_minimal.py: Main MARL environment with BlueSky integration
   ├── Inputs: Scenario JSON files, shift configurations (optional)
   ├── Outputs: Trajectory CSV files with real-time hallucination detection
   └── Purpose: High-fidelity multi-agent simulation with reward shaping

3. POLICY TRAINING (src.training)
   ├── train_frozen_scenario.py: Multi-algorithm training with GPU acceleration
   ├── Inputs: Scenarios from step 1, environment from step 2
   ├── Outputs: Trained model checkpoints, training progress logs
   └── Purpose: Learn collision avoidance policies with parameter sharing

4. ROBUSTNESS TESTING (src.testing)
   ├── targeted_shift_tester.py: Systematic distribution shift evaluation
   ├── baseline_vs_shift_matrix.py: Comparative baseline analysis
   ├── Inputs: Trained models from step 3, scenarios from step 1
   ├── Outputs: Performance matrices, shift test results
   └── Purpose: Evaluate policy robustness and identify failure modes

5. ANALYSIS AND VISUALIZATION (src.analysis)
   ├── hallucination_detector_enhanced.py: Real-time conflict prediction analysis
   ├── viz_*.py: Geographic, temporal, and statistical visualization tools
   ├── trajectory_comparison_*.py: Shift analysis and interactive dashboards
   ├── Inputs: Trajectory data from steps 2&4, test results from step 4
   ├── Outputs: Performance metrics, visualizations, academic reports
   └── Purpose: Quantify hallucinations and generate publication-ready analysis

RESEARCH PIPELINE FLOW:
scenarios → environment → training → testing → analysis
    ↓           ↓            ↓          ↓         ↓
JSON files → trajectories → models → shift tests → metrics

KEY FEATURES:
- Real-time hallucination detection during simulation
- Intent-aware and threat-aware alert filtering
- IoU-based event matching for robust performance evaluation
- Geographic trajectory visualization with conflict overlays
- Interactive dashboards for academic presentation
- Comprehensive statistical analysis with significance testing

INTEGRATION POINTS:
- BlueSky flight simulator for realistic aircraft dynamics
- RLlib for state-of-the-art MARL algorithms (PPO, SAC, etc.)
- PettingZoo for standardized multi-agent environment interface
- Ray for distributed training and evaluation
- Pandas/NumPy for data analysis and statistical evaluation
- Plotly/Folium for interactive visualizations

ACADEMIC APPLICATIONS:
This framework supports systematic evaluation of MARL collision avoidance
policies for academic research, providing rigorous statistical analysis
and publication-quality visualizations suitable for conference presentations
and journal submissions.
"""

__version__ = "1.0.0"
__author__ = "ATC Hallucination Detection Research Team"

# Core module imports for convenience
from . import scenarios
from . import environment  
from . import training
from . import testing
from . import analysis

__all__ = [
    'scenarios',
    'environment', 
    'training',
    'testing', 
    'analysis'
]