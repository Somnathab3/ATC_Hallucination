"""
Multi-Agent Reinforcement Learning Environment Module.

This module implements the core MARL environment for training air traffic control
policies using BlueSky flight simulation. The environment bridges high-fidelity
aircraft dynamics with RL training frameworks through standardized interfaces.

Core Components:
- MARLCollisionEnv: Main environment class implementing PettingZoo ParallelEnv interface
- BlueSky Integration: Real-time flight dynamics simulation with wind modeling
- Observation Spaces: Relative positioning (no raw lat/lon) for generalization
- Action Spaces: Continuous heading and speed control with physical scaling
- Reward Design: Multi-component rewards balancing safety, efficiency, and coordination
- Real-time Logging: Comprehensive trajectory data with hallucination detection

Key Features:
- Team-based PBRS: Potential-based reward shaping for multi-agent coordination
- Distribution Shift Support: Targeted and unison shift testing capabilities
- Waypoint Navigation: Goal-directed flight with completion tracking
- Conflict Detection: Real-time separation monitoring and LOS event logging
- Environmental Factors: Wind field simulation for robustness testing

Integration:
The environment loads scenarios from src.scenarios, provides training data to
src.training algorithms, generates trajectory logs for src.analysis evaluation,
and supports distribution shift testing via src.testing frameworks.
"""