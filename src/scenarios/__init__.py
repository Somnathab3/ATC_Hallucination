"""
Air Traffic Control Scenario Generation Module.

This module provides systematic generation of conflict scenarios for MARL training
and evaluation. All scenarios are designed to create inevitable conflicts without
intervention, testing collision avoidance capabilities of trained policies.

Core Functionality:
- Scenario Generator: Creates standardized conflict geometries (head-on, T-formation,
  parallel, converging, canonical crossing)
- JSON Configuration: Saves scenarios in machine-readable format for environment loading
- Reproducible Conflicts: Each scenario generates predictable conflict situations
  at specific times and locations

Scenario Types:
- head_on.json: Two aircraft on reciprocal headings
- t_formation.json: Perpendicular crossing with multiple aircraft  
- parallel.json: In-trail separation challenges
- converging.json: Multiple aircraft targeting nearby waypoints
- canonical_crossing.json: Four-way orthogonal intersection

Integration:
Scenarios feed directly into the MARLCollisionEnv via scenario_path configuration,
providing the initial conditions and waypoint assignments for training episodes.
"""