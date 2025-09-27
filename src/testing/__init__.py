"""
Distribution Shift Testing and Robustness Evaluation Module.

This module implements systematic frameworks for evaluating trained MARL policies
under distribution shifts and targeted perturbations to assess robustness and
identify failure modes beyond the training envelope.

Core Testing Frameworks:
- Targeted Shift Testing: Individual agent modifications (speed, position, heading)
  with micro to macro magnitude ranges to identify single points of failure
- Baseline vs Shift Matrix: Comparative analysis between nominal and perturbed conditions
- Environmental Shift Testing: Wind field variations and aircraft type modifications
- Comprehensive Analysis: Statistical evaluation across shift types and magnitudes

Testing Philosophy:
- Systematic Perturbations: Controlled modifications to specific system parameters
- Conflict-Inducing Design: Shifts intentionally increase collision probability
- Academic Rigor: Statistical significance testing and comprehensive reporting
- Real-time Detection: Integration with hallucination detection during evaluation

Key Features:  
- Multi-dimensional Testing: Speed, position, heading, aircraft type, waypoint shifts
- Range-based Analysis: Micro (training envelope) vs macro (beyond training) variations
- Visualization Integration: Geographic trajectory comparison and interactive dashboards
- Performance Metrics: Precision, recall, F1-score, alert burden, timing analysis

Integration:
Uses trained models from src.training, loads scenarios from src.scenarios,
executes tests in src.environment, and generates analysis using src.analysis
hallucination detection and visualization capabilities.
"""