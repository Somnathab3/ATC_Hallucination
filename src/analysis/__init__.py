"""
Module Name: __init__.py
Description: Analysis package for ATC hallucination visualization and metrics.
             Provides comprehensive tools for aircraft collision avoidance system evaluation.
Author: Som
Date: 2025-10-04

This package includes:
- Geographic trajectory mapping with folium
- Interactive temporal analysis with plotly
- Publication-quality static plots with matplotlib
- Hotspot clustering and similarity analysis
- Safety margin statistics and performance metrics
"""

__version__ = "1.0.0"
__author__ = "Som"

# Main visualization modules
from .viz_geographic import build_map, make_basemap, add_trajectories
from .viz_plotly import time_series_panel, animated_geo
from .viz_matplotlib import plot_degradation_curves, heatmap_agent_vulnerability

# Analysis support modules  
from .ingest import load_trajectories, load_hallucination_results
from .analysis_safety import dwell_fraction, min_cpa, action_oscillation

__all__ = [
    'build_map', 'make_basemap', 'add_trajectories',
    'time_series_panel', 'animated_geo', 
    'plot_degradation_curves', 'heatmap_agent_vulnerability',
    'load_trajectories', 'load_hallucination_results',
    'dwell_fraction', 'min_cpa', 'action_oscillation'
]