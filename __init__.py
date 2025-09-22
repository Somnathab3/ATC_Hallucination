"""
ATC Hallucination Detection Package

A comprehensive Multi-Agent Reinforcement Learning framework for Air Traffic Control
with advanced hallucination detection and robustness testing capabilities.
"""

__version__ = "1.0.0"
__author__ = "ATC Hallucination Detection Project"
__email__ = "contact@atc-hallucination.project"
__license__ = "MIT"

# Version information
VERSION = __version__

# Package metadata
DESCRIPTION = "Multi-Agent RL framework for ATC with hallucination detection"
LONG_DESCRIPTION = """
A comprehensive Multi-Agent Reinforcement Learning framework for Air Traffic Control
with advanced hallucination detection and robustness testing capabilities.
"""

# Available scenarios
SCENARIOS = [
    "parallel",      # 3-agent parallel formation (baseline)
    "head_on",       # 2-agent head-on encounter
    "converging",    # 4-agent converging scenario
    "canonical_crossing",  # 4-agent crossing pattern
    "t_formation",   # 3-agent T-shaped configuration
]

# Supported algorithms
ALGORITHMS = ["PPO", "SAC"]

# Test configurations
TEST_TYPES = ["unison_shifts", "targeted_shifts"]

def get_version():
    """Return the package version."""
    return __version__

def get_scenarios():
    """Return list of available scenarios."""
    return SCENARIOS.copy()

def get_algorithms():
    """Return list of supported RL algorithms."""
    return ALGORITHMS.copy()

def get_test_types():
    """Return list of available test types."""
    return TEST_TYPES.copy()