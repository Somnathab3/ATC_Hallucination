# Changelog

All notable changes to the ATC Hallucination Detection project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-09-22

### Added
- Initial release of ATC Hallucination Detection framework
- Multi-agent reinforcement learning environment for air traffic control
- PPO and SAC training algorithms for collision avoidance
- Comprehensive hallucination detection system with enhanced metrics
- Distribution shift testing framework with unison and targeted shifts
- Targeted shift testing with single-agent modifications (96 configurations)
- Command-line interface for flexible testing and analysis
- Real-time visualization system with conflict detection
- Multiple predefined scenarios (parallel, head-on, converging, etc.)
- Comprehensive analysis and reporting system
- Automated result organization and documentation

### Features
- **Training**: Multi-agent collision avoidance using PPO/SAC
- **Testing**: Systematic evaluation of model robustness
- **Analysis**: Detailed hallucination and performance metrics
- **Visualization**: Real-time air traffic control simulation
- **CLI Tools**: Easy-to-use command-line interfaces
- **Scenarios**: Multiple aircraft configurations for testing

### Metrics
- Safety: Loss of Separation (LoS) events, minimum separation
- Detection: True/False positives/negatives for conflict prediction  
- Resolution: Success rate of conflict resolution maneuvers
- Efficiency: Path efficiency, waypoint completion rates
- Stability: Action oscillation and control smoothness

### Research Applications
- Multi-agent reinforcement learning safety evaluation
- Hallucination detection in safety-critical systems
- Distribution shift analysis for deep RL policies
- Air traffic control conflict resolution assessment

---

## Development Guidelines

This changelog follows the principles of:
- **Keep a Changelog**: Human-readable format for tracking changes
- **Semantic Versioning**: MAJOR.MINOR.PATCH version numbering
- **Clear Categories**: Added, Changed, Deprecated, Removed, Fixed, Security