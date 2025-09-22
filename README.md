# ATC Hallucination Detection

A Multi-Agent Reinforcement Learning framework for Air Traffic Control conflict detection and resolution with comprehensive hallucination detection capabilities.

## 🎯 Overview

This project implements a sophisticated MARL system for air traffic control that can:
- Train multi-agent collision avoidance policies using PPO/SAC
- Detect and analyze hallucination patterns in trained models
- Perform comprehensive distribution shift testing
- Evaluate model robustness through targeted perturbations

### Key Features

- **Multi-Agent Training**: PPO algorithms for collaborative conflict resolution
- **Hallucination Detection**: Advanced metrics for identifying false positives/negatives
- **Distribution Shift Testing**: Systematic evaluation of model robustness
- **Targeted Perturbations**: Single-agent modifications to test edge cases
- **Comprehensive Analysis**: Detailed performance metrics and visualizations

## 🏗️ Architecture

```
src/
├── environment/          # MARL collision avoidance environment
│   └── marl_collision_env_minimal.py
├── training/             # Training scripts and utilities
│   └── train_frozen_scenario.py
├── analysis/             # Hallucination detection and analysis
│   └── hallucination_detector_enhanced.py
├── testing/              # Distribution shift testing
│   ├── shift_tester.py
│   └── targeted_shift_tester.py
└── scenarios/            # Test scenarios and configuration
    └── scenario_generator.py

scenarios/                # JSON scenario definitions
├── parallel.json         # 3-agent parallel formation
├── head_on.json          # 2-agent head-on encounter
├── converging.json       # 4-agent converging scenario
└── ...

tests/                    # Test scripts and validation
├── train.py              # Main training script
├── test_shifts.py        # Distribution shift testing
├── test_targeted_shifts.py # Targeted shift analysis
└── visualize_air_traffic.py # Real-time visualization
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install ray[rllib] bluesky-simulator pandas numpy matplotlib pygame pettingzoo
```

### Training a Model

```bash
# Train PPO on parallel scenario
python train.py

# Train with specific configuration
python src/training/train_frozen_scenario.py --scenario parallel --episodes 1000
```

### Testing for Hallucinations

```bash
# Run comprehensive shift testing
python test_shifts.py

# Run targeted single-agent shifts
python src/testing/targeted_shift_tester.py --scenario parallel --episodes 10

# Test specific configurations
python src/testing/targeted_shift_tester.py \
  --checkpoint models/your_model \
  --scenario head_on \
  --episodes 5 \
  --algorithm SAC
```

### Visualization

```bash
# Visualize trained model in action
python visualize_air_traffic.py --checkpoint models/your_model --scenario parallel
```

## 📊 Testing Framework

### Distribution Shift Testing

The framework supports two types of distribution shift testing:

#### 1. Unison Shifts
All agents are modified equally:
- Speed variations: ±5 to ±30 kt
- Position shifts: 0.1 to 1.0 NM
- Heading changes: ±3 to ±15°
- Action delays: 0 to 5 steps

#### 2. Targeted Shifts
Only one agent is modified per test:
- **Micro perturbations**: Small variations within training envelope
- **Macro deviations**: Large changes to test failure modes
- **Conflict-inducing**: Position shifts moving agents closer
- **96 test configurations** covering all agents and parameters

### Key Metrics

- **Safety**: Loss of Separation (LoS) events, minimum separation
- **Detection**: True/False positives/negatives for conflict prediction
- **Resolution**: Success rate of conflict resolution maneuvers
- **Efficiency**: Path efficiency, waypoint completion rates
- **Stability**: Action oscillation and control smoothness

## 🔧 Command Line Interface

### Basic Usage

```bash
# Auto-detect checkpoint and run with defaults
python src/testing/targeted_shift_tester.py

# List available options
python src/testing/targeted_shift_tester.py --list-scenarios
python src/testing/targeted_shift_tester.py --list-checkpoints
python src/testing/targeted_shift_tester.py --help
```

### Advanced Configuration

```bash
# Custom testing with specific parameters
python src/testing/targeted_shift_tester.py \
  --checkpoint "models/results_20231201_120000_Parallel/models" \
  --scenario converging \
  --episodes 20 \
  --algorithm PPO \
  --verbose

# Dry run to verify configuration
python src/testing/targeted_shift_tester.py \
  --scenario parallel \
  --episodes 5 \
  --dry-run
```

## 📈 Results Analysis

Test results are organized in timestamped directories:

```
results/targeted_shift_analysis_parallel_YYYYMMDD_HHMMSS/
├── shifts/                     # Individual test episodes
│   ├── speed_micro_A1_+5kt/   # Episode data and trajectories
│   ├── pos_closer_macro_A2_north_0.30deg/
│   └── ...
├── analysis/                   # Aggregated analysis
│   ├── A1_analysis.csv        # Agent-specific statistics
│   ├── conflict_inducing_shifts.csv # Most problematic configs
│   ├── type_range_analysis.csv # Micro vs macro comparison
│   └── targeted_shift_summary.json
├── targeted_shift_test_summary.csv # Main results
└── README.md                   # Detailed results summary
```

### Key Analysis Files

- **`conflict_inducing_shifts.csv`**: Configurations causing most conflicts
- **`A1_analysis.csv`, `A2_analysis.csv`, `A3_analysis.csv`**: Agent-specific vulnerabilities
- **`type_range_analysis.csv`**: Performance by shift type and magnitude
- **`targeted_shift_summary.json`**: Overall statistics and findings

## 🧪 Scenarios

The framework includes several pre-defined scenarios:

| Scenario | Agents | Description |
|----------|--------|-------------|
| `parallel` | 3 | Parallel formation (baseline) |
| `head_on` | 2 | Direct head-on encounter |
| `converging` | 4 | Four aircraft converging |
| `canonical_crossing` | 4 | Standard crossing pattern |
| `t_formation` | 3 | T-shaped configuration |

### Custom Scenarios

Create new scenarios by adding JSON files to the `scenarios/` directory:

```json
{
  "scenario_name": "custom_test",
  "seed": 42,
  "center": {"lat": 52.0, "lon": 4.0, "alt_ft": 10000.0},
  "agents": [
    {
      "id": "A1",
      "type": "A320",
      "lat": 51.6, "lon": 4.0,
      "hdg_deg": 0.0, "spd_kt": 250.0,
      "alt_ft": 10000.0,
      "waypoint": {"lat": 52.4, "lon": 4.0}
    }
  ]
}
```

## 🔬 Research Applications

This framework is designed to support research in:

- **MARL Safety**: Evaluating collision avoidance in safety-critical systems
- **Hallucination Detection**: Identifying false predictions in learned policies
- **Robustness Testing**: Systematic evaluation of model performance under perturbations
- **Distribution Shift**: Understanding model behavior outside training data
- **Air Traffic Control**: Practical applications in aviation safety systems

## 📚 Key Publications and References

The testing methodology follows best practices from recent MARL and safety literature:

- Distribution shift testing for safety-critical systems
- Hallucination detection in reinforcement learning
- Multi-agent conflict resolution evaluation metrics
- Robustness testing for deep RL policies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings to all functions
- Include unit tests for new functionality
- Update documentation for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BlueSky simulator for air traffic simulation
- Ray RLLib for reinforcement learning framework
- PettingZoo for multi-agent environment standardization
- Research community for MARL safety methodologies

## 📞 Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project is focused on research applications. For production air traffic control systems, additional safety certifications and validations would be required.