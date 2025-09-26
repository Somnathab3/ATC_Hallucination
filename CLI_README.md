# ATC Hallucination Project - Unified CLI

This comprehensive command-line interface provides a single entry point for all ATC Hallucination project operations, including scenario generation, model training, shift testing, hallucination detection analysis, and visualization.

## Features

- **Scenario Generation**: Create custom air traffic scenarios with configurable parameters
- **Model Training**: Train PPO/SAC models with various scenarios and hyperparameters  
- **Distribution Shift Testing**: Run targeted shift testing to evaluate model robustness
- **Hallucination Analysis**: Detect false positives/negatives in conflict prediction
- **Visualization**: Generate trajectory plots and analysis visualizations
- **Full Pipeline**: Execute complete workflows from scenario generation to analysis

## Quick Start

```bash
# Generate all basic scenarios
python atc_cli.py generate-scenarios --all

# Train a PPO model on head-on scenario
python atc_cli.py train --scenario head_on --algo PPO --timesteps 100000

# Run targeted shift testing
python atc_cli.py test-shifts --targeted --episodes 5 --viz

# Analyze latest results
python atc_cli.py analyze

# Run complete pipeline
python atc_cli.py full-pipeline --scenario head_on --train-timesteps 50000 --test-episodes 3
```

## Commands

### Scenario Generation

Generate air traffic scenario files with custom parameters:

```bash
# Generate all scenario types
python atc_cli.py generate-scenarios --all

# Generate specific scenarios
python atc_cli.py generate-scenarios --scenario head_on --scenario parallel

# Generate with custom parameters
python atc_cli.py generate-scenarios --scenario head_on --params "approach_nm=25,spd_kt=300"
```

**Available scenario types:**
- `head_on`: Two aircraft on collision course
- `t_formation`: T-shaped formation with crossing paths
- `parallel`: Multiple aircraft on parallel tracks
- `converging`: Aircraft converging from multiple directions

**Custom parameters:**
- `approach_nm`: Approach distance in nautical miles
- `spd_kt`: Aircraft speed in knots
- `alt_ft`: Altitude in feet
- And more (see scenario generator documentation)

### Model Training

Train reinforcement learning models:

```bash
# Basic training
python atc_cli.py train --scenario head_on --algo PPO --timesteps 100000

# Advanced training with custom checkpoints
python atc_cli.py train --scenario parallel --algo SAC --timesteps 200000 --checkpoint-every 20000

# Train multiple scenarios sequentially
python atc_cli.py train --scenario all --timesteps 50000
```

**Algorithms:**
- `PPO`: Proximal Policy Optimization (default)
- `SAC`: Soft Actor-Critic

**Parameters:**
- `--timesteps`: Total training timesteps (default: 100000)
- `--checkpoint-every`: Checkpoint frequency (default: timesteps/10)

### Distribution Shift Testing

Test model robustness under distribution shifts:

```bash
# Targeted shifts (recommended)
python atc_cli.py test-shifts --targeted --episodes 5 --viz

# Test specific scenario
python atc_cli.py test-shifts --scenario parallel --checkpoint path/to/model --episodes 3

# Auto-detect latest checkpoint
python atc_cli.py test-shifts --targeted --episodes 5
```

**Shift types:**
- **Targeted**: Modify one agent at a time (more realistic)
- **Unison**: Modify all agents equally (legacy support)

**Parameters:**
- `--episodes`: Episodes per shift configuration (default: 3)
- `--viz`: Generate visualizations
- `--checkpoint`: Model checkpoint path (auto-detected if not specified)

### Analysis

Analyze training and testing results:

```bash
# Analyze latest results
python atc_cli.py analyze

# Analyze specific directory
python atc_cli.py analyze --results-dir results_PPO_head_on_20250923_190203

# Skip hallucination analysis (faster)
python atc_cli.py analyze --no-hallucination
```

**Output includes:**
- Trajectory statistics (conflicts, separations, waypoint completion)
- Hallucination detection metrics (TP/FP/FN/TN, precision, recall)
- Training progress analysis
- Safety and efficiency metrics

### Visualization

Generate plots and visualizations:

```bash
# Visualize specific trajectory
python atc_cli.py visualize --trajectory traj_ep_0001.csv

# Visualize entire results directory
python atc_cli.py visualize --results-dir results_PPO_head_on_20250923_190203

# Custom output directory
python atc_cli.py visualize --trajectory traj_ep_0001.csv --output-dir my_plots/
```

### Full Pipeline

Execute complete workflows:

```bash
# Basic pipeline
python atc_cli.py full-pipeline --scenario head_on

# Customized pipeline
python atc_cli.py full-pipeline \
    --scenario parallel \
    --algo SAC \
    --train-timesteps 200000 \
    --test-episodes 5 \
    --no-viz

# High-performance pipeline
python atc_cli.py full-pipeline \
    --scenario converging \
    --train-timesteps 500000 \
    --test-episodes 10
```

**Pipeline steps:**
1. Generate scenarios (if missing)
2. Train model
3. Run shift testing
4. Analyze results
5. Generate visualizations (optional)

### Utility Commands

List available resources:

```bash
# List scenarios
python atc_cli.py list scenarios

# List model checkpoints
python atc_cli.py list checkpoints

# List results directories
python atc_cli.py list results
```

## Configuration

### Global Options

```bash
# Enable verbose logging
python atc_cli.py --verbose <command>

# Specify project root
python atc_cli.py --repo-root /path/to/project <command>
```

### Environment Variables

Set default parameters using environment variables:

```bash
export SCENARIO=head_on
export ALGO=PPO
export TIMESTEPS=100000

python atc_cli.py train  # Uses defaults above
```

## Output Structure

The CLI creates organized output directories:

```
project/
├── scenarios/          # Generated scenario files
│   ├── head_on.json
│   ├── parallel.json
│   └── ...
├── results_PPO_head_on_20250923_190203/  # Training results
│   ├── models/         # Model checkpoints
│   ├── traj_ep_*.csv   # Episode trajectories
│   └── training_progress.csv
├── targeted_shift_analysis_parallel_20250923_200145/  # Shift testing
│   ├── shifts/         # Individual shift results
│   ├── analysis/       # Aggregated analysis
│   └── README.md
└── vis_temp/           # Visualizations
    ├── trajectory_plots/
    └── analysis_charts/
```

## Examples

### Example 1: Quick Training Test

```bash
# Generate scenarios and train quickly
python atc_cli.py generate-scenarios --all
python atc_cli.py train --scenario head_on --timesteps 5000
python atc_cli.py analyze
```

### Example 2: Comprehensive Evaluation

```bash
# Full evaluation pipeline
python atc_cli.py full-pipeline \
    --scenario parallel \
    --algo PPO \
    --train-timesteps 100000 \
    --test-episodes 10 \
    --viz
```

### Example 3: Model Comparison

```bash
# Train multiple algorithms
python atc_cli.py train --scenario head_on --algo PPO --timesteps 50000
python atc_cli.py train --scenario head_on --algo SAC --timesteps 50000

# Test both
python atc_cli.py test-shifts --checkpoint results_PPO_*/models --targeted --episodes 5
python atc_cli.py test-shifts --checkpoint results_SAC_*/models --targeted --episodes 5
```

### Example 4: Custom Scenario Testing

```bash
# Create custom scenario
python atc_cli.py generate-scenarios \
    --scenario head_on \
    --params "approach_nm=30,spd_kt=350"

# Train and test
python atc_cli.py train --scenario head_on --timesteps 100000
python atc_cli.py test-shifts --targeted --episodes 10 --viz
```

## Advanced Usage

### Batch Processing

Use shell scripting for batch operations:

```bash
#!/bin/bash
# Train multiple scenarios
for scenario in head_on parallel converging; do
    python atc_cli.py train --scenario $scenario --timesteps 50000
    python atc_cli.py test-shifts --targeted --episodes 5
done
```

### Integration with Analysis Tools

The CLI outputs structured data that can be processed by external tools:

```python
import pandas as pd
import json

# Load analysis results
with open('results_*/analysis/targeted_shift_summary.json') as f:
    analysis = json.load(f)

# Load trajectory data
df = pd.read_csv('results_*/traj_ep_0001.csv')

# Custom analysis...
```

## Troubleshooting

### Common Issues

1. **"Scenario not found"**: Run `python atc_cli.py generate-scenarios --all` first
2. **"No checkpoint found"**: Train a model with `python atc_cli.py train` first
3. **Import errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
4. **BlueSky errors**: Check that BlueSky is properly installed and configured

### Debug Mode

Use verbose logging to debug issues:

```bash
python atc_cli.py --verbose train --scenario head_on --timesteps 1000
```

### Performance Tips

1. **Use fewer timesteps** for quick testing
2. **Skip visualization** (`--no-viz`) for faster execution
3. **Reduce episodes** for shift testing during development
4. **Use targeted shifts** instead of unison for more efficient testing

## Contributing

When adding new features to the CLI:

1. Add the command parser in `main()`
2. Implement the functionality in `ATCController`
3. Update this documentation
4. Add example usage
5. Test with various scenarios

## See Also

- [Environment Documentation](src/environment/README.md)
- [Training Documentation](src/training/README.md)
- [Shift Testing Documentation](src/testing/README.md)
- [Analysis Documentation](src/analysis/README.md)