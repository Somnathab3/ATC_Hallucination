# CLI Examples Reference

Complete copy-pasteable command examples extracted from `atc_cli.py` with practical usage patterns.

## Scenario Generation

### Generate All Scenarios
```bash
# Create all 5 standardized scenarios with FIXED distances
python atc_cli.py generate-scenarios --all
```

### Generate Specific Scenarios
```bash
# Generate head-on scenario with custom approach distance
python atc_cli.py generate-scenarios --scenario head_on --params "approach_nm=20"

# Generate multiple scenarios
python atc_cli.py generate-scenarios --scenario head_on
python atc_cli.py generate-scenarios --scenario t_formation
python atc_cli.py generate-scenarios --scenario parallel
```

### Custom Parameters
```bash
# T-formation with adjusted distances
python atc_cli.py generate-scenarios --scenario t_formation --params "arm_nm=10,stem_nm=15"

# Converging with larger radius
python atc_cli.py generate-scenarios --scenario converging --params "radius_nm=15"
```

## Model Training

### Basic Training
```bash
# Train PPO on canonical crossing (recommended scenario)
python atc_cli.py train --scenario canonical_crossing --algo PPO --timesteps 100000

# Quick training with GPU acceleration
python atc_cli.py train --scenario head_on --algo PPO --timesteps 50000 --gpu
```

### Advanced Training Options
```bash
# High-timestep training with frequent checkpoints
python atc_cli.py train --scenario parallel --algo PPO --timesteps 2000000 --checkpoint-every 50000 --gpu

# Training with trajectory logging (slower but detailed)
python atc_cli.py train --algo PPO --timesteps 500000 --scenario canonical_crossing --log-trajectories

# CPU-only training (force disable GPU)
python atc_cli.py train --scenario head_on --algo SAC --timesteps 100000 --no-gpu
```

### Multiple Algorithms
```bash
# Train all algorithms on head_on scenario
python atc_cli.py train --scenario head_on --algo all --timesteps 100000

# Train all scenarios with PPO
python atc_cli.py train --scenario all --timesteps 50000 --algo PPO
```

### Algorithm-Specific Examples
```bash
# PPO with GPU acceleration
python atc_cli.py train --scenario parallel --algo PPO --timesteps 200000 --gpu

# SAC for continuous control
python atc_cli.py train --scenario converging --algo SAC --timesteps 150000 --gpu

# IMPALA for distributed training
python atc_cli.py train --scenario t_formation --algo IMPALA --timesteps 100000

# CQL for offline learning
python atc_cli.py train --scenario canonical_crossing --algo CQL --timesteps 100000

# APPO for asynchronous training
python atc_cli.py train --scenario head_on --algo APPO --timesteps 100000
```

## Distribution Shift Testing

### Targeted Shift Testing (Recommended)
```bash
# Basic targeted shifts with visualizations
python atc_cli.py test-shifts --targeted --episodes 5 --viz

# Comprehensive targeted testing
python atc_cli.py test-shifts --targeted --episodes 10 --viz --scenario parallel

# Quick targeted test (3 episodes per shift)
python atc_cli.py test-shifts --targeted --episodes 3 --scenario head_on
```

### Specific Checkpoint Testing
```bash
# Test specific checkpoint
python atc_cli.py test-shifts --checkpoint models/PPO_parallel_20250928_011245 --targeted --episodes 5

# Test with SAC algorithm
python atc_cli.py test-shifts --checkpoint latest --algo SAC --targeted --episodes 5 --viz

# Test on specific scenario
python atc_cli.py test-shifts --checkpoint latest --scenario canonical_crossing --targeted --viz
```

### Unison Shift Testing
```bash
# Basic unison shifts (all agents modified equally)
python atc_cli.py test-shifts --episodes 5 --scenario parallel

# Unison shifts without targeted flag
python atc_cli.py test-shifts --checkpoint latest --scenario head_on --episodes 3
```

## Analysis and Visualization

### Result Analysis
```bash
# Analyze latest results with hallucination detection
python atc_cli.py analyze

# Analyze specific results directory
python atc_cli.py analyze --results-dir results_PPO_head_on_20250928_190203

# Skip hallucination analysis for faster processing
python atc_cli.py analyze --results-dir results_targeted_shift_analysis_parallel_20250928_160641 --no-hallucination
```

### Trajectory Visualization
```bash
# Visualize specific trajectory file
python atc_cli.py visualize --trajectory traj_ep_0001.csv

# Visualize entire results directory
python atc_cli.py visualize --results-dir results_PPO_parallel_20250928_055800

# Custom output directory for visualizations
python atc_cli.py visualize --trajectory traj_ep_0001.csv --output-dir custom_viz/
```

## Full Pipeline Execution

### Complete Workflows
```bash
# Basic full pipeline (generate → train → test → analyze)
python atc_cli.py full-pipeline --scenario head_on --train-timesteps 50000 --test-episodes 3

# Comprehensive pipeline with visualizations
python atc_cli.py full-pipeline --scenario parallel --train-timesteps 100000 --test-episodes 5 --algo PPO

# Quick evaluation pipeline
python atc_cli.py full-pipeline --scenario canonical_crossing --train-timesteps 25000 --test-episodes 3 --algo SAC
```

### Advanced Pipeline Options
```bash
# Pipeline with trajectory logging during training
python atc_cli.py full-pipeline --scenario converging --train-timesteps 75000 --test-episodes 5 --log-trajectories

# Pipeline without visualization generation (faster)
python atc_cli.py full-pipeline --scenario t_formation --train-timesteps 50000 --test-episodes 3 --no-viz

# Pipeline with unison shifts instead of targeted
python atc_cli.py full-pipeline --scenario head_on --train-timesteps 50000 --test-episodes 5 --no-targeted
```

## Resource Management

### List Available Resources
```bash
# List all available scenarios
python atc_cli.py list scenarios

# List all model checkpoints (newest first)
python atc_cli.py list checkpoints

# List all results directories
python atc_cli.py list results
```

### Checkpoint Management Examples
```bash
# Train and then immediately test the result
python atc_cli.py train --scenario head_on --algo PPO --timesteps 50000
python atc_cli.py test-shifts --checkpoint latest --targeted --episodes 3
```

## Practical Workflows

### Research Workflow
```bash
# 1. Generate all scenarios
python atc_cli.py generate-scenarios --all

# 2. Train baseline models
python atc_cli.py train --scenario canonical_crossing --algo PPO --timesteps 100000 --gpu
python atc_cli.py train --scenario parallel --algo PPO --timesteps 100000 --gpu
python atc_cli.py train --scenario head_on --algo PPO --timesteps 100000 --gpu

# 3. Comprehensive robustness testing
python atc_cli.py test-shifts --targeted --episodes 10 --viz --scenario canonical_crossing
python atc_cli.py test-shifts --targeted --episodes 10 --viz --scenario parallel
python atc_cli.py test-shifts --targeted --episodes 10 --viz --scenario head_on

# 4. Analysis and reporting
python atc_cli.py analyze
```

### Development Workflow
```bash
# Quick iteration cycle
python atc_cli.py train --scenario head_on --algo PPO --timesteps 25000 --gpu
python atc_cli.py test-shifts --targeted --episodes 3 --scenario head_on
python atc_cli.py analyze
```

### Algorithm Comparison
```bash
# Train multiple algorithms on same scenario
python atc_cli.py train --scenario parallel --algo PPO --timesteps 100000 --gpu
python atc_cli.py train --scenario parallel --algo SAC --timesteps 100000 --gpu
python atc_cli.py train --scenario parallel --algo IMPALA --timesteps 100000 --gpu

# Test each algorithm's robustness
python atc_cli.py test-shifts --checkpoint models/PPO_parallel_* --targeted --episodes 5 --viz
python atc_cli.py test-shifts --checkpoint models/SAC_parallel_* --targeted --episodes 5 --viz
python atc_cli.py test-shifts --checkpoint models/IMPALA_parallel_* --targeted --episodes 5 --viz
```

## Debugging and Troubleshooting

### Verbose Output
```bash
# Enable verbose logging for detailed output
python atc_cli.py train --scenario head_on --algo PPO --timesteps 50000 --verbose
python atc_cli.py test-shifts --targeted --episodes 3 --verbose
```

### Resource-Constrained Environments
```bash
# CPU-only training with minimal resources
python atc_cli.py train --scenario head_on --algo PPO --timesteps 25000 --no-gpu

# Quick testing with minimal episodes
python atc_cli.py test-shifts --targeted --episodes 1 --scenario head_on

# Memory-efficient analysis
python atc_cli.py analyze --no-hallucination
```

### Checkpoint Recovery
```bash
# List available checkpoints
python atc_cli.py list checkpoints

# Test specific checkpoint if auto-detection fails
python atc_cli.py test-shifts --checkpoint models/PPO_head_on_20250928_123456 --targeted --episodes 3
```

## Performance Optimization

### GPU-Accelerated Training
```bash
# Maximum performance training
python atc_cli.py train --scenario canonical_crossing --algo PPO --timesteps 500000 --gpu --checkpoint-every 25000

# GPU-accelerated full pipeline
python atc_cli.py full-pipeline --scenario parallel --train-timesteps 200000 --test-episodes 10 --algo PPO
```

### Batch Processing
```bash
# Process multiple scenarios efficiently
for scenario in head_on parallel t_formation converging canonical_crossing; do
    python atc_cli.py train --scenario $scenario --algo PPO --timesteps 100000 --gpu
done

# Batch testing (Windows PowerShell)
$scenarios = @("head_on", "parallel", "t_formation", "converging", "canonical_crossing")
foreach ($scenario in $scenarios) {
    python atc_cli.py test-shifts --scenario $scenario --targeted --episodes 5 --viz
}
```

## Integration Examples

### External Analysis Integration
```bash
# Generate data for external analysis tools
python atc_cli.py train --scenario parallel --algo PPO --timesteps 100000 --log-trajectories
python atc_cli.py test-shifts --targeted --episodes 10 --scenario parallel
python atc_cli.py visualize --results-dir results_targeted_shift_analysis_parallel_* --output-dir external_analysis/
```

### Continuous Integration Workflow
```bash
# CI-friendly commands (deterministic, quick)
python atc_cli.py generate-scenarios --all
python atc_cli.py train --scenario head_on --algo PPO --timesteps 10000 --no-gpu
python atc_cli.py test-shifts --targeted --episodes 1 --scenario head_on
```

## Command Combinations

### Chain Commands with Success Checking
```bash
# Train and immediately test if successful (Windows PowerShell)
python atc_cli.py train --scenario head_on --algo PPO --timesteps 50000
if ($LASTEXITCODE -eq 0) {
    python atc_cli.py test-shifts --targeted --episodes 5 --viz
}

# Full workflow with error handling (Bash)
python atc_cli.py train --scenario parallel --algo PPO --timesteps 100000 && \
python atc_cli.py test-shifts --targeted --episodes 5 --viz && \
python atc_cli.py analyze
```

### Parallel Processing (if supported)
```bash
# Train multiple models in parallel (separate terminals)
# Terminal 1:
python atc_cli.py train --scenario head_on --algo PPO --timesteps 100000 --gpu

# Terminal 2:
python atc_cli.py train --scenario parallel --algo SAC --timesteps 100000 --gpu

# Terminal 3:
python atc_cli.py train --scenario t_formation --algo IMPALA --timesteps 100000 --gpu
```

## Help and Documentation

### Get Help
```bash
# General help
python atc_cli.py --help

# Command-specific help
python atc_cli.py train --help
python atc_cli.py test-shifts --help
python atc_cli.py full-pipeline --help

# List all available options
python atc_cli.py list --help
```

### Version and Status Information
```bash
# Check available scenarios and checkpoints
python atc_cli.py list scenarios
python atc_cli.py list checkpoints
python atc_cli.py list results
```