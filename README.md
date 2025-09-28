# ATC Hallucination Project

A comprehensive Multi-Agent Reinforcement Learning framework for air traffic collision avoidance with systematic robustness testing and hallucination detection capabilities.

## Overview

This project provides end-to-end capabilities for training, testing, and analyzing MARL collision avoidance policies. Key components include multi-algorithm training (PPO/SAC/IMPALA/CQL/APPO), targeted distribution shift testing, real-time hallucination detection, and comprehensive analysis/visualization packages.

**Core Features:**
- **Training**: Multi-algorithm MARL training with shared policies and unified reward systems
- **Testing**: Targeted shifts and baseline-vs-shift matrix evaluation for robustness assessment  
- **Analysis**: Real-time and offline hallucination detection with TCPA/DCPA ground truth
- **Scenarios**: Five standardized air traffic conflict scenarios with "FIXED" distances for convergence
- **CLI**: Unified command-line interface for all operations with copy-pasteable examples

## Quick Start (CLI-first)

Install dependencies:
```bash
# Install PyTorch with CUDA support + RLlib + BlueSky
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ray[rllib] bluesky-simulator>=1.3.0 pandas numpy matplotlib pettingzoo gymnasium
```

Generate scenarios → Train → Test → Analyze → Visualize:

```bash
# Generate all scenarios
python atc_cli.py generate-scenarios --all

# Train PPO on canonical crossing (50k timesteps)
python atc_cli.py train --scenario canonical_crossing --algo PPO --timesteps 50000 --gpu

# Test robustness with targeted shifts (visualizations included)
python atc_cli.py test-shifts --targeted --episodes 5 --viz

# Analyze results from latest run
python atc_cli.py analyze

# Full pipeline: scenario generation → training → testing → analysis
python atc_cli.py full-pipeline --scenario head_on --train-timesteps 50000 --test-episodes 3

# List available resources
python atc_cli.py list scenarios
python atc_cli.py list checkpoints
```

## Training (algorithms & hyperparameters)

**Supported algorithms:** PPO, SAC, IMPALA, CQL, APPO

### PPO Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | [256, 256] tanh | Hidden layer sizes and activation |
| **Learning** | lr=5e-4 (GPU) / 3e-4 (CPU) | Learning rate |
| **Training** | gamma=0.995, epochs=10/8 | Discount factor, SGD epochs |
| **Batch Size** | 8192 (GPU) / 4096 (CPU) | Training batch size |
| **Rollout** | length=200, num_workers=4 | Fragment length, parallel workers |
| **Clipping** | grad_clip=1.0, clip_param=0.1 | Gradient and policy clipping |
| **Regularization** | entropy_coeff=0.01, kl_coeff=0.2 | Exploration and stability |
| **Resources** | num_gpus=1 (if available) | GPU allocation per algorithm |
| **Evaluation** | interval=5, duration=5 episodes | Progress tracking frequency |

### SAC Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | [256, 256] tanh | Hidden layer sizes and activation |
| **Learning** | lr=5e-4, critic_lr=5e-4, alpha_lr=5e-4 | Actor/critic learning rates |
| **Training** | gamma=0.995, tau=0.01, training_intensity=1.5 | Q-function updates |
| **Batch Size** | 8192 (GPU) / 4096 (CPU) | Training batch size |
| **Replay Buffer** | 1M (GPU) / 500k (CPU) timesteps | Experience replay capacity |
| **Exploration** | initial_alpha=0.1, target_entropy=auto | Temperature parameter |
| **Warmup** | 2k (GPU) / 5k (CPU) steps before learning | Pre-training sampling |

### IMPALA Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | [256, 256] tanh | Hidden layer sizes and activation |
| **Learning** | lr=6e-4, minibatch_size=512 | Learning rate and batch processing |
| **V-trace** | clip_rho=1.0, clip_pg_rho=1.0 | Off-policy correction |
| **Regularization** | entropy_coeff=0.01, vf_loss_coeff=0.5 | Value function weighting |

**Multi-agent setup:** All algorithms use shared_policy with parameter sharing across agents. Observation/action spaces discovered via temporary environment instantiation. Policy mapping function routes all agents to "shared_policy".

**Training outputs:** Timestamped `training/results_ALGO_SCENARIO_TIMESTAMP/` directory containing:
- `training_progress.csv` with timestep, reward_mean, zero_conflict_streak columns
- Checkpoints saved on reward improvements in `checkpoints/best_*` subdirectory  
- Final model copied to `models/ALGO_SCENARIO_TIMESTAMP/` with training metadata

## Environment & Observations

**Observation space:** Relative observations only (no raw lat/lon) for generalization

| Key | Shape | Range | Description |
|-----|-------|-------|-------------|
| `wp_dist_norm` | (1,) | [-1, 1] | Normalized distance to waypoint (tanh) |
| `cos_to_wp` | (1,) | [-1, 1] | Cosine of direction to waypoint |
| `sin_to_wp` | (1,) | [-1, 1] | Sine of direction to waypoint |
| `airspeed` | (1,) | [-10, 10] | Normalized airspeed around 150 m/s |
| `progress_rate` | (1,) | [-1, 1] | Waypoint progress rate (+ approaching) |
| `safety_rate` | (1,) | [-1, 1] | Minimum separation rate (+ getting safer) |
| `x_r` | (3,) | [-12, 12] | Relative positions of top-3 neighbors |
| `y_r` | (3,) | [-12, 12] | Relative positions of top-3 neighbors |
| `vx_r` | (3,) | [-150, 150] | Relative velocities of neighbors |
| `vy_r` | (3,) | [-150, 150] | Relative velocities of neighbors |

**Action scaling:** Normalized [-1, 1] → ±18° heading, ±10 kt speed per 10-second step

**Episode termination:** Max 100 steps (1000 seconds) or all agents reach waypoints within 1 NM

**Reward system:** Unified components eliminate double-counting:
- **Signed progress:** +/- for movement toward/away from waypoint (0.04/km)
- **Well-clear violations:** Entry penalty (-25) + severity-scaled step penalties (-1.0×depth)
- **Drift improvement:** Rewards heading optimization toward waypoint (0.01/degree)
- **Team PBRS:** Enhanced coordination with 5 NM sensitivity (weight=0.6)
- **Action costs:** Penalty for non-neutral control inputs (-0.01/unit)
- **Time penalty:** Efficiency incentive (-0.0005/second)

**Trajectory CSV columns:** 
`episode_id, step_idx, sim_time_s, agent_id, lat_deg, lon_deg, alt_ft, hdg_deg, tas_kt, action_hdg_delta_deg, action_spd_delta_kt, reward, min_separation_nm, conflict_flag, waypoint_reached, gt_conflict, predicted_alert, tp, fp, fn, tn`

## Scenarios

Five standardized scenarios with "FIXED" distances for convergence within episode limits:

| Scenario | Agents | Description | Key Parameters |
|----------|--------|-------------|----------------|
| `head_on` | 2 | Direct approach on reciprocal headings | approach_nm=18.0 |
| `t_formation` | 3 | Horizontal bar + vertical stem crossing | arm_nm=7.5, stem_nm=10.0 |
| `parallel` | 3 | In-trail same-direction with 8 NM gaps | gaps_nm=8.0, south_nm=18.0 |
| `converging` | 4 | Multiple aircraft to clustered waypoints | radius_nm=12.0 (reduced from 25) |
| `canonical_crossing` | 4 | Orthogonal 4-way intersection | radius_nm=12.5 |

**Scenario generation:** Python helpers in `src.scenarios.scenario_generator`:
```bash
# Generate all scenarios with FIXED distances
python -c "from src.scenarios.scenario_generator import *; [make_head_on(), make_t_formation(), make_parallel(), make_converging(), make_canonical_crossing()]"

# Quick visualization
python src/scenarios/visualize_scenarios.py  # → scenario_plots/*.png
```

All scenarios centered at 52.0°N, 4.0°E (Netherlands airspace) at FL100 with 250kt cruise speed.

## Testing & Robustness

### Targeted Shifts

Individual agent modifications to test single points of failure. Algorithm restoration via `Algorithm.from_checkpoint` with `shared_policy`. Environment config matches training exactly to avoid observation space mismatches.

**Shift generation:** Speed/position/heading/type/waypoint modifications:
- **Micro range:** ±5-15 units (small perturbations within training envelope)
- **Macro range:** ±20-40 units (large deviations to test failure modes)
- **Position shifts:** Move agents closer to increase conflict probability
- **Targeted per-agent:** Only one agent modified per test case

**Per-shift folder structure:**
```
shifts/speed_micro_A2_+10kt/
├── trajectory_speed_micro_A2_+10kt_ep0.json    # Agent positions/actions over time
├── traj_speed_micro_A2_+10kt_ep0.csv          # Rich CSV with hallucination data
├── summary_speed_micro_A2_+10kt_ep0.csv       # Episode-level metrics
└── README.md                                   # Test configuration details
```

**Key CLI command:**
```bash
python atc_cli.py test-shifts --targeted --episodes 5 --viz --scenario parallel
```

**Run metadata:** `targeted_run_metadata.json` with shift ranges, total configurations, episodes per shift.

**Wind configuration:** Environmental shifts include layered wind fields and uniform winds scaled by shift characteristics.

### Baseline vs Shifted Scenarios

Loads baseline scenario (from checkpoint filename pattern), runs all other scenarios, computes performance deltas. Metrics extracted from CSV via hallucination detector. Outputs summary tables + interactive dashboards/maps.

**Key outputs:**
- `baseline_vs_shift_summary.csv`: Statistical comparison across all scenarios
- `summary_f1_score.png`, `summary_path_efficiency.png`: Performance visualizations  
- Interactive trajectory maps and comparison plots in `<model>__visualizations/` directories

## Analysis & Visualization

### Hallucination Detector

**Method:** TCPA/DCPA ground truth vs. policy action patterns with IoU-based window matching for robust event-level evaluation.

**Core metrics:**
- **Confusion matrix:** TP/FP/FN/TN (step-level) → precision/recall/F1 (event-level post-IoU)
- **Alert burden:** duty_cycle, alerts_per_min, total_alert_time_s (operator workload)
- **Lead time:** avg_lead_time_s (negative=early alerts, positive=late alerts)
- **Resolution:** TP_res/FP_res/FN_res within 60s after alert (intervention success)
- **Efficiency:** path_efficiency, flight_time_s, waypoint_completion_rate
- **Stability:** action_oscillation_rate (behavioral smoothness)

**Real-time usage:** Embedded in environment during training/testing via `enable_hallucination_detection=True`

**Offline usage:** CSV trajectory analysis via `HallucinationDetector().compute()` with trajectory dict input

### Visualization Capabilities

Comprehensive analysis package in `src/analysis/` - see [analysis/README.md](src/analysis/README.md) for full capabilities.

**Figure types supported:**
- **Geographic maps:** Interactive Folium with trajectory overlays, safety circles, conflict heatmaps
- **Temporal plots:** Plotly time series, animated trajectories, degradation curves
- **Publication figures:** Matplotlib with bootstrap confidence intervals, vulnerability matrices
- **Comparison dashboards:** Baseline vs shifted trajectory analysis

**Make all figures:**
```bash
python src/analysis/make_all_figures.py --results-dir results/
```

**Dependencies:** folium>=0.14.0, plotly>=5.15.0, matplotlib>=3.6.0, scikit-learn>=1.2.0

## CLI Reference

Complete command-line interface with copy-pasteable examples:

**generate-scenarios:** Create air traffic scenarios
```bash
python atc_cli.py generate-scenarios --all
python atc_cli.py generate-scenarios --scenario head_on --params "approach_nm=20"
```

**train:** Train MARL models with algorithm-specific hyperparameters  
```bash
python atc_cli.py train --scenario canonical_crossing --algo PPO --timesteps 100000 --gpu
python atc_cli.py train --scenario all --timesteps 50000 --checkpoint-every 10000
```

**test-shifts:** Run targeted or unison distribution shift testing
```bash
python atc_cli.py test-shifts --targeted --episodes 5 --viz --scenario parallel
python atc_cli.py test-shifts --checkpoint latest --algo SAC
```

**analyze:** Process training/testing results with hallucination detection
```bash
python atc_cli.py analyze --results-dir results_PPO_head_on_20250923_190203
python atc_cli.py analyze --no-hallucination
```

**visualize:** Generate trajectory plots and analysis figures  
```bash
python atc_cli.py visualize --trajectory traj_ep_0001.csv
python atc_cli.py visualize --results-dir results_PPO_parallel_latest
```

**full-pipeline:** Complete workflow from scenario generation to analysis
```bash
python atc_cli.py full-pipeline --scenario head_on --train-timesteps 50000 --test-episodes 3
```

**list:** Browse available resources
```bash
python atc_cli.py list scenarios
python atc_cli.py list checkpoints  
python atc_cli.py list results
```

## Troubleshooting

**Ray worker issues on Windows:** Set `num_rollout_workers=0` for driver-only sampling if Ray fails to spawn workers.

**Environment config alignment:** Ensure test environment configuration matches training settings (neighbor_topk, collision_nm, team coordination weights) to avoid observation space mismatches.

**GPU auto-detection:** Algorithm configurations automatically detect CUDA availability. Use `--gpu`/`--no-gpu` flags to override.

**Checkpoint restoration:** Use `Algorithm.from_checkpoint()` with `shared_policy` ID. CLI automatically handles proper multi-agent policy mapping restoration.

**BlueSky initialization:** Only one BlueSky instance per process. Environment handles global initialization and cleanup automatically.