# ATC Hallucination Detection Project

**Simulation and Quantification of ML-Based Hallucination Effects on Safety Margins in En Route Control**

> Master's Thesis Project - Air Transport and Logistics  
> Technical University Dresden  
> Author: Somnath Panigrahi

---

## 🎯 Project Overview

This project investigates **hallucination phenomena** in Multi-Agent Reinforcement Learning (MARL) systems applied to air traffic collision avoidance. Hallucinations manifest as:

- **False Alerts (Ghost Conflicts)**: AI predicts conflicts that don't exist, causing unnecessary interventions
- **Missed Conflicts (Invisible Threats)**: AI fails to detect actual collision risks, compromising safety

Using BlueSky air traffic simulation and Ray/RLlib training infrastructure, we train cooperative collision avoidance policies and systematically test their robustness under distribution shifts. Real-time hallucination detection compares ground truth conflict predictions (TCPA/DCPA) against policy action patterns to quantify safety degradation.

### Research Questions

1. **How do distribution shifts affect hallucination rates in trained MARL policies?**
2. **What is the relationship between false alerts and missed conflicts under operational stress?**
3. **Can survival analysis quantify safety margin reliability across scenarios?**
4. **Which shift categories (speed, position, aircraft type) cause the most severe degradation?**

---

## 📊 Methodology Overview

![Methodology Flowchart](docs/methodology_overview.png)

### Training Phase
1. **Scenario Generator** creates standardized conflict geometries (head-on, parallel, converging, t-formation, canonical crossing)
2. **BlueSky Environment** simulates realistic aircraft dynamics with 18-dimensional relative observations
3. **MARL Training** uses PPO/SAC/IMPALA with shared policies and team-based reward shaping
4. **Model Checkpointing** saves best-performing policies for evaluation

### Testing Phase
5. **Distribution Shifts** apply targeted perturbations (speed, position, heading, aircraft type, waypoint)
6. **Baseline vs Shift Testing** compares performance on training scenario vs cross-scenario generalization
7. **Hallucination Detection** identifies false alerts and missed conflicts in real-time
8. **Survival Analysis** quantifies episode-level minimum separation reliability

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Somnathab3/ATC_Hallucination.git
cd ATC_Hallucination

# Install dependencies (Python 3.8-3.10 recommended)
pip install -r requirements.txt

# Install PyTorch with CUDA (optional, for GPU training)
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Complete Pipeline Example

```bash
# 1. Generate conflict scenarios
python atc_cli.py generate-scenarios --all

# 2. Train MARL model (50k timesteps, ~30 min on GPU)
python atc_cli.py train --scenario head_on --algo PPO --timesteps 50000 --gpu

# 3. Test robustness with distribution shifts (100 episodes per shift)
python atc_cli.py test-shifts --targeted --episodes 100 --viz

# 4. Generate survival analysis and bundle reports
python src/analysis/survival_curve_analysis.py --data_dir results_baseline_vs_shift --output survival_analysis
python src/analysis/shift_bundle_analysis.py --data_dir results_baseline_vs_shift --output bundle_analysis

# 5. View interactive visualizations
# Open: results_baseline_vs_shift/scenario_centric_visualizations/master_scenario_analysis_index.html
```

### CLI Quick Reference

```bash
# List available resources
python atc_cli.py list scenarios
python atc_cli.py list checkpoints
python atc_cli.py list results

# Full end-to-end pipeline (single command)
python atc_cli.py full-pipeline --scenario head_on --train-timesteps 50000 --test-episodes 100
```

---

## 📐 Scenario Visualizations

Our five standardized scenarios create inevitable conflicts without intervention:

### Initial Scenario Geometries

<table>
<tr>
<td align="center"><b>Head-On</b><br><img src="docs/scenario_plots/head_on_radar.png" width="200"/><br>2 aircraft, reciprocal headings<br>18 NM approach distance</td>
<td align="center"><b>T-Formation</b><br><img src="docs/scenario_plots/t_formation_radar.png" width="200"/><br>3 aircraft, perpendicular crossing<br>7.5 NM arm, 10 NM stem</td>
<td align="center"><b>Parallel</b><br><img src="docs/scenario_plots/parallel_radar.png" width="200"/><br>3 aircraft, same direction<br>8 NM in-trail spacing</td>
</tr>
<tr>
<td align="center"><b>Converging</b><br><img src="docs/scenario_plots/converging_radar.png" width="200"/><br>4 aircraft, clustered waypoints<br>12 NM radius placement</td>
<td align="center"><b>Canonical Crossing</b><br><img src="docs/scenario_plots/canonical_crossing_radar.png" width="200"/><br>4 aircraft, orthogonal<br>12.5 NM radius, 4-way intersection</td>
<td></td>
</tr>
</table>

**Key Features:**
- Centered at 52.0°N, 4.0°E (Netherlands airspace)
- FL100 altitude, 250kt cruise speed
- "FIXED" distances ensure episode convergence within 100 steps (1000 seconds)

### Trained Policy Behavior

Example collision avoidance trajectories after training:

<table>
<tr>
<td><img src="docs/scenario_plots/Trained Viz/Best_headon.gif" width="250"/><br><i>Head-On: Successful conflict resolution</i></td>
<td><img src="docs/scenario_plots/Trained Viz/best Parallel.gif" width="250"/><br><i>Parallel: In-trail coordination</i></td>
<td><img src="docs/scenario_plots/Trained Viz/Best_run_T_formation_+35.gif" width="250"/><br><i>T-Formation: Multi-agent crossing</i></td>
</tr>
<tr>
<td><img src="docs/scenario_plots/Trained Viz/Converging.gif" width="250"/><br><i>Converging: Clustered waypoint navigation</i></td>
<td><img src="docs/scenario_plots/Trained Viz/cannonical.gif" width="250"/><br><i>Canonical Crossing: 4-way intersection</i></td>
<td></td>
</tr>
</table>

---

## 📈 Key Results: Inter-Scenario Reliability

Our survival analysis reveals dramatic safety degradation when models face unfamiliar scenarios:

<table>
<tr>
<td align="center"><b>Canonical Crossing Model</b><br><img src="docs/survival_analysis/survival_curve_canonical_crossing.png" width="400"/></td>
<td align="center"><b>Converging Model</b><br><img src="docs/survival_analysis/survival_curve_converging.png" width="400"/></td>
</tr>
<tr>
<td align="center"><b>Head-On Model</b><br><img src="docs/survival_analysis/survival_curve_head_on.png" width="400"/></td>
<td align="center"><b>Parallel Model</b><br><img src="docs/survival_analysis/survival_curve_parallel.png" width="400"/></td>
</tr>
<tr>
<td align="center"><b>T-Formation Model</b><br><img src="docs/survival_analysis/survival_curve_t_formation.png" width="400"/></td>
<td align="center"><b>Summary Statistics</b><br>(see docs/survival_analysis/survival_curve_statistics.csv)</td>
</tr>
</table>

### Statistical Summary

| Model (Trained On) | Test Scenario | Mean Min Sep (NM) | LoS Risk (%) | Median Min Sep (NM) |
|-------------------|---------------|-------------------|--------------|---------------------|
| PPO_head_on | **head_on** (baseline) | **13.75** | 3.4% | **13.11** |
| PPO_head_on | converging (shift) | 0.93 | **100.0%** | 0.74 |
| PPO_parallel | **parallel** (baseline) | **7.88** | 0.0% | **7.97** |
| PPO_parallel | converging (shift) | 1.03 | **100.0%** | 0.97 |
| PPO_converging | **converging** (baseline) | **5.61** | 23.8% | **6.25** |
| PPO_converging | canonical_crossing (shift) | 1.74 | **100.0%** | 1.53 |

**Key Findings:**
- ✅ **Baseline performance**: Models maintain 5+ NM separation on training scenarios
- ⚠️ **Cross-scenario failure**: 50-100% LoS risk when tested on different geometries
- 📉 **Generic model**: Trained on dynamic conflicts shows moderate performance across all scenarios
- 🎯 **Trade-off confirmed**: Scenario-specific models excel at training task but fail to generalize

*Full statistics: [survival_analysis/survival_curve_statistics.csv](survival_analysis/survival_curve_statistics.csv)*

---

## 🏗️ Project Architecture

This project follows a modular design with specialized components for each phase of the research workflow:

### Core Modules

<table>
<tr>
<th>Module</th>
<th>Purpose</th>
<th>Key Components</th>
<th>Documentation</th>
</tr>
<tr>
<td><code>src/environment/</code></td>
<td>MARL collision avoidance environment</td>
<td>
• BlueSky integration<br>
• 18D relative observations<br>
• Team-based PBRS rewards<br>
• Real-time hallucination detection
</td>
<td><a href="src/environment/README.md">README</a></td>
</tr>
<tr>
<td><code>src/scenarios/</code></td>
<td>Standardized conflict generation</td>
<td>
• 5 geometric conflict types<br>
• Parametric scenario builder<br>
• Interactive visualizations<br>
• JSON export
</td>
<td><a href="src/scenarios/README.md">README</a></td>
</tr>
<tr>
<td><code>src/training/</code></td>
<td>Multi-algorithm MARL training</td>
<td>
• PPO/SAC/IMPALA/CQL/APPO<br>
• Shared policy architecture<br>
• Checkpoint management<br>
• Training progress tracking
</td>
<td><a href="src/training/README.md">README</a></td>
</tr>
<tr>
<td><code>src/testing/</code></td>
<td>Distribution shift robustness</td>
<td>
• Targeted agent shifts<br>
• Baseline vs shift matrix<br>
• 5 shift categories<br>
• Automated test execution
</td>
<td><a href="src/testing/README.md">README</a></td>
</tr>
<tr>
<td><code>src/analysis/</code></td>
<td>Academic analysis & visualization</td>
<td>
• Hallucination detection<br>
• Survival curve analysis<br>
• Bundle performance metrics<br>
• Interactive dashboards
</td>
<td><a href="src/analysis/README.md">✓ Existing</a></td>
</tr>
</table>

### Data Flow

```
Scenarios (JSON) → Environment (BlueSky) → Training (Ray/RLlib) → Models (Checkpoints)
                                                                         ↓
Results (CSV/JSON) ← Analysis (Detection) ← Testing (Shifts) ← Models
       ↓
Visualizations (HTML/PNG) + Reports (CSV)
```

---

## 🔬 Technical Details

### Environment: Multi-Agent Observations

Each agent receives **18-dimensional relative observations** (no raw lat/lon for generalization):

**Navigation (6D):**
- `wp_dist_norm`: Normalized distance to waypoint (tanh)
- `cos_to_wp`, `sin_to_wp`: Direction to waypoint (unit circle)
- `airspeed`: Normalized speed around 150 m/s
- `progress_rate`: Waypoint approach rate
- `safety_rate`: Minimum separation change rate

**Neighbor Awareness (12D = 3 neighbors × 4 features):**
- `x_r`, `y_r`: Relative positions of top-3 nearest neighbors
- `vx_r`, `vy_r`: Relative velocities

**Action Space (2D continuous):**
- Heading change: ±18° per 10-second step
- Speed change: ±10 kt per step

*Full specification: [src/environment/README.md](src/environment/README.md)*

### Reward System: Team-Based PBRS

Unified reward components (no double-counting):

$$R_{total} = R_{progress} + R_{violations} + R_{drift} + R_{team} + R_{action} + R_{time}$$

**Key components:**
- **Signed progress**: ±0.04 per km toward/away from waypoint
- **Well-clear violations**: Entry penalty (-25) + depth-scaled step penalties (-1.0×)
- **Drift improvement**: +0.01 per degree of heading optimization
- **Team PBRS**: Shared potential function with 5 NM neighbor sensitivity (weight=0.6)
- **Action costs**: -0.01 per non-neutral control input
- **Time penalty**: -0.0005 per second for efficiency

*Detailed formulas: [src/environment/README.md](src/environment/README.md)*

### Hallucination Detection: TCPA/DCPA vs Action Patterns

**Ground Truth (Physics-Based):**
$$TCPA = \frac{-(\Delta x \cdot \Delta v_x + \Delta y \cdot \Delta v_y)}{|\Delta v|^2}$$
$$DCPA = \sqrt{(\Delta x + \Delta v_x \cdot TCPA)^2 + (\Delta y + \Delta v_y \cdot TCPA)^2}$$
- Conflict flagged if: TCPA ∈ [0, 120s] AND DCPA < 5 NM

**Policy Prediction (Behavior-Based):**
- Alert detection: |Δheading| > 3° OR |Δspeed| > 5 kt
- Intent-aware filtering: Ignore actions toward waypoint
- Threat-aware gating: Require near-term intruder presence

**IoU-Based Event Matching:**
$$IoU = \frac{|T_{GT} \cap T_{Pred}|}{|T_{GT} \cup T_{Pred}|}$$
- Windows matched if IoU ≥ 0.3
- Metrics: Precision, Recall, F1 (event-level, not step-level)

*Implementation: [src/analysis/README.md](src/analysis/README.md)*

---

## 📚 Usage Guides

### 1. Training a New Model

```bash
# Train on single scenario (recommended starting point)
python atc_cli.py train --scenario head_on --algo PPO --timesteps 100000 --gpu

# Train on all scenarios sequentially
python atc_cli.py train --scenario all --algo PPO --timesteps 50000

# Algorithm comparison (PPO vs SAC)
python atc_cli.py train --scenario converging --algo SAC --timesteps 100000 --gpu
```

**Output:** `models/PPO_head_on_YYYYMMDD_HHMMSS/` with checkpoints and training logs

*Detailed guide: [src/training/README.md](src/training/README.md)*

### 2. Running Distribution Shift Tests

```bash
# Targeted shifts (individual agent modifications)
python atc_cli.py test-shifts --targeted --episodes 100 --viz

# Baseline vs shift matrix (cross-scenario evaluation)
```bash
python -m src.testing.intershift_matrix \
  --models-dir models \
  --scenarios-dir scenarios \
  --episodes 100 \
  --use-gpu
```

**Output:** `results_baseline_vs_shift/` with episode CSVs, metrics, and interactive visualizations

*Detailed guide: [src/testing/README.md](src/testing/README.md)*

### 3. Generating Analysis Reports

```bash
# Survival curves (reliability analysis)
python src/analysis/survival_curve_analysis.py \
  --data_dir results_baseline_vs_shift \
  --output survival_analysis

# Bundle analysis (grouped shift evaluation)
python src/analysis/shift_bundle_analysis.py \
  --data_dir results_baseline_vs_shift \
  --output bundle_analysis
```

**Output:** PNG plots, CSV statistics, and HTML dashboards

*Detailed guide: [src/analysis/README.md](src/analysis/README.md)*

---

## 🔧 Development Resources

### Configuration Files

- `requirements.txt`: Python dependencies
- `setup.py`: Package installation
- `.github/copilot-instructions.md`: AI assistant project context
- `atc_cli.py`: Unified command-line interface

### Directory Structure

```
ATC_Hallucination/
├── src/                          # Source code modules
│   ├── environment/              # MARL environment (BlueSky wrapper)
│   ├── scenarios/                # Conflict scenario generation
│   ├── training/                 # Multi-algorithm training
│   ├── testing/                  # Distribution shift testing
│   └── analysis/                 # Hallucination detection & visualization
├── scenarios/                    # Generated scenario JSON files
│   └── scenario_plots/           # Interactive HTML radar plots
├── models/                       # Trained model checkpoints
├── results/                      # Test execution outputs
├── survival_analysis/            # Reliability analysis results
├── episode_gifs/                 # Training trajectory animations
├── atc_cli.py                   # Main CLI entry point
├── train.py                     # Legacy training script
└── README.md                    # This file
```

### Extending the Project

**Adding New Scenarios:**
See [src/scenarios/README.md](src/scenarios/README.md) for scenario generation API

**Adding New Reward Components:**
See [src/environment/README.md](src/environment/README.md) for reward system architecture

**Adding New Analysis Metrics:**
See [src/analysis/README.md](src/analysis/README.md) for hallucination detector extension

**Adding New RL Algorithms:**
See [src/training/README.md](src/training/README.md) for RLlib integration patterns

---

## 📖 Academic Context

This project supports research on:

### 1. Distribution Shift Robustness in Safety-Critical AI
- **Research Question**: How do operational variations affect learned safety policies?
- **Approach**: Targeted single-agent shifts isolate failure modes
- **Metrics**: LoS risk, false alert rate, missed conflict rate

### 2. Hallucination Detection Methodologies
- **Research Question**: Can we detect conflict prediction errors in real-time?
- **Approach**: Compare ground truth (TCPA/DCPA) vs behavioral proxy (actions)
- **Metrics**: Precision, Recall, F1 (event-level with IoU matching)

### 3. Multi-Agent Coordination Under Stress
- **Research Question**: Does team-based reward shaping maintain coordination under distribution shift?
- **Approach**: PBRS with shared potential function and neighbor sensitivity
- **Metrics**: Team reward contribution, pairwise separation maintenance

### 4. Safety Margin Reliability Analysis
- **Research Question**: What is the probability distribution of minimum separation?
- **Approach**: Survival analysis on episode-level minimum separation
- **Metrics**: Survival curves, percentile statistics, LoS risk rates

---

## 🐛 Troubleshooting

### Common Issues

**1. BlueSky initialization errors**
```bash
# Only one BlueSky instance per process - restart Python if needed
# Environment handles automatic initialization and cleanup
```

**2. Ray worker failures on Windows**
```python
# If Ray fails to spawn workers, use driver-only mode:
env_config = {"num_rollout_workers": 0}
```

**3. Environment observation space mismatches**
```python
# Ensure test environment matches training configuration:
env_config = {
    "neighbor_topk": 3,              # Must match training
    "collision_nm": 3.0,             # Must match training
    "team_coordination_weight": 0.2, # Must match training
}
```

**4. Checkpoint restoration issues**
```python
# Use Algorithm.from_checkpoint() with correct policy ID:
from ray.rllib.algorithms.ppo import PPO
algo = PPO.from_checkpoint("models/PPO_head_on_20250928/")
policy = algo.get_policy("shared_policy")  # Note: "shared_policy" is the correct ID
```

**5. Missing visualization files**
```bash
# Check that hallucination detection is enabled:
env_config = {"enable_hallucination_detection": True}

# Verify trajectory CSV generation:
ls results/PPO_*/ep_*/traj_*.csv
```

### Performance Optimization

- **GPU Training**: Use `--gpu` flag for 3-5× speedup
- **Parallel Workers**: Increase `num_rollout_workers` for faster sampling (4-8 on multi-core CPUs)
- **Episode Count**: Use 100+ episodes for statistical validity in analysis
- **Checkpoint Frequency**: Save every 10k-20k timesteps to balance storage vs granularity

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@mastersthesis{panigrahi2025atc_hallucination,
  title={Simulation and Quantification of ML-Based Hallucination Effects on Safety Margins in En Route Control},
  author={Panigrahi, Somnath},
  year={2025},
  school={Technical University Dresden},
  department={Air Transport and Logistics}
}
```

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with appropriate tests
4. Update relevant README files in `src/` subfolders
5. Commit with descriptive messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📞 Contact

**Somnath Panigrahi**  
Master's Student - Air Transport and Logistics  
Technical University Dresden

**Project Repository:** [https://github.com/Somnathab3/ATC_Hallucination](https://github.com/Somnathab3/ATC_Hallucination)

---

## 🙏 Acknowledgments

- **BlueSky**: Open-source air traffic simulator (TU Delft)
- **Ray/RLlib**: Scalable reinforcement learning framework (Anyscale)
- **PettingZoo**: Multi-agent environment standardization (Farama Foundation)
- **Technical University Dresden**: Academic supervision and resources

---

**Last Updated:** October 2025  
**Project Status:** Active Development (Thesis Completion Q1 2025)

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