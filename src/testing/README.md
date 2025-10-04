# Distribution Shift Robustness Testing

This module implements **systematic distribution shift testing** for MARL collision avoidance policies, enabling quantification of generalization capabilities and identification of failure modes under operational variations.

---

## 📋 Overview

### Key Features

- **Targeted Shifts**: Individual agent modifications isolate single points of failure
- **Baseline vs Shift Matrix**: Cross-scenario generalization assessment
- **5 Shift Categories**: Speed, position, heading, aircraft type, waypoint
- **Micro to Macro Ranges**: Small perturbations (±5-15 units) to large deviations (±20-40 units)
- **Real-Time Hallucination Detection**: Ground truth vs policy action comparison during testing
- **Comprehensive Analysis**: Automatic survival curves, bundle metrics, and interactive visualizations

### Testing Philosophy

Distribution shifts represent **operational realities**:
- Speed variations: Wind effects, aircraft performance differences
- Position shifts: Initial condition uncertainty, ATC handoff errors
- Heading deviations: Navigation errors, late conflict detection
- Aircraft types: Mixed fleet operations (A320 vs B738 vs B77W)
- Waypoint changes: Route modifications, dynamic rerouting

Testing evaluates whether policies trained on **idealized scenarios** maintain safety margins under **realistic variations**.

---

## 🎯 Testing Methodologies

### 1. Targeted Shift Testing

**Purpose:** Identify which agents and shift types cause policy degradation

**Approach:**
- Select 1 agent per test case (others remain at baseline)
- Apply shift to that agent only
- Run 100+ episodes per shift configuration
- Compare performance to baseline (no shifts)

**Shift generation:**
```python
shift_config = {
    "target_agent": "A1",              # Agent to modify
    "shift_type": "speed_kt_delta",    # Shift category
    "shift_value": 20.0,               # +20 kt
    "shift_range": "macro"             # Micro or macro
}
```

**Example shifts:**

| Category | Micro Range | Macro Range | Unit |
|----------|-------------|-------------|------|
| Speed | ±5 to ±15 | ±20 to ±30 | kt |
| Heading | ±5 to ±15 | ±20 to ±30 | degrees |
| Position (lat/lon) | ±0.05 to ±0.15 | ±0.2 to ±0.4 | degrees |
| Position (closer) | -1 to -3 | -4 to -8 | NM toward center |
| Waypoint | ±0.1 to ±0.2 | ±0.3 to ±0.5 | degrees |
| Aircraft type | A320 ↔ B738 | A320 ↔ B77W | categorical |

**CLI command:**
```bash
python atc_cli.py test-shifts --targeted --episodes 100 --viz --scenario head_on
```

**Output structure:**
```
results/targeted_shift_analysis_head_on_20250928/
├── targeted_run_metadata.json        # Test configuration
├── shift_speed_micro_A1_+10kt/      # Per-shift directory
│   ├── traj_ep_0001.csv             # Episode trajectories
│   ├── traj_ep_0002.csv
│   ├── ...
│   ├── summary_speed_micro_A1_+10kt.csv  # Aggregated metrics
│   └── README.md                    # Shift description
├── shift_heading_macro_A2_-25deg/
├── ...
└── visualizations/                   # Interactive plots
    ├── comparison_map_speed_micro.html
    └── degradation_curves.png
```

---

### 2. Baseline vs Shift Matrix

**Purpose:** Quantify cross-scenario generalization (inter-scenario robustness)

**Approach:**
- Train model on Scenario A (e.g., head_on)
- Test on Scenario A (baseline performance)
- Test on Scenarios B, C, D, E (shifted scenarios)
- Compare metrics: LoS risk, hallucination rates, efficiency

**Test matrix:**

|  | Test: canonical | Test: converging | Test: head_on | Test: parallel | Test: t_formation |
|---|---|---|---|---|---|
| **Model: canonical** | ✓ Baseline | Shift | Shift | Shift | Shift |
| **Model: converging** | Shift | ✓ Baseline | Shift | Shift | Shift |
| **Model: head_on** | Shift | Shift | ✓ Baseline | Shift | Shift |
| **Model: parallel** | Shift | Shift | Shift | ✓ Baseline | Shift |
| **Model: t_formation** | Shift | Shift | Shift | Shift | ✓ Baseline |
| **Model: generic** | Test | Test | Test | Test | Test |

**CLI command:**
```bash
python -m src.testing.baseline_vs_shift_matrix \
  --models-dir models \
  --scenarios-dir scenarios \
  --episodes 100 \
  --use-gpu
```

**Output structure:**
```
results_baseline_vs_shift/
├── baseline_vs_shift_summary.csv           # Model-scenario aggregated metrics
├── baseline_vs_shift_detailed_summary.csv  # Episode-level data
├── PPO_head_on__on__head_on__baseline/    # Baseline performance
│   ├── ep_001/, ep_002/, ...
│   ├── episode_metrics.csv
│   └── minsep.png
├── PPO_head_on__on__converging/           # Shifted scenario
│   ├── ep_001/, ep_002/, ...
│   └── episode_metrics.csv
├── ...
└── scenario_centric_visualizations/        # Interactive analysis
    ├── master_scenario_analysis_index.html  # Central dashboard
    └── scenario_head_on_analysis/
        ├── scenario_head_on_index.html
        ├── scenario_head_on_all_models_comparison.html
        └── scenario_head_on_canonical_vs_converging_map.html
```

---

## 📊 Key Metrics

### Safety Metrics

**Loss of Separation (LoS) Risk:**
$$P(\text{min\_sep} < 5 \text{ NM}) = \frac{\text{episodes with LoS}}{\text{total episodes}}$$

**Separation statistics:**
- Mean, median, std of episode minimum separation
- Percentiles: p5, p10, p25, p75, p90, p95
- Min/max separation values

**Collision events:**
- Critical collision count (< 0.5 NM)
- LoS event count (< 5 NM)
- LoS duration (seconds in violation)

### Hallucination Detection Metrics

**Confusion Matrix (Event-Level with IoU Matching):**
- **True Positives (TP)**: Detected conflict that actually exists
- **False Positives (FP)**: Ghost conflict (detected but no threat)
- **False Negatives (FN)**: Missed conflict (threat exists but not detected)
- **True Negatives (TN)**: Correctly identified safe state

**Derived metrics:**
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Alert burden:**
- Duty cycle: Fraction of episode in alert state
- Alerts per minute: Frequency of alert activations
- Total alert time: Cumulative seconds in alert

**Resolution assessment (60s post-alert window):**
- TP resolved: Conflict resolved after true alert
- FP resolved: False alert followed by no conflict
- FN resolved: Missed conflict that resolved anyway

### Efficiency Metrics

**Path efficiency:**
$$\text{Efficiency} = \frac{\text{Direct distance}}{\text{Actual path length}}$$

**Flight time:**
- Episode duration (seconds)
- Waypoint completion rate (fraction of agents reaching goals)

**Action smoothness:**
- Action oscillation rate: Frequency of direction changes
- Average action magnitude: Control input intensity

---

## 🚀 Usage Examples

### Targeted Shift Testing

**Basic targeted test:**
```bash
python atc_cli.py test-shifts --targeted --episodes 100 --viz
```

**With specific checkpoint:**
```bash
python atc_cli.py test-shifts \
  --targeted \
  --checkpoint models/PPO_head_on_20250928 \
  --episodes 100 \
  --viz
```

**With custom seeds (for reproducibility):**
```bash
python -m src.testing.targeted_shift_tester \
  --checkpoint models/PPO_head_on_20250928 \
  --scenario head_on \
  --episodes 100 \
  --seeds 42,123,456,789,101112 \
  --viz
```

**Programmatic usage:**
```python
from src.testing.targeted_shift_tester import TargetedShiftTester

tester = TargetedShiftTester(
    checkpoint_path="models/PPO_head_on_20250928",
    scenario_path="scenarios/head_on.json",
    output_dir="results/my_targeted_test"
)

# Run all shift configurations
results = tester.run_all_shifts(
    episodes_per_shift=100,
    seeds=[42, 123, 456],
    enable_viz=True
)

# Analyze results
print(f"Total configurations: {len(results)}")
print(f"Mean LoS risk: {results['los_risk'].mean():.3f}")
```

---

### Baseline vs Shift Matrix

**Full matrix evaluation:**
```bash
python -m src.testing.baseline_vs_shift_matrix \
  --models-dir models \
  --scenarios-dir scenarios \
  --episodes 100 \
  --use-gpu
```

**Specific model-scenario pairs:**
```bash
python -m src.testing.baseline_vs_shift_matrix \
  --models-dir models \
  --scenarios-dir scenarios \
  --episodes 100 \
  --model-filter "PPO_head_on" \
  --scenario-filter "head_on,converging" \
  --use-gpu
```

**With generic model inclusion:**
```bash
python -m src.testing.baseline_vs_shift_matrix \
  --models-dir models \
  --scenarios-dir scenarios \
  --episodes 100 \
  --include-generic \
  --use-gpu
```

**Programmatic usage:**
```python
from src.testing.baseline_vs_shift_matrix import BaselineVsShiftTester

tester = BaselineVsShiftTester(
    models_dir="models",
    scenarios_dir="scenarios",
    output_dir="results_baseline_vs_shift"
)

# Run full matrix
results = tester.run_full_matrix(
    episodes_per_scenario=100,
    use_gpu=True,
    include_generic=True
)

# Generate summary
summary_df = tester.generate_summary()
summary_df.to_csv("matrix_summary.csv", index=False)
```

---

## 📈 Analysis Workflows

### 1. Survival Curve Analysis

**Purpose:** Quantify episode-level minimum separation reliability

**Command:**
```bash
python src/analysis/survival_curve_analysis.py \
  --data_dir results_baseline_vs_shift \
  --output survival_analysis
```

**Output:**
- `survival_curve_{scenario}.png`: Survival function plots
- `survival_curve_statistics.csv`: Statistical summary

**Interpretation:**
- Survival function: $S(x) = P(\text{min\_sep} \geq x)$
- LoS risk: $1 - S(5.0)$ (probability of separation < 5 NM)
- Median separation: $S^{-1}(0.5)$

See [survival_analysis/README.md](../../survival_analysis/README.md) for detailed results.

---

### 2. Bundle Analysis

**Purpose:** Grouped shift evaluation (kinematics, geometry, environment)

**Command:**
```bash
python src/analysis/shift_bundle_analysis.py \
  --data_dir results_baseline_vs_shift \
  --output bundle_analysis
```

**Bundles:**
- **KINEMATICS**: Speed, heading shifts
- **GEOMETRY**: Position, waypoint shifts
- **AIRFRAME**: Aircraft type variations
- **ENVIRONMENT**: Wind fields (future)
- **CONTROL**: Action delay, sensor noise (future)

**Output:**
- `bundle_{category}/kpi_heatmap.png`: Performance heatmaps
- `bundle_{category}/kpi_stats.csv`: Statistical summary
- `bundle_summary.csv`: Cross-bundle comparison

---

### 3. Interactive Visualizations

**Trajectory comparison maps (automatic during testing):**
```bash
# View results
open results_baseline_vs_shift/scenario_centric_visualizations/master_scenario_analysis_index.html
```

**Features:**
- Geographic Folium maps with trajectory overlays
- Plotly time series with separation, reward, action traces
- Comparison dashboards showing baseline vs shifted performance
- Conflict markers (TP, FP, FN, TN) with time stamps

---

## 🔬 Advanced Configuration

### Custom Shift Definitions

**Create custom shift configuration:**
```python
custom_shifts = {
    "A1": {
        "speed_kt_delta": 25.0,        # +25 kt
        "heading_deg_delta": -20.0,    # -20 degrees
        "position_lat_delta": 0.15,    # +0.15 degrees north
    },
    "A2": {
        "waypoint_lat_delta": -0.3,    # Shift waypoint 0.3° south
    }
}

# Apply to environment
env = MARLCollisionEnv({
    "scenario_path": "scenarios/head_on.json",
    "shift_config": custom_shifts,
    "enable_hallucination_detection": True,
    "enable_trajectory_logging": True
})
```

### Shift Ranges Configuration

**Modify default shift ranges:**
```python
from src.testing.targeted_shift_tester import TargetedShiftTester

tester = TargetedShiftTester(
    checkpoint_path="models/PPO_head_on",
    scenario_path="scenarios/head_on.json"
)

# Override shift ranges
tester.shift_ranges = {
    "speed": {
        "micro": (-10, 10),   # ±10 kt instead of ±5-15
        "macro": (-40, 40)    # ±40 kt instead of ±20-30
    },
    "heading": {
        "micro": (-8, 8),
        "macro": (-35, 35)
    }
}

# Run with custom ranges
results = tester.run_all_shifts(episodes_per_shift=50)
```

### Episode Count Guidelines

**Statistical validity requirements:**

| Test Type | Minimum Episodes | Recommended | Purpose |
|-----------|-----------------|-------------|---------|
| Quick smoke test | 10 | 20 | Sanity check |
| Development testing | 20 | 50 | Debugging |
| Research evaluation | 50 | 100 | Statistical analysis |
| Publication results | 100 | 200+ | Confidence intervals |

**Rationale:**
- Bootstrap CI requires ≥20 episodes
- Rare events (LoS) need ≥50 for detection
- Publication-grade statistics: ≥100 for robustness

---

## 🐛 Troubleshooting

### Common Issues

**1. Checkpoint restoration fails**
```python
# Ensure policy ID matches
algo = PPO.from_checkpoint("path/to/checkpoint")
policy = algo.get_policy("shared_policy")  # Not "default_policy"

# Verify environment config alignment
env_config = {
    "neighbor_topk": 3,        # Must match training
    "collision_nm": 3.0,       # Must match training
    # ...
}
```

**2. No trajectory files found**
```bash
# Check directory structure
ls results/PPO_*/ep_*/traj_*.csv

# Ensure trajectory logging enabled
env_config = {"enable_trajectory_logging": True}
```

**3. Hallucination detection missing**
```python
# Enable during testing (not training)
env_config = {
    "enable_hallucination_detection": True,  # Must be True for testing
    "scenario_path": "scenarios/head_on.json"
}
```

**4. Baseline vs shift matrix incomplete**
```bash
# Check model-scenario matching
python -m src.testing.baseline_vs_shift_matrix \
  --models-dir models \
  --scenarios-dir scenarios \
  --dry-run  # Show matched pairs without running
```

**5. Visualization not loading**
```bash
# Check dependencies
pip install plotly>=5.15.0 folium>=0.14.0

# Check browser compatibility
# Use Chrome/Firefox (Safari may have issues with Folium)
```

### Performance Optimization

**Faster testing:**
1. Reduce episodes: `--episodes 50` (instead of 100)
2. Use GPU: `--use-gpu`
3. Disable visualizations: Remove `--viz` flag
4. Parallel workers: Increase `num_rollout_workers` if not using GPU

**Better statistical validity:**
1. More episodes: `--episodes 200`
2. Multiple seeds: `--seeds 42,123,456,789,101112`
3. Bootstrap CI: Enabled by default in survival analysis

---

## 📊 Expected Results Patterns

### Typical Performance Degradation

**Baseline (training scenario):**
- Mean min sep: 7-12 NM
- LoS risk: 0-10%
- F1 score: 0.7-0.9
- Path efficiency: 0.85-0.95

**Cross-scenario (inter-scenario shift):**
- Mean min sep: 0.5-3 NM (⚠️ 50-80% reduction)
- LoS risk: 50-100% (⚠️ dramatic increase)
- F1 score: 0.3-0.6 (hallucination rate increases)
- Path efficiency: 0.6-0.8 (more evasive maneuvers)

**Targeted shift (micro range):**
- Mean min sep: 5-10 NM (10-30% degradation)
- LoS risk: 10-30%
- F1 score: 0.6-0.8
- Path efficiency: 0.75-0.90

**Targeted shift (macro range):**
- Mean min sep: 2-6 NM (30-60% degradation)
- LoS risk: 30-70%
- F1 score: 0.4-0.7
- Path efficiency: 0.65-0.85

---

## 📚 References

- **Distribution Shift**: Quionero-Candela et al. (2009), Dataset Shift in Machine Learning
- **Survival Analysis**: Kaplan & Meier (1958), Nonparametric Estimation
- **IoU Matching**: Intersection over Union for temporal event alignment
- **TCPA/DCPA**: ICAO Annex 10 conflict detection standards

---

**Module Files:**
- `targeted_shift_tester.py`: Individual agent shift testing (1894 lines)
- `baseline_vs_shift_matrix.py`: Cross-scenario evaluation (1200+ lines)

**Related Modules:**
- [src/environment/](../environment/README.md): Environment with shift support
- [src/analysis/](../analysis/README.md): Hallucination detection and analysis
- [src/training/](../training/README.md): Model training

---

**Output Directories:**
- `results/targeted_shift_analysis_{scenario}_{timestamp}/`: Targeted test results
- `results_baseline_vs_shift/`: Matrix evaluation results
- `survival_analysis/`: Reliability analysis outputs
- `bundle_analysis/`: Grouped shift performance metrics
