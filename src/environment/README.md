# Multi-Agent Collision Avoidance Environment

This module implements the core **Multi-Agent Reinforcement Learning (MARL) environment** for air traffic collision avoidance using BlueSky as the flight dynamics simulator. It provides a PettingZoo-compatible interface for training cooperative policies with Ray/RLlib.

---

## 📋 Overview

### Key Features

- **BlueSky Integration**: Realistic flight dynamics simulation with Bluesky ATC simulator
- **Relative Observations**: 18D observation space with no raw lat/lon for generalization
- **Team-Based PBRS**: Potential-based reward shaping for multi-agent coordination
- **Real-Time Hallucination Detection**: Optional ground truth vs policy action comparison
- **Comprehensive Logging**: Rich trajectory CSVs with 40+ columns for analysis
- **Distribution Shift Support**: Dynamic scenario modification for robustness testing
- **Wind Field Simulation**: Environmental perturbations for realistic testing

### Architecture

```
MARLCollisionEnv (PettingZoo ParallelEnv)
    ├── BlueSky Simulator (flight dynamics)
    ├── Observation Builder (relative state extraction)
    ├── Reward System (unified components)
    ├── Hallucination Detector (optional, real-time)
    └── Trajectory Logger (CSV/JSON export)
```

---

## 🔍 Environment Specification

### State Space

**Each agent observes 18 dimensions:**

| Category | Features | Dimensions | Range | Description |
|----------|----------|------------|-------|-------------|
| **Navigation** | `wp_dist_norm` | 1 | [-1, 1] | Tanh-normalized distance to waypoint |
| | `cos_to_wp` | 1 | [-1, 1] | Cosine of direction to waypoint |
| | `sin_to_wp` | 1 | [-1, 1] | Sine of direction to waypoint |
| | `airspeed` | 1 | [-10, 10] | Normalized airspeed (mean=150 m/s, std=30) |
| | `progress_rate` | 1 | [-1, 1] | Rate of waypoint approach (+forward, -backtrack) |
| | `safety_rate` | 1 | [-1, 1] | Minimum separation change rate (+safer, -riskier) |
| **Neighbors** | `x_r` (×3) | 3 | [-12, 12] | Relative X positions of top-3 neighbors (km) |
| | `y_r` (×3) | 3 | [-12, 12] | Relative Y positions of top-3 neighbors (km) |
| | `vx_r` (×3) | 3 | [-150, 150] | Relative X velocities (m/s) |
| | `vy_r` (×3) | 3 | [-150, 150] | Relative Y velocities (m/s) |

**Total:** 6 (own) + 12 (neighbors) = **18 dimensions**

**Observation Padding:**
- If fewer than 3 neighbors exist, observations are zero-padded
- Neighbors ranked by Euclidean distance (closest first)

**Normalization Details:**
```python
wp_dist_norm = tanh(dist_km / 50.0)  # Soft saturation at 50 km
airspeed_norm = (airspeed_ms - 150.0) / 30.0  # Z-score normalization
progress_rate = clamp(wp_approach_rate / 10.0, -1, 1)  # ±10 m/s nominal
safety_rate = clamp(minsep_change_rate / 5.0, -1, 1)   # ±5 NM/step nominal
```

### Action Space

**Continuous 2D space:**
- **Heading change**: [-1, 1] → scaled to ±18° per 10-second step
- **Speed change**: [-1, 1] → scaled to ±10 kt per step

**Scaling formula:**
```python
actual_heading_change = action[0] * 18.0  # degrees
actual_speed_change = action[1] * 10.0    # knots
```

**Action constraints:**
- Actions applied every 10 seconds (configurable via `action_period_s`)
- BlueSky handles physical constraints (max bank angle, acceleration limits)
- Neutral action [0, 0] maintains current heading and speed

### Episode Termination

An episode terminates when **any** condition is met:

1. **Time limit**: 100 steps (1000 seconds) reached
2. **Waypoint completion**: All agents within 1 NM of their waypoints
3. **Critical collision**: Any pairwise separation < 0.5 NM (emergency cutoff)

**Truncation vs Termination:**
- `truncated = True`: Time limit reached
- `terminated = True`: Waypoint completion or critical collision

---

## 🎁 Reward System

### Unified Reward Function

The environment uses a **single reward value per agent** composed of multiple components. This avoids double-counting and ensures clean credit assignment:

$$R_{total} = R_{progress} + R_{violations} + R_{drift} + R_{team} + R_{action} + R_{time}$$

### Component Breakdown

#### 1. Signed Progress Reward
**Purpose:** Encourage movement toward waypoint, penalize backtracking

$$R_{progress} = k_{prog} \cdot \Delta d_{wp}$$

Where:
- $\Delta d_{wp}$ = distance change to waypoint (negative if approaching, positive if retreating)
- $k_{prog}$ = 0.04 (reward per km of progress)

**Example:**
- Moving 1 km closer to waypoint: +0.04
- Drifting 0.5 km away: -0.02

#### 2. Well-Clear Violation Penalties
**Purpose:** Penalize separation violations with severity-based scaling

$$R_{violations} = \begin{cases}
R_{entry} & \text{if first violation timestep} \\
R_{entry} + k_{step} \cdot \text{depth} & \text{if ongoing violation}
\end{cases}$$

Where:
- $R_{entry}$ = -25.0 (one-time entry penalty)
- $k_{step}$ = -1.0 (per-step penalty scaling)
- $\text{depth}$ = severity factor based on separation distance

**Severity calculation:**
```python
if min_sep < DEEP_BREACH_NM (1.0 NM):
    depth = (DEEP_BREACH_NM - min_sep) / DEEP_BREACH_NM * 2.0
else:
    depth = (collision_nm - min_sep) / (collision_nm - DEEP_BREACH_NM)
```

**Example episode:**
- Step 50: First violation at 4.5 NM → -25.0 (entry) + -0.1 (depth) = -25.1
- Step 51: Still violating at 4.0 NM → -0.2 (depth only)
- Step 52: Separation restored to 5.5 NM → 0.0

#### 3. Drift Improvement Reward
**Purpose:** Reward heading adjustments toward waypoint

$$R_{drift} = k_{drift} \cdot \max(0, |\text{drift}_{prev}| - |\text{drift}_{curr}|)$$

Where:
- $\text{drift}$ = angular difference between current heading and waypoint bearing
- $k_{drift}$ = 0.01 (reward per degree of improvement)
- Deadzone of ±8° to prevent oscillation penalties

**Example:**
- Drift was 30°, now 20° → +0.1 (10° improvement)
- Drift was 5°, now 3° → 0.0 (within deadzone)

#### 4. Team Coordination Reward (PBRS)
**Purpose:** Encourage cooperative separation maintenance

$$R_{team} = \gamma_{team} \cdot \Phi(s_{t+1}) - \Phi(s_t)$$

Where:
- $\Phi(s)$ = team potential function (shared across agents)
- $\gamma_{team}$ = 0.99 (team discount factor)

**Potential function:**
```python
Φ(s) = Σ pairwise_safety_potential(agent_i, agent_j)
     = Σ sigmoid((separation_ij - threshold) / sensitivity)
```

With:
- `threshold` = 5.0 NM (well-clear boundary)
- `sensitivity` = 5.0 NM (shaping smoothness)
- Team weight = 0.2 (fraction of total reward from team component)

**Distribution modes:**
- `responsibility`: Each agent gets full $\Delta\Phi$ (default)
- `shared`: $\Delta\Phi$ divided equally among all agents

**Example:**
- Pairwise separation improves 5 NM → 6 NM: +0.02 (per agent)
- Pairwise separation degrades 6 NM → 4 NM: -0.03

#### 5. Action Cost Penalty
**Purpose:** Encourage smooth control, penalize unnecessary maneuvers

$$R_{action} = k_{action} \cdot (|\Delta hdg| + |\Delta spd|)$$

Where:
- $k_{action}$ = -0.01 (cost per action unit)
- Action units: normalized [-1, 1] scale

**Example:**
- Neutral action [0, 0] → 0.0
- Max heading change [1.0, 0] → -0.01
- Combined maneuver [0.5, -0.3] → -0.008

#### 6. Time Penalty
**Purpose:** Encourage efficiency (faster waypoint completion)

$$R_{time} = k_{time} \cdot \Delta t$$

Where:
- $k_{time}$ = -0.0005 (penalty per second)
- $\Delta t$ = 10 seconds (action period)

**Per-step penalty:** -0.005 (constant)

#### 7. Terminal Rewards
**On episode completion:**
- **Waypoint reached**: +10.0
- **Episode timeout without reaching waypoint**: -10.0
- **Critical collision**: Automatic termination (no additional penalty beyond violation)

### Reward Magnitudes (Typical Episode)

| Component | Typical Range | Dominant Scenarios |
|-----------|---------------|-------------------|
| Progress | ±0.5 per step | Forward motion (+), backtracking (-) |
| Violations | -25 to -50 | Conflict entry and persistence |
| Drift | 0 to +0.2 | Course corrections toward waypoint |
| Team PBRS | ±0.1 | Multi-agent proximity changes |
| Action cost | -0.02 | Active maneuvering |
| Time | -0.005 | Every step |
| **Total** | **-10 to +2** per step | Varies by conflict phase |

**Episode totals:**
- Successful conflict avoidance: +50 to +150 (waypoint bonus + progress + team)
- Collision episode: -200 to -400 (violations dominate)

---

## 🔧 Configuration Parameters

### Environment Config Dictionary

```python
env_config = {
    # Scenario and initialization
    "scenario_path": "scenarios/head_on.json",  # Path to scenario JSON
    "seed": 42,                                  # Random seed for reproducibility
    
    # Observation space
    "neighbor_topk": 3,                          # Number of nearest neighbors to observe
    
    # Safety thresholds
    "collision_nm": 3.0,                         # Well-clear violation threshold (NM)
    "waypoint_threshold_nm": 1.0,                # Waypoint capture distance (NM)
    
    # Reward system
    "progress_reward_per_km": 0.04,              # Progress reward coefficient
    "time_penalty_per_sec": -0.0005,             # Time penalty coefficient
    "reach_reward": 10.0,                        # Waypoint completion bonus
    "violation_entry_penalty": -25.0,            # First violation penalty
    "violation_step_scale": -1.0,                # Ongoing violation scaling
    "deep_breach_nm": 1.0,                       # Threshold for steeper penalties
    "drift_improve_gain": 0.01,                  # Drift improvement reward
    "drift_deadzone_deg": 8.0,                   # Drift penalty deadzone
    "action_cost_per_unit": -0.01,               # Action magnitude cost
    "terminal_not_reached_penalty": -10.0,       # Timeout penalty
    
    # Team coordination (PBRS)
    "team_coordination_weight": 0.2,             # Team reward fraction (0-1)
    "team_gamma": 0.99,                          # Team potential discount
    "team_share_mode": "responsibility",         # Distribution mode
    "team_ema": 0.001,                           # EMA smoothing for stability
    "team_cap": 0.01,                            # Max team reward magnitude
    "team_anneal": 1.0,                          # Annealing factor (1.0=no anneal)
    "team_neighbor_threshold_km": 10.0,          # Neighbor proximity threshold
    
    # Simulation
    "max_steps": 100,                            # Episode step limit
    "action_period_s": 10.0,                     # Action frequency (seconds)
    "sim_dt_s": 1.0,                             # BlueSky timestep (seconds)
    
    # Testing features
    "enable_hallucination_detection": False,     # Real-time detection (testing only)
    "enable_trajectory_logging": True,           # CSV trajectory export
    "log_dir": "trajectories",                   # Trajectory output directory
    
    # Distribution shifts (for testing)
    "shift_config": None,                        # Dict of agent-specific shifts
    # Example: {"A1": {"speed_kt_delta": +10}, "A2": {"heading_deg_delta": -15}}
}
```

### Shift Configuration Format

For distribution shift testing, provide per-agent modifications:

```python
shift_config = {
    "A1": {
        "speed_kt_delta": 10.0,           # Add 10 kt to initial speed
        "heading_deg_delta": -15.0,       # Subtract 15° from initial heading
        "position_lat_delta": 0.1,        # Move 0.1° north
        "position_lon_delta": 0.05,       # Move 0.05° east
        "waypoint_lat_delta": -0.2,       # Shift waypoint 0.2° south
        "aircraft_type": "B738",          # Change aircraft type
    },
    "A2": {
        "speed_kt_delta": -5.0,           # Reduce speed by 5 kt
    }
    # Other agents inherit baseline scenario values
}
```

---

## 📊 Trajectory Logging

### CSV Output Format

When `enable_trajectory_logging=True`, episodes are saved as CSV files with 40+ columns:

#### Identification Columns
- `episode_id`: Episode number
- `step_idx`: Timestep index (0-99)
- `sim_time_s`: Simulation time in seconds

#### Aircraft State
- `agent_id`: Aircraft identifier (A0, A1, A2, ...)
- `lat_deg`, `lon_deg`: Position (degrees)
- `alt_ft`: Altitude (feet)
- `hdg_deg`: Heading (degrees, 0-360)
- `tas_kt`: True airspeed (knots)
- `cas_kt`: Calibrated airspeed (knots)
- `vs_fpm`: Vertical speed (feet per minute)

#### Actions
- `action_hdg_delta_deg`: Applied heading change
- `action_spd_delta_kt`: Applied speed change
- `action_raw_hdg`: Raw normalized action [-1, 1]
- `action_raw_spd`: Raw normalized action [-1, 1]

#### Rewards (Per Component)
- `reward`: Total reward
- `reward_progress`: Progress component
- `reward_violations`: Violation penalties
- `reward_drift`: Drift improvement
- `reward_team`: Team PBRS component
- `reward_action`: Action cost
- `reward_time`: Time penalty

#### Safety Metrics
- `min_separation_nm`: Minimum pairwise separation
- `conflict_flag`: Well-clear violation (0/1)
- `collision_flag`: Critical collision (0/1)
- `los_event_id`: Loss of separation event identifier

#### Pairwise Distances
- `dist_to_A0_nm`, `dist_to_A1_nm`, ...: Distance to each other agent

#### Navigation
- `wp_dist_nm`: Distance to assigned waypoint
- `wp_bearing_deg`: Bearing to waypoint
- `wp_drift_deg`: Angular difference (heading - bearing)
- `waypoint_reached`: Completion flag (0/1)
- `waypoint_hits`: Cumulative completions

#### Hallucination Detection (if enabled)
- `gt_conflict`: Ground truth conflict flag (TCPA/DCPA)
- `predicted_alert`: Policy-detected alert flag
- `tp`, `fp`, `fn`, `tn`: Confusion matrix flags
- `tcpa_s`: Time to closest point of approach
- `dcpa_nm`: Distance at closest approach

### JSON Episode Metadata

Alongside CSVs, metadata is saved in JSON format:

```json
{
  "episode_id": 1,
  "scenario_name": "head_on",
  "seed": 42,
  "shift_config": {...},
  "total_steps": 85,
  "completion_status": "waypoints_reached",
  "episode_reward": 127.5,
  "min_separation_nm": 5.2,
  "num_los_events": 0,
  "agents": {
    "A1": {
      "initial_pos": [52.0, 4.0],
      "final_pos": [52.15, 4.01],
      "waypoint_reached": true,
      "total_reward": 63.2
    },
    ...
  }
}
```

---

## 🚀 Usage Examples

### Basic Training Setup

```python
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from src.environment.marl_collision_env_minimal import MARLCollisionEnv

# Register environment
def env_creator(env_config):
    return ParallelPettingZooEnv(MARLCollisionEnv(env_config))

register_env("marl_collision", env_creator)

# Training configuration
config = {
    "env": "marl_collision",
    "env_config": {
        "scenario_path": "scenarios/head_on.json",
        "neighbor_topk": 3,
        "collision_nm": 3.0,
        "team_coordination_weight": 0.2,
        "enable_trajectory_logging": True,
    },
    "framework": "torch",
    "num_rollout_workers": 4,
    "train_batch_size": 4096,
    "sgd_minibatch_size": 512,
    "num_sgd_iter": 10,
    "lr": 5e-4,
    "gamma": 0.995,
}

# Create and train algorithm
algo = PPO(config=config)
for i in range(100):
    result = algo.train()
    print(f"Iteration {i}: reward={result['episode_reward_mean']:.2f}")

algo.save("models/my_model")
```

### Testing with Distribution Shifts

```python
# Load trained model
algo = PPO.from_checkpoint("models/PPO_head_on_20250928/")

# Test with speed shift
env = MARLCollisionEnv({
    "scenario_path": "scenarios/head_on.json",
    "shift_config": {
        "A1": {"speed_kt_delta": 20.0}  # A1 20 kt faster
    },
    "enable_hallucination_detection": True,
    "enable_trajectory_logging": True,
    "log_dir": "shifted_test"
})

# Run episode
obs, info = env.reset()
done = False
while not done:
    actions = {aid: algo.compute_single_action(o, policy_id="shared_policy") 
               for aid, o in obs.items()}
    obs, rewards, terminateds, truncateds, infos = env.step(actions)
    done = all(terminateds.values()) or all(truncateds.values())

# Trajectory saved to: shifted_test/traj_ep_0001.csv
```

### Analyzing Hallucination Detection Results

```python
import pandas as pd

# Load trajectory CSV
df = pd.read_csv("shifted_test/traj_ep_0001.csv")

# Episode-level metrics (aggregated in last row per agent)
episode_metrics = df.groupby('agent_id').last()

print("Hallucination Detection Summary:")
print(f"  True Positives:  {episode_metrics['tp'].sum()}")
print(f"  False Positives: {episode_metrics['fp'].sum()}")
print(f"  False Negatives: {episode_metrics['fn'].sum()}")
print(f"  Precision: {episode_metrics['precision'].mean():.3f}")
print(f"  Recall:    {episode_metrics['recall'].mean():.3f}")
print(f"  F1 Score:  {episode_metrics['f1_score'].mean():.3f}")
```

---

## 🔬 Technical Implementation Details

### BlueSky Integration

**Initialization:**
- Environment handles global BlueSky initialization on first instantiation
- Only one BlueSky instance allowed per Python process
- Automatic cleanup registered with `atexit`

**Simulation loop:**
1. Actions accumulated during `action_period_s` (10 seconds)
2. BlueSky advances simulation in 1-second timesteps
3. Aircraft positions/headings/speeds queried after each BlueSky step
4. Observations constructed after full action period

**Aircraft commands:**
```python
bs.traf.ap.selhdgcmd(agent_id, new_heading)  # Heading change
bs.traf.ap.selspd(agent_id, new_speed_kt)    # Speed change
```

### Observation Builder

**Relative position calculation:**
```python
# Convert lat/lon to local Cartesian (km)
x_i = (lon_i - center_lon) * 111.32 * cos(radians(center_lat))
y_i = (lat_i - center_lat) * 111.32

# Relative to agent
x_r[j] = x_j - x_i
y_r[j] = y_j - y_i
```

**Relative velocity calculation:**
```python
# Convert heading/speed to velocity components
vx_i = speed_i * sin(radians(heading_i))
vy_i = speed_i * cos(radians(heading_i))

# Relative velocity
vx_r[j] = vx_j - vx_i
vy_r[j] = vy_j - vy_i
```

**Neighbor ranking:**
```python
distances = [haversine(agent_i, agent_j) for j in all_agents if j != i]
nearest_indices = argsort(distances)[:neighbor_topk]
```

### Team Potential Function

**Sigmoid safety potential:**
```python
def pairwise_potential(sep_nm, threshold=5.0, sensitivity=5.0):
    """Smooth transition from 0 (unsafe) to 1 (safe)"""
    return 1.0 / (1.0 + exp(-(sep_nm - threshold) / sensitivity))

# Sum over all pairs
Φ(state) = Σ_{i<j} pairwise_potential(separation_ij)
```

**PBRS reward calculation:**
```python
ΔΦ = Φ(next_state) - Φ(current_state)
team_reward = team_weight * team_gamma * ΔΦ

# Distribute to agents
if mode == "responsibility":
    for agent in agents:
        agent_reward += team_reward  # Full amount
elif mode == "shared":
    for agent in agents:
        agent_reward += team_reward / len(agents)  # Split equally
```

### Hallucination Detection Integration

When `enable_hallucination_detection=True`:

1. **Ground truth calculation** (every timestep):
   ```python
   for agent_i in agents:
       for agent_j in agents (j > i):
           tcpa, dcpa = compute_tcpa_dcpa(agent_i, agent_j)
           if 0 <= tcpa <= 120 and dcpa < 5.0:
               gt_conflict[i] = 1
               gt_conflict[j] = 1
   ```

2. **Policy prediction** (every action):
   ```python
   if abs(action_hdg) > 3.0 or abs(action_spd) > 5.0:
       if not_toward_waypoint(action) and threat_nearby():
           predicted_alert = 1
   ```

3. **Confusion matrix** (per timestep):
   ```python
   tp = (gt_conflict == 1) and (predicted_alert == 1)
   fp = (gt_conflict == 0) and (predicted_alert == 1)
   fn = (gt_conflict == 1) and (predicted_alert == 0)
   tn = (gt_conflict == 0) and (predicted_alert == 0)
   ```

See [src/analysis/README.md](../analysis/README.md) for detailed detection methodology.

---

## 🐛 Debugging

### Common Issues

**1. BlueSky "already initialized" error**
```python
# Solution: Restart Python kernel or use separate processes
# Environment handles initialization automatically
```

**2. Observation space mismatch during testing**
```python
# Cause: Test environment config differs from training
# Solution: Ensure exact parameter match:
env_config = {
    "neighbor_topk": 3,        # Must match training
    "collision_nm": 3.0,       # Must match training
    # ... other parameters
}
```

**3. Agents not moving**
```python
# Check action scaling:
print(f"Action: {action}")  # Should be [-1, 1]
print(f"Scaled: {action[0]*18} deg, {action[1]*10} kt")

# Verify BlueSky command execution:
print(bs.traf.id)  # List of active aircraft
print(bs.traf.hdg)  # Current headings
```

**4. Reward explosion/collapse**
```python
# Check reward components:
for agent_id in env.agents:
    print(f"{agent_id}: {env.reward_breakdown[agent_id]}")

# Typical ranges:
# - Progress: ±0.5
# - Violations: -25 to -50
# - Team: ±0.1
# If outside these, check config parameters
```

### Logging and Visualization

**Enable detailed logging:**
```python
import logging
logging.getLogger("marl_collision_env").setLevel(logging.DEBUG)
```

**Visualize episode trajectory:**
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("trajectories/traj_ep_0001.csv")

for agent_id in df['agent_id'].unique():
    agent_df = df[df['agent_id'] == agent_id]
    plt.plot(agent_df['lon_deg'], agent_df['lat_deg'], 
             label=agent_id, marker='o')

plt.xlabel("Longitude (deg)")
plt.ylabel("Latitude (deg)")
plt.legend()
plt.title("Episode Trajectory")
plt.grid(True)
plt.show()
```

---

## 📚 References

- **BlueSky Documentation**: https://github.com/TUDelft-CNS-ATM/bluesky
- **PettingZoo API**: https://pettingzoo.farama.org/
- **Ray RLlib**: https://docs.ray.io/en/latest/rllib/index.html
- **PBRS Theory**: Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations

---

**Module Files:**
- `marl_collision_env_minimal.py`: Main environment implementation (1429 lines)
- `marl_collision_env_generic.py`: Dynamic conflict generation variant

**Related Modules:**
- [src/scenarios/](../scenarios/README.md): Scenario generation
- [src/analysis/](../analysis/README.md): Hallucination detection
- [src/training/](../training/README.md): Training workflows
