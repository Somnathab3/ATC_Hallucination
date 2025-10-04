# Multi-Agent Reinforcement Learning Training

This module provides comprehensive training capabilities for **MARL collision avoidance policies** using multiple state-of-the-art algorithms with optimized hyperparameters and early stopping mechanisms.

---

## 📋 Overview

### Key Features

- **Multi-Algorithm Support**: PPO, SAC, IMPALA, CQL, APPO with algorithm-specific tuning
- **Shared Policy Architecture**: Parameter sharing across agents for efficient learning
- **Adaptive Early Stopping**: Band-based performance evaluation prevents premature termination
- **GPU Acceleration**: Automatic CUDA detection and optimized configurations
- **Progress Tracking**: Comprehensive logging with zero-conflict streak monitoring
- **Checkpoint Management**: Best-model saving based on reward improvements

### Training Philosophy

Training uses a **unified reward system** (see [src/environment/README.md](../environment/README.md)) with:
- Signed progress rewards (±0.04/km)
- Well-clear violation penalties (-25 entry, depth-scaled continuation)
- Drift improvement shaping (+0.01/degree)
- Team-based PBRS (5 NM sensitivity, weight=0.6)

**Termination criteria:**
- Zero-conflict streak ≥ 20 evaluations (early success)
- Maximum timesteps reached
- Manual interrupt (Ctrl+C)

---

## 🏗️ Architecture

### Training Pipeline

```
Scenario JSON → Environment Registration → Algorithm Config → Ray Training Loop
                                                                      ↓
                                                            Checkpoint Saving
                                                                      ↓
                                                            Best Model Export
```

### Shared Policy Design

All agents share a **single policy network** for:
- **Parameter efficiency**: N agents use 1 network instead of N networks
- **Generalization**: Policy learns role-agnostic conflict resolution
- **Coordination emergence**: Agents learn cooperative behavior implicitly

**Implementation:**
```python
policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "shared_policy"
```

---

## 🤖 Supported Algorithms

### 1. Proximal Policy Optimization (PPO)

**When to use:** Default choice for most scenarios, stable and sample-efficient

**Hyperparameters:**

| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| **Learning Rate** | 5e-4 | 3e-4 | Higher LR on GPU for faster training |
| **Model** | [256, 256] | [256, 256] | Hidden layer sizes (tanh activation) |
| **Batch Size** | 8192 | 4096 | Training batch size |
| **Minibatch Size** | 512 | 256 | SGD minibatch size |
| **SGD Iterations** | 10 | 8 | Epochs per training batch |
| **Rollout Length** | 200 | 200 | Fragment length for sampling |
| **Gamma** | 0.995 | 0.995 | Discount factor (long horizon) |
| **Lambda (GAE)** | 0.95 | 0.95 | Advantage estimation smoothing |
| **Clip Param** | 0.1 | 0.1 | PPO policy clipping threshold |
| **Grad Clip** | 1.0 | 1.0 | Gradient norm clipping |
| **Entropy Coeff** | 0.01 | 0.01 | Exploration bonus |
| **KL Coeff** | 0.2 | 0.2 | KL divergence penalty |
| **Workers** | 4 | 4 | Parallel rollout workers |

**Training command:**
```bash
python atc_cli.py train --scenario head_on --algo PPO --timesteps 100000 --gpu
```

**Typical convergence:** 50k-100k timesteps for 2-agent scenarios, 150k-250k for 4-agent

---

### 2. Soft Actor-Critic (SAC)

**When to use:** Off-policy learning, sample efficiency, exploration-heavy scenarios

**Hyperparameters:**

| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| **Actor LR** | 5e-4 | 3e-4 | Policy network learning rate |
| **Critic LR** | 5e-4 | 3e-4 | Q-function learning rate |
| **Alpha LR** | 5e-4 | 3e-4 | Temperature parameter LR |
| **Model** | [256, 256] | [256, 256] | Hidden layers (ReLU activation) |
| **Batch Size** | 8192 | 4096 | Training batch size |
| **Replay Buffer** | 1M | 500k | Experience replay capacity (timesteps) |
| **Gamma** | 0.995 | 0.995 | Discount factor |
| **Tau** | 0.01 | 0.01 | Target network update rate |
| **Initial Alpha** | 0.1 | 0.1 | Temperature (entropy weight) |
| **Target Entropy** | auto | auto | Automatic entropy tuning |
| **Training Intensity** | 1.5 | 1.0 | Gradient updates per env step |
| **Warmup Steps** | 2000 | 5000 | Random exploration before learning |

**Training command:**
```bash
python atc_cli.py train --scenario converging --algo SAC --timesteps 150000 --gpu
```

**Typical convergence:** 100k-200k timesteps (better for complex scenarios)

---

### 3. IMPALA (Importance Weighted Actor-Learner Architecture)

**When to use:** Massive parallelism, distributed training, high throughput

**Hyperparameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Learning Rate** | 6e-4 | Slightly higher for off-policy |
| **Model** | [256, 256] | Hidden layers (tanh activation) |
| **Minibatch Size** | 512 | Batch processing size |
| **V-trace Clip Rho** | 1.0 | Off-policy correction clipping |
| **V-trace Clip PG** | 1.0 | Policy gradient clipping |
| **Entropy Coeff** | 0.01 | Exploration bonus |
| **VF Loss Coeff** | 0.5 | Value function loss weight |
| **Gamma** | 0.995 | Discount factor |

**Training command:**
```bash
python atc_cli.py train --scenario parallel --algo IMPALA --timesteps 200000 --gpu
```

**Typical convergence:** 150k-300k timesteps (scales well with workers)

---

### 4. Conservative Q-Learning (CQL)

**When to use:** Offline RL, safety-critical applications, conservative policies

**Training command:**
```bash
python atc_cli.py train --scenario canonical_crossing --algo CQL --timesteps 300000 --gpu
```

**Note:** Slower convergence but more conservative conflict resolution

---

### 5. Asynchronous PPO (APPO)

**When to use:** Large-scale distributed training, faster wall-clock time

**Training command:**
```bash
python atc_cli.py train --scenario t_formation --algo APPO --timesteps 250000 --gpu
```

**Note:** Combines PPO stability with IMPALA throughput

---

## 🎓 Training Workflow

### 1. Scenario Selection

Choose scenario based on training objectives:

```bash
# Foundation: 2-agent head-on conflict
python atc_cli.py train --scenario head_on --algo PPO --timesteps 50000 --gpu

# Intermediate: 3-agent coordination
python atc_cli.py train --scenario parallel --algo PPO --timesteps 100000 --gpu

# Advanced: 4-agent symmetric
python atc_cli.py train --scenario canonical_crossing --algo PPO --timesteps 200000 --gpu

# Expert: 4-agent complex
python atc_cli.py train --scenario converging --algo SAC --timesteps 300000 --gpu
```

### 2. Training Execution

**CLI interface (recommended):**
```bash
python atc_cli.py train \
  --scenario head_on \
  --algo PPO \
  --timesteps 100000 \
  --checkpoint-every 20000 \
  --gpu
```

**Programmatic interface:**
```python
from src.training.train_frozen_scenario import train_frozen

checkpoint_path = train_frozen(
    repo_root="/path/to/ATC_Hallucination",
    algo="PPO",
    seed=42,
    scenario_name="head_on",
    timesteps_total=100000,
    checkpoint_every=20000,
    use_gpu=True,
    log_trajectories=False  # Disable for faster training
)

print(f"Training complete: {checkpoint_path}")
```

### 3. Progress Monitoring

**Training outputs directory structure:**
```
training/results_PPO_head_on_20250928_143052/
├── training_progress.csv         # Timestep, reward, conflict metrics
├── checkpoints/
│   ├── best_reward_12500/        # Best model so far
│   ├── best_reward_37500/        # Improved model
│   └── checkpoint_100000/        # Final checkpoint
├── config.json                   # Full training configuration
└── training_log.txt              # Detailed event log
```

**Key metrics in `training_progress.csv`:**

| Column | Description |
|--------|-------------|
| `timesteps_total` | Cumulative environment steps |
| `episode_reward_mean` | Average episode return |
| `episode_reward_min` | Worst episode return |
| `episode_reward_max` | Best episode return |
| `episode_len_mean` | Average episode length (steps) |
| `zero_conflict_streak` | Consecutive conflict-free evaluations |
| `num_env_steps_sampled` | Total environment interactions |
| `num_agent_steps_sampled` | Total agent interactions (×num_agents) |

**Monitoring during training:**
```bash
# Watch progress (Unix/Linux)
tail -f training/results_PPO_head_on_*/training_progress.csv

# Windows PowerShell
Get-Content training_progress.csv -Wait -Tail 10
```

### 4. Early Stopping

Training automatically terminates when:

**Success criteria:**
- `zero_conflict_streak >= 20` (20 consecutive conflict-free evaluations)
- Implies stable conflict avoidance has been learned

**Or timeout:**
- `timesteps_total >= timesteps_total` (maximum budget exhausted)

**Manual interrupt:**
- Ctrl+C saves current checkpoint before exiting

---

## 📊 Checkpointing Strategy

### Automatic Checkpoint Saving

**Best reward checkpoints:**
- Saved whenever `episode_reward_mean` improves
- Directory: `checkpoints/best_reward_{timesteps}/`
- Includes full algorithm state (policy, optimizer, replay buffer)

**Periodic checkpoints:**
- Saved every `checkpoint_every` timesteps (default: 20k)
- Directory: `checkpoints/checkpoint_{timesteps}/`

**Final model export:**
- Copied to `models/PPO_{scenario}_{timestamp}/`
- Includes training metadata JSON

### Checkpoint Structure

```
models/PPO_head_on_20250928_143052/
├── algorithm_state.pkl           # RLlib Algorithm object
├── policies/
│   └── shared_policy/
│       ├── policy_state.pkl      # Network weights
│       └── rllib_checkpoint.json # Policy configuration
├── metadata.json                 # Training info
└── README.txt                    # Human-readable summary
```

### Loading Checkpoints

```python
from ray.rllib.algorithms.ppo import PPO

# Load trained model
algo = PPO.from_checkpoint("models/PPO_head_on_20250928_143052")

# Get policy for inference
policy = algo.get_policy("shared_policy")

# Compute actions
actions = {
    agent_id: algo.compute_single_action(obs, policy_id="shared_policy")
    for agent_id, obs in observations.items()
}
```

---

## 🔧 Configuration Parameters

### Core Training Config

```python
{
    # Scenario
    "scenario_name": "head_on",          # Scenario identifier
    "seed": 42,                          # Random seed
    
    # Algorithm
    "algo": "PPO",                       # Algorithm choice
    "timesteps_total": 100000,           # Training budget
    "checkpoint_every": 20000,           # Checkpoint frequency
    
    # Resources
    "use_gpu": True,                     # GPU acceleration
    "num_rollout_workers": 4,            # Parallel sampling workers
    
    # Environment
    "neighbor_topk": 3,                  # Observation neighbor count
    "collision_nm": 3.0,                 # Well-clear threshold
    "team_coordination_weight": 0.6,     # Team PBRS weight (0-1)
    
    # Evaluation
    "evaluation_interval": 5,            # Eval every N iterations
    "evaluation_duration": 5,            # Episodes per evaluation
    
    # Logging
    "log_trajectories": False,           # Disable for training speed
    "enable_hallucination_detection": False  # Testing feature only
}
```

### Algorithm-Specific Overrides

**PPO with custom hyperparameters:**
```python
from src.training.train_frozen_scenario import train_frozen

checkpoint = train_frozen(
    repo_root="/path/to/project",
    algo="PPO",
    scenario_name="head_on",
    timesteps_total=100000,
    use_gpu=True,
    
    # PPO-specific overrides
    lr=3e-4,                    # Lower learning rate
    train_batch_size=16384,     # Larger batches
    sgd_minibatch_size=1024,    # Larger minibatches
    num_sgd_iter=15,            # More SGD epochs
    gamma=0.99,                 # Shorter horizon
    entropy_coeff=0.02          # More exploration
)
```

**SAC with custom replay buffer:**
```python
checkpoint = train_frozen(
    repo_root="/path/to/project",
    algo="SAC",
    scenario_name="converging",
    timesteps_total=200000,
    use_gpu=True,
    
    # SAC-specific overrides
    replay_buffer_capacity=2_000_000,  # Larger replay buffer
    training_intensity=2.0,             # More gradient updates
    initial_alpha=0.2,                  # Higher temperature
    n_step=3                            # Multi-step returns
)
```

---

## 📈 Performance Expectations

### Training Time Estimates

**Hardware:** RTX 3080 (10GB), AMD Ryzen 9 5900X

| Scenario | Agents | Timesteps | Wall-Clock Time | GPU Util |
|----------|--------|-----------|-----------------|----------|
| head_on | 2 | 50k | ~15 min | 60-80% |
| parallel | 3 | 100k | ~30 min | 70-85% |
| t_formation | 3 | 100k | ~35 min | 70-85% |
| canonical_crossing | 4 | 200k | ~55 min | 75-90% |
| converging | 4 | 300k | ~90 min | 75-90% |

**Scaling:**
- CPU-only: 3-5× slower
- More workers: ~20% faster with 8 workers vs 4
- Larger batches: ~10% faster but may hurt sample efficiency

### Convergence Indicators

**Healthy training:**
- Episode reward increasing over time
- Zero-conflict streak growing
- Episode length decreasing (faster waypoint completion)

**Problem signs:**
- Reward plateau at negative values → insufficient exploration
- Oscillating rewards → learning rate too high
- Zero-conflict streak stuck at 0 → need more timesteps or easier scenario

**Typical reward trajectories:**
```
Timesteps: 0      25k     50k     75k     100k
Reward:    -150 → -50 → 0 → +50 → +100
Streak:    0    →  2 → 8 → 12 → 20 (SUCCESS)
```

---

## 🚀 Advanced Usage

### Multi-Scenario Training

Train on all scenarios sequentially:

```bash
python atc_cli.py train --scenario all --algo PPO --timesteps 100000 --gpu
```

**Output:** 5 trained models in `models/PPO_{scenario}_*/`

### Algorithm Comparison

Train same scenario with multiple algorithms:

```bash
for algo in PPO SAC IMPALA; do
    python atc_cli.py train --scenario converging --algo $algo --timesteps 200000 --gpu
done
```

**Analysis:** Compare final rewards and conflict rates

### Hyperparameter Sweeps

Programmatic sweep over learning rates:

```python
from src.training.train_frozen_scenario import train_frozen

learning_rates = [1e-4, 3e-4, 5e-4, 1e-3]
results = []

for lr in learning_rates:
    print(f"Training with lr={lr}")
    checkpoint = train_frozen(
        repo_root="/path/to/project",
        algo="PPO",
        scenario_name="head_on",
        timesteps_total=50000,
        use_gpu=True,
        lr=lr,
        seed=42  # Keep seed constant for fair comparison
    )
    
    # Evaluate final model
    # ... evaluation code ...
    results.append({"lr": lr, "reward": final_reward})

# Find best hyperparameter
best = max(results, key=lambda x: x['reward'])
print(f"Best learning rate: {best['lr']}")
```

### Resume Training

Continue from checkpoint:

```python
from ray.rllib.algorithms.ppo import PPO

# Load checkpoint
algo = PPO.from_checkpoint("models/PPO_head_on_20250928/")

# Continue training
for i in range(100):  # 100 more iterations
    result = algo.train()
    print(f"Iteration {i}: reward={result['episode_reward_mean']:.2f}")

# Save updated model
algo.save("models/PPO_head_on_continued")
```

---

## 🐛 Troubleshooting

### Common Issues

**1. Ray initialization errors**
```bash
# Clear Ray state
rm -rf /tmp/ray/*

# Or disable Ray for debugging
python atc_cli.py train --scenario head_on --num-workers 0
```

**2. GPU out of memory**
```python
# Reduce batch sizes
train_batch_size=2048  # Instead of 8192
sgd_minibatch_size=256  # Instead of 512

# Or reduce workers
num_rollout_workers=2  # Instead of 4
```

**3. Training hangs**
```bash
# Check Ray dashboard
# http://localhost:8265

# Or enable detailed logging
export RAY_LOG_TO_STDERR=1
python atc_cli.py train ...
```

**4. Reward not improving**
```python
# Increase exploration
entropy_coeff=0.02  # Default: 0.01

# Reduce learning rate
lr=3e-4  # Default: 5e-4

# Increase training budget
timesteps_total=200000  # Instead of 100000
```

**5. Checkpoint restoration fails**
```python
# Ensure exact algorithm match
algo = PPO.from_checkpoint("path/to/checkpoint")  # Not SAC

# Verify policy ID
policy = algo.get_policy("shared_policy")  # Not "default_policy"
```

### Performance Optimization

**Faster training:**
1. Disable trajectory logging: `log_trajectories=False`
2. Disable hallucination detection: `enable_hallucination_detection=False`
3. Increase workers: `num_rollout_workers=8`
4. Use GPU: `use_gpu=True`
5. Larger batches (if GPU memory allows): `train_batch_size=16384`

**Better sample efficiency:**
1. SAC instead of PPO for complex scenarios
2. Increase gamma: `gamma=0.999` (longer horizon)
3. GAE lambda: `lambda_=0.98` (smoother advantages)
4. Reduce exploration: `entropy_coeff=0.005`

---

## 📚 References

- **PPO**: Schulman et al. (2017), Proximal Policy Optimization Algorithms
- **SAC**: Haarnoja et al. (2018), Soft Actor-Critic Algorithms
- **IMPALA**: Espeholt et al. (2018), IMPALA: Scalable Distributed Deep-RL
- **RLlib Documentation**: https://docs.ray.io/en/latest/rllib/index.html

---

**Module Files:**
- `train_frozen_scenario.py`: Main training script (840 lines)
- `train_generic.py`: Generic environment trainer (wrapper)

**Related Modules:**
- [src/environment/](../environment/README.md): Environment implementation
- [src/scenarios/](../scenarios/README.md): Scenario generation
- [src/testing/](../testing/README.md): Model evaluation

---

**Output Directories:**
- `training/results_{algo}_{scenario}_{timestamp}/`: Training artifacts
- `models/{algo}_{scenario}_{timestamp}/`: Final trained models
