# Training Hyperparameters Reference

Complete hyperparameter configurations for all supported algorithms extracted from `src/training/train_frozen_scenario.py`.

## PPO (Proximal Policy Optimization)

### Model Architecture
| Parameter | Value | Description |
|-----------|-------|-------------|
| `fcnet_hiddens` | [256, 256] | Hidden layer sizes |
| `fcnet_activation` | "tanh" | Activation function |
| `free_log_std` | False | Fixed vs. learnable log std |

### Learning Configuration
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `lr` | 5e-4 | 3e-4 | Learning rate |
| `gamma` | 0.995 | 0.995 | Discount factor |
| `num_epochs` | 10 | 8 | SGD epochs per training iteration |
| `train_batch_size` | 8192 | 4096 | Training batch size |
| `grad_clip` | 1.0 | 1.0 | Gradient clipping norm |
| `num_sgd_iter` | 4 | 4 | SGD iterations per epoch |

### Rollout Configuration
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `num_env_runners` | 4 | 1 | Parallel environment workers |
| `rollout_fragment_length` | 200 | 200 | Steps per rollout fragment |
| `batch_mode` | "truncate_episodes" | "truncate_episodes" | Batch composition mode |
| `num_cpus_per_env_runner` | 2 | 2 | CPU allocation per worker |

### Policy Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `kl_coeff` | 0.2 | KL divergence coefficient |
| `entropy_coeff` | 0.01 | Entropy coefficient for exploration |
| `vf_loss_coeff` | 1.0 | Value function loss coefficient |
| `clip_param` | 0.1 | PPO clipping parameter |

### Evaluation
| Parameter | Value | Description |
|-----------|-------|-------------|
| `evaluation_interval` | 5 | Evaluation frequency (iterations) |
| `evaluation_duration` | 5 | Episodes per evaluation |
| `evaluation_duration_unit` | "episodes" | Evaluation unit |

## SAC (Soft Actor-Critic)

### Model Architecture
| Parameter | Value | Description |
|-----------|-------|-------------|
| `fcnet_hiddens` | [256, 256] | Hidden layer sizes |
| `fcnet_activation` | "tanh" | Activation function |
| `free_log_std` | False | Fixed vs. learnable log std |

### Learning Configuration
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `lr` | 5e-4 | 3e-4 | Actor learning rate |
| `critic_lr` | 5e-4 | 3e-4 | Critic learning rate |
| `alpha_lr` | 5e-4 | 3e-4 | Temperature parameter learning rate |
| `gamma` | 0.995 | 0.995 | Discount factor |
| `tau` | 0.01 | 0.005 | Target network soft update rate |
| `grad_clip` | 5.0 | 5.0 | Gradient clipping norm |

### Exploration Configuration
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `target_entropy` | "auto" | "auto" | Target entropy for temperature |
| `initial_alpha` | 0.1 | 0.1 | Initial temperature parameter |
| `exploration_config` | StochasticSampling | StochasticSampling | Exploration strategy |

### Replay Buffer
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `buffer_size` | 1,000,000 | 500,000 | Replay buffer capacity |
| `num_steps_sampled_before_learning_starts` | 2,000 | 5,000 | Warmup steps |
| `training_intensity` | 1.5 | 1.0 | Training steps per env step |

### Rollout Configuration
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `num_env_runners` | 4 | 1 | Parallel environment workers |
| `rollout_fragment_length` | 200 | 200 | Steps per rollout fragment |
| `num_cpus_per_env_runner` | 2 | 2 | CPU allocation per worker |

## IMPALA (Importance Weighted Actor-Learner Architecture)

### Model Architecture
| Parameter | Value | Description |
|-----------|-------|-------------|
| `fcnet_hiddens` | [256, 256] | Hidden layer sizes |
| `fcnet_activation` | "tanh" | Activation function |
| `free_log_std` | False | Fixed vs. learnable log std |

### Learning Configuration
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `lr` | 6e-4 | 3e-4 | Learning rate |
| `gamma` | 0.995 | 0.995 | Discount factor |
| `train_batch_size` | 8192 | 4096 | Training batch size |
| `grad_clip` | 5.0 | 5.0 | Gradient clipping norm |

### V-trace Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `vtrace` | True | Enable V-trace correction |
| `vtrace_clip_rho_threshold` | 1.0 | Importance sampling clipping |
| `vtrace_clip_pg_rho_threshold` | 1.0 | Policy gradient clipping |

### Training Configuration
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `entropy_coeff` | 0.01 | 0.01 | Entropy coefficient |
| `vf_loss_coeff` | 0.5 | 0.5 | Value function loss coefficient |
| `minibatch_size` | 512 | 256 | Minibatch size for training |
| `num_epochs` | 1 | 1 | Training epochs per iteration |
| `shuffle_buffer_size` | 1024 | 512 | Experience shuffling buffer |

### Rollout Configuration
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `num_env_runners` | 2 | 1 | Parallel environment workers |
| `rollout_fragment_length` | 50 | 50 | Steps per rollout fragment |

## CQL (Conservative Q-Learning)

### Model Architecture
| Parameter | Value | Description |
|-----------|-------|-------------|
| `fcnet_hiddens` | [256, 256] | Hidden layer sizes |
| `fcnet_activation` | "tanh" | Activation function |
| `free_log_std` | False | Fixed vs. learnable log std |

### Learning Configuration
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `lr` | 5e-4 | 3e-4 | Actor learning rate |
| `critic_lr` | 6e-4 | 3e-4 | Critic learning rate |
| `gamma` | 0.995 | 0.995 | Discount factor |
| `tau` | 0.01 | 0.005 | Target network soft update rate |
| `grad_clip` | 5.0 | 5.0 | Gradient clipping norm |

### CQL-Specific Configuration
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `min_q_weight` | 4.0 | 5.0 | Conservative Q-learning weight |
| `n_step` | 1 | 1 | N-step TD backup |
| `target_entropy` | "auto" | "auto" | Target entropy |
| `initial_alpha` | 0.1 | 0.1 | Initial temperature |

### Replay Buffer
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `buffer_size` | 1,000,000 | 500,000 | Replay buffer capacity |
| `num_steps_sampled_before_learning_starts` | 1,000 | 5,000 | Warmup steps |
| `training_intensity` | 1.5 | 1.0 | Training steps per env step |

### Rollout Configuration
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `num_env_runners` | 1 | 0 | Parallel environment workers |
| `rollout_fragment_length` | 200 | 200 | Steps per rollout fragment |

## APPO (Asynchronous Proximal Policy Optimization)

### Model Architecture
| Parameter | Value | Description |
|-----------|-------|-------------|
| `fcnet_hiddens` | [128, 128] | Hidden layer sizes (smaller for stability) |
| `fcnet_activation` | "relu" | Activation function |
| `vf_share_layers` | False | Separate value function network |
| `use_lstm` | False | LSTM usage |
| `free_log_std` | False | Fixed vs. learnable log std |

### Learning Configuration
| Parameter | GPU Value | CPU Value | Description |
|-----------|-----------|-----------|-------------|
| `lr` | 3e-4 | 3e-4 | Learning rate |
| `gamma` | 0.995 | 0.995 | Discount factor |
| `train_batch_size` | 4096 | 2048 | Training batch size |
| `grad_clip` | 5.0 | 5.0 | Gradient clipping norm |

### Policy Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `entropy_coeff` | 0.001 | Entropy coefficient (lower for stability) |

### Rollout Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_env_runners` | 1 | Conservative worker count for BlueSky |
| `rollout_fragment_length` | 100 | Shorter fragments to reduce staleness |
| `batch_mode` | "truncate_episodes" | Batch composition mode |

## Multi-Agent Configuration (All Algorithms)

### Policy Setup
| Parameter | Value | Description |
|-----------|-------|-------------|
| Policy ID | "shared_policy" | Single shared policy for all agents |
| Policy Mapping | All agents → "shared_policy" | Parameter sharing configuration |
| Observation Space | Dict with relative features | Discovered via temporary environment |
| Action Space | Box([-1, 1], shape=(2,)) | Normalized heading/speed actions |

### Environment Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| `neighbor_topk` | 3 | Number of neighbor observations |
| `max_episode_steps` | 100 | Episode termination limit |
| `separation_nm` | 5.0 | Conflict detection threshold |
| `collision_nm` | 3.0 | Physical collision threshold |
| `log_trajectories` | Configurable | Detailed trajectory logging |

### Early Stopping (Band-Based)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `BAND_STEPS` | 8,200 | Steps per evaluation band |
| `MIN_BANDS_BEFORE_ESTOP` | 3 | Minimum bands before early stop |
| `GOOD_BANDS_TO_STOP` | 2 | Consecutive good bands needed |
| `BAND_ZCS_TARGET` | 20 | Zero-conflict episodes per band |

## Resource Configuration

### GPU Settings
| Parameter | GPU Available | No GPU | Description |
|-----------|---------------|--------|-------------|
| `num_gpus` | 1 | 0 | GPUs allocated to learner |
| `num_gpus_per_env_runner` | 0 | 0 | GPUs per environment worker |
| Ray initialization | Standard | local_mode=True | Ray execution mode |

### CPU Settings
| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_cpus_per_env_runner` | 2 | CPU cores per environment worker |
| `num_cpus_per_learner` | 1 | CPU cores for learner process |

## Training Outputs

### Directory Structure
```
training/results_ALGO_SCENARIO_TIMESTAMP/
├── training_progress.csv           # Timestep, reward, conflict metrics
├── checkpoints/
│   ├── best_50k_r12.3/            # Best reward checkpoints
│   └── band_3_24600/              # Band-aligned checkpoints
└── traj_ep_*.csv                  # Episode trajectories (if enabled)
```

### Progress CSV Columns
| Column | Description |
|--------|-------------|
| `ts` | Unix timestamp |
| `iter` | Training iteration |
| `steps_sampled` | Total environment steps |
| `episodes` | Total episodes completed |
| `reward_mean` | Mean episode reward |
| `zero_conflict_streak` | Consecutive conflict-free episodes |

### Final Model Location
```
models/ALGO_SCENARIO_TIMESTAMP/
├── algorithm_state.pkl             # Algorithm state
├── policies/                       # Policy checkpoints
└── training_metadata.json         # Training configuration and results
```