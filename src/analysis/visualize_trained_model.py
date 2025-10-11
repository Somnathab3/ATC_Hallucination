#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Multi-Agent Air Traffic Control Visualization (Training Standards Compatible)

This script loads a trained RLlib model and visualizes it with pygame rendering.
Updated to match training CLI standards with scenario type and checkpoint validation.

Usage:
    python visualize_trained_model.py --scenario merge_2x2 --checkpoint F:\ATC_Hallucination\models\PPO_merge_2x2_20251008_170616
    python visualize_trained_model.py --scenario head_on --checkpoint path/to/checkpoint --episodes 10
    python visualize_trained_model.py --scenario cross_2x2 --checkpoint path/to/checkpoint --no-record-gifs
"""

import os
import sys

# Fix Windows console encoding for emoji/Unicode support
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        # Fallback: continue without UTF-8 encoding
        pass

import json
import argparse
import numpy as np
import pygame
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from math import cos, sin, radians, degrees, sqrt, atan2
from datetime import datetime

# Try to import PIL for GIF creation
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL (Pillow) not available. GIF recording disabled. Install with: pip install Pillow")

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import exact distance calculation functions from training environment
try:
    from src.environment.marl_collision_env_minimal import haversine_nm, NM_TO_KM
except ImportError:
    # Fallback haversine function if import fails
    def haversine_nm(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
        """Compute great-circle distance in nautical miles (exact copy from training env)."""
        from math import radians, sin, cos, sqrt, atan2
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1_deg, lon1_deg, lat2_deg, lon2_deg])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        R_nm = 3440.065  # Earth radius in nautical miles
        return R_nm * c
    
    NM_TO_KM = 1.852  # Nautical miles to kilometers conversion

# Colors for pygame visualization
COLORS = {
    'background': (15, 25, 35),
    'agent_trained': (100, 255, 150),
    'waypoint': (255, 120, 120),
    'trajectory_trained': (120, 255, 180),
    'intrusion_zone': (255, 80, 80),
    'text': (255, 255, 255),
    'grid': (50, 60, 75),
    'speed_vector': (255, 255, 150),
    'conflict': (255, 50, 50),
    'success': (100, 255, 100),
    'panel_bg': (25, 35, 45),
}

# Enhanced agent colors for better distinction
AGENT_COLORS = [
    (120, 255, 120),  # Bright green
    (120, 150, 255),  # Bright blue  
    (255, 255, 120),  # Bright yellow
    (255, 120, 255),  # Bright magenta
    (120, 255, 255),  # Bright cyan
    (255, 180, 120),  # Bright orange
]

class EpisodeGIFRecorder:
    """Records visualization frames and saves episodes as separate GIFs"""
    
    def __init__(self, output_dir: str = "episode_gifs", enabled: bool = True, custom_name: Optional[str] = None):
        self.enabled = enabled and PIL_AVAILABLE
        self.output_dir = Path(output_dir)
        self.custom_name = custom_name  # Custom GIF name (without .gif)
        self.frames = []
        self.current_episode = 0
        self.recording = False
        
        if self.enabled:
            self.output_dir.mkdir(exist_ok=True)
            print(f"GIF recording enabled. Output directory: {self.output_dir}")

    def start_episode(self, episode_num: int):
        """Start recording a new episode"""
        if not self.enabled:
            return
            
        self.current_episode = episode_num
        self.frames = []
        self.recording = True
        print(f"🎬 Recording Episode {episode_num}")
    
    def capture_frame(self, pygame_surface: pygame.Surface):
        """Capture a frame from pygame surface"""
        if not self.enabled or not self.recording:
            return
            
        frame_array = pygame.surfarray.array3d(pygame_surface)
        frame_array = frame_array.swapaxes(0, 1)[:, :, ::-1]
        frame_image = Image.fromarray(frame_array, 'RGB')
        self.frames.append(frame_image)
    
    def end_episode(self, episode_stats: Optional[Dict[str, Any]] = None):
        """End recording and save GIF for current episode"""
        if not self.enabled or not self.recording or len(self.frames) == 0:
            return
        
        # Use custom name if provided, otherwise use episode numbering
        if self.custom_name:
            filename = f"{self.custom_name}.gif"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stats_str = ""
            if episode_stats:
                steps = episode_stats.get('steps', 0)
                waypoints = episode_stats.get('waypoints_reached', 0)
                conflicts = episode_stats.get('total_conflicts', 0)
                stats_str = f"_S{steps}_WP{waypoints}_C{conflicts}"
            
            filename = f"episode_{self.current_episode:03d}_{timestamp}{stats_str}.gif"
        filepath = self.output_dir / filename
        
        try:
            self.frames[0].save(
                filepath,
                save_all=True,
                append_images=self.frames[1:],
                duration=125,
                loop=0,
                optimize=True
            )
            
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"💾 Saved Episode {self.current_episode} GIF: {filename} ({file_size_mb:.1f} MB, {len(self.frames)} frames)")
            
        except Exception as e:
            print(f"❌ Error saving GIF for Episode {self.current_episode}: {e}")
        
        finally:
            self.frames = []
            self.recording = False

class TrainedModelVisualizer:
    """Visualize trained models with enhanced pygame rendering and training standards compatibility"""
    
    def __init__(self, checkpoint_path: str, render_config: Dict[str, Any]):
        self.checkpoint_path = checkpoint_path
        self.render_config = render_config
        
        # German city names for waypoints
        self.german_cities = {
            'A1': 'Dresden', 'A2': 'Berlin', 'A3': 'Hamburg',
            'A4': 'Munich', 'A5': 'Frankfurt', 'A6': 'Cologne'
        }
        
        # Icon variables (ASCII fallbacks for better compatibility)
        self.icon_episode = "EPISODE"
        self.icon_conflicts = "CONFLICTS"
        self.icon_agents = "AGENTS"
        self.icon_controls = "CONTROLS"
        self.city_marker = "*"  # Simple marker instead of emoji
        
        # Initialize pygame
        pygame.init()
        
        # Screen setup
        info = pygame.display.Info()
        self.width = min(render_config.get('width', 1400), info.current_w - 50)
        self.height = min(render_config.get('height', 1000), info.current_h - 100)
        
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        scenario_name = render_config.get('scenario', 'Unknown')
        algo_name = render_config.get('algo', 'Unknown')
        pygame.display.set_caption(f"ATC Visualization - {scenario_name} ({algo_name})")
        self.clock = pygame.time.Clock()
        
        # Fonts with wide-glyph coverage
        base_font_size = max(20, min(28, self.height // 40))
        self._init_fonts(base_font_size)
        
        # Performance caches
        self._ring_cache = {}  # Cache ring surfaces by (radius, style)
        
        # Visualization state
        self.running = True
        self.paused = False
        self.step_by_step = False
        self.single_step_advance = False
        self.show_trajectories = True
        self.show_intrusion_zones = True
        self.show_speed_vectors = True
        self.show_waypoint_lines = True
        self.show_metrics_panel = True
        self.show_help = False
        self.show_grid = True
        self.show_breadcrumbs = False
        self.show_scale_compass = True
        self.show_pair_matrix = True
        self.show_timeline = True
        self.follow_agent = None  # None or agent name to follow
        
        # Zoom and pan
        self.base_scale = 3.0  # Reduced from 4.0 to make rings appear at correct size
        self.zoom_level = 1.0
        self.min_zoom = 0.2
        self.max_zoom = 8.0
        self.scale = self.base_scale * self.zoom_level
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        
        # Visualization data
        self.agent_trajectories = {}
        self.episode_count = 0
        self.step_count = 0
        
        # Conflict history tracking
        self.conflict_history = []  # List of (timestep, agent1, agent2, separation_nm) tuples
        self.max_conflict_history = 10  # Keep last 10 conflicts
        
        # Timeline data for sparkline
        self.min_sep_timeline = []  # List of (timestep, min_separation) tuples
        self.max_timeline_length = 100  # Keep last 100 steps
        
        # Agent waypoint reached status
        self.agents_reached_waypoint = set()  # Set of agent IDs that reached their waypoint
        
        # GIF recorder
        self.gif_recorder = EpisodeGIFRecorder(
            output_dir=render_config.get('gif_output_dir', 'episode_gifs'),
            enabled=render_config.get('record_gifs', True),
            custom_name=render_config.get('gif_name', None)
        )
        
        # Store shift configuration for environment reset
        self.shift_config = render_config.get('shift_config', None)
        
        # Load model and create environment
        self.algorithm = None
        self.env = None
        self.model_loaded = False
        self._load_model()
    
    def _init_fonts(self, base_size: int):
        """Initialize fonts with wide-glyph coverage"""
        # Try to find a font with good symbol/emoji support
        candidates = [
            "assets/fonts/NotoSansSymbols2-Regular.ttf",
            "assets/fonts/DejaVuSans.ttf",
            "C:/Windows/Fonts/seguisym.ttf",  # Windows
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  # macOS
        ]
        
        font_path = None
        for path in candidates:
            if os.path.exists(path):
                font_path = path
                break
        
        # Initialize fonts (None = default pygame font)
        self.font = pygame.font.Font(font_path, base_size)
        self.small_font = pygame.font.Font(font_path, base_size - 6)
        self.large_font = pygame.font.Font(font_path, base_size + 6)
        self.title_font = pygame.font.Font(font_path, base_size + 12)
    
    def get_city_name(self, agent: str) -> str:
        """Get German city name for agent"""
        return self.german_cities.get(agent, agent.replace('_', ' ').title())
    
    def _load_model(self):
        """Load the trained model from checkpoint"""
        if not self.checkpoint_path:
            print("🤖 No checkpoint provided, using rule-based policy")
            self._create_fallback_environment()
            return
            
        try:
            import ray
            from ray.rllib.algorithms.ppo import PPO
            from ray.rllib.algorithms.sac import SAC
            from ray.rllib.algorithms.impala import IMPALA
            from ray.tune.registry import register_env
            from ray.rllib.env import ParallelPettingZooEnv
            from src.environment.marl_collision_env_minimal import MARLCollisionEnv
            
            print(f"🔄 Loading trained model from: {self.checkpoint_path}")
            
            # Initialize Ray
            if not ray.is_initialized():
                ray.init(local_mode=True, log_to_driver=False, configure_logging=False)
            
            # Register environment 
            register_env("marl_collision_env_v0", 
                        lambda cfg: ParallelPettingZooEnv(MARLCollisionEnv(cfg)))
            
            # Detect algorithm type
            algo_type = self.render_config.get('algo', 'PPO')
            
            # Load the appropriate algorithm
            if algo_type == "SAC":
                self.algorithm = SAC.from_checkpoint(self.checkpoint_path)
            elif algo_type == "IMPALA":
                self.algorithm = IMPALA.from_checkpoint(self.checkpoint_path)
            else:  # Default to PPO
                self.algorithm = PPO.from_checkpoint(self.checkpoint_path)
                
            print(f"✅ {algo_type} model loaded successfully!")
            
            # Create environment
            self._create_environment()
            self.model_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Falling back to rule-based policy for visualization")
            self._create_fallback_environment()
    
    def _create_environment(self):
        """Create environment with trained model"""
        from src.environment.marl_collision_env_minimal import MARLCollisionEnv
        from ray.rllib.env import ParallelPettingZooEnv
        import os
        
        scenario_name = self.render_config.get('scenario', 'canonical_crossing')
        # Use project root to find scenarios directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        scenario_path = os.path.join(project_root, "scenarios", f"{scenario_name}.json")
        
        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
        
        env_config = {
            "scenario_path": scenario_path,
            "action_delay_steps": 0,
            "max_episode_steps": self.render_config.get('max_steps', 200),  # Trained for 150 steps, 200 provides buffer
            "separation_nm": 5.0,
            "log_trajectories": self.render_config.get('log_trajectories', True),  # Enable trajectory logging
            "seed": self.render_config.get('seed', 42),  # Use seed from config
            "results_dir": os.path.abspath(self.render_config.get('results_dir', 'vis_results')),

            # Observation configuration - match training exactly
            "neighbor_topk": 3,
            
            # Collision and conflict settings
            "collision_nm": 3.0,

            # === UNIFIED REWARD SYSTEM (EXACT MATCH WITH TRAINING) ===
            
            # Enhanced team coordination (PBRS) with 5 NM sensitivity
            "team_coordination_weight": 0.6,           # Increased coordination signal strength
            "team_gamma": 0.99,
            "team_share_mode": "responsibility",        # Share rewards based on agent responsibility
            "team_ema": 0.05,                          # Faster team phi response
            "team_cap": 0.05,                          # Higher team reward magnitude cap
            "team_anneal": 1.0,
            "team_neighbor_threshold_km": 10.0,
            
            # Signed progress reward (unified forward/backward movement)
            "progress_reward_per_km": 0.04,            # Positive for progress, negative for backtracking
            
            # Unified well-clear violation system (no double-counting)
            "violation_entry_penalty": -25.0,          # One-time penalty on separation violation
            "violation_step_scale": -1.0,              # Per-step penalty scaled by severity
            "deep_breach_nm": 1.0,                     # Steeper scaling for close approaches
            
            # Drift improvement shaping (rewards heading optimization)
            "drift_improve_gain": 0.01,                # Reward per degree of drift reduction
            "drift_deadzone_deg": 8.0,                 # Deadzone prevents oscillation penalties
            
            # Other individual reward components
            "time_penalty_per_sec": -0.0005,           # Efficiency incentive
            "reach_reward": 10.0,                       # Waypoint achievement bonus
            "action_cost_per_unit": -0.01,             # Cost for non-neutral actions
            "terminal_not_reached_penalty": -10.0,     # Penalty for episode termination without goal
        }
        
        self.env = ParallelPettingZooEnv(MARLCollisionEnv(env_config))
        
        # Store environment parameters for accurate visualization
        # NM_TO_KM already imported at top of file
        self.separation_km = env_config.get("separation_nm", 5.0) * NM_TO_KM
        self.collision_km = env_config.get("collision_nm", 3.0) * NM_TO_KM
        self.deep_breach_km = env_config.get("deep_breach_nm", 1.0) * NM_TO_KM
        
        print("✅ Environment created successfully!")
        print(f"   Separation zone: {env_config.get('separation_nm', 5.0):.1f} NM ({self.separation_km:.2f} km)")
        print(f"   Collision zone: {env_config.get('collision_nm', 3.0):.1f} NM ({self.collision_km:.2f} km)")
        print(f"   📊 Trajectory logging: {'Enabled' if env_config.get('log_trajectories', False) else 'Disabled'}")
        if env_config.get('log_trajectories', False):
            print(f"   📁 Results directory: {env_config.get('results_dir', 'vis_results')}")
    
    def _create_fallback_environment(self):
        """Create environment with fallback rule-based policy"""
        try:
            from src.environment.marl_collision_env_minimal import MARLCollisionEnv
            from ray.rllib.env import ParallelPettingZooEnv
            import os
            
            scenario_name = self.render_config.get('scenario', 'canonical_crossing')
            # Use project root to find scenarios directory
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            scenario_path = os.path.join(project_root, "scenarios", f"{scenario_name}.json")
            
            if not os.path.exists(scenario_path):
                raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
            
            # Use same configuration as main environment for consistency
            env_config = {
                "scenario_path": scenario_path,
                "action_delay_steps": 0,
                "max_episode_steps": self.render_config.get('max_steps', 200),  # Trained for 150 steps, 200 provides buffer
                "separation_nm": 5.0,
                "log_trajectories": self.render_config.get('log_trajectories', True),  # Enable trajectory logging
                "seed": self.render_config.get('seed', 42),  # Use seed from config
                "results_dir": os.path.abspath(self.render_config.get('results_dir', 'vis_results')),
                "neighbor_topk": 3,
                "collision_nm": 3.0,
                # Include essential reward parameters for consistency
                "team_coordination_weight": 0.6,
                "team_gamma": 0.99,
                "team_share_mode": "responsibility",
                "progress_reward_per_km": 0.04,
                "violation_entry_penalty": -25.0,
                "violation_step_scale": -1.0,
                "reach_reward": 10.0,
            }
            
            self.env = ParallelPettingZooEnv(MARLCollisionEnv(env_config))
            
            # Store environment parameters for accurate visualization
            # NM_TO_KM already imported at top of file
            self.separation_km = env_config.get("separation_nm", 5.0) * NM_TO_KM
            self.collision_km = env_config.get("collision_nm", 3.0) * NM_TO_KM
            self.deep_breach_km = env_config.get("deep_breach_nm", 1.0) * NM_TO_KM
            
            print("✅ Fallback environment created successfully!")
            print(f"   Separation zone: {env_config.get('separation_nm', 5.0):.1f} NM ({self.separation_km:.2f} km)")
            print(f"   📊 Trajectory logging: {'Enabled' if env_config.get('log_trajectories', False) else 'Disabled'}")
            if env_config.get('log_trajectories', False):
                print(f"   📁 Results directory: {env_config.get('results_dir', 'vis_results')}")
            
        except Exception as e:
            print(f"❌ Critical error: Could not create environment: {e}")
            raise
    
    def get_action(self, agent: str, observation: np.ndarray) -> np.ndarray:
        """Get action from trained model or fallback policy"""
        if self.model_loaded and self.algorithm:
            try:
                # Use deterministic actions (no exploration) for baseline visualization
                result = self.algorithm.compute_single_action(
                    observation, 
                    policy_id="shared_policy",
                    explore=False  # Disable exploration for deterministic baseline behavior
                )
                
                if isinstance(result, tuple):
                    action = result[0]
                else:
                    action = result
                    
                return np.array(action, dtype=np.float32)
                
            except Exception as e:
                if self.render_config.get('verbose', False):
                    print(f"⚠️ Model inference error for {agent}: {e}")
                return self._waypoint_seeking_policy(agent, observation)
        else:
            return self._waypoint_seeking_policy(agent, observation)
    
    def _waypoint_seeking_policy(self, agent: str, observation) -> np.ndarray:
        """Rule-based policy for fallback - handles Dict observations from training environment"""
        # The training environment uses Dict observations with specific keys
        if isinstance(observation, dict):
            # Extract components from Dict observation (training environment format)
            cos_drift = observation.get('cos_to_wp', np.array([0.0]))[0]
            sin_drift = observation.get('sin_to_wp', np.array([0.0]))[0]
            norm_speed = observation.get('airspeed', np.array([0.0]))[0]
        elif hasattr(observation, '__len__') and len(observation) >= 3:
            # Fallback for array-based observations
            cos_drift = float(observation[0]) if len(observation) > 0 else 0.0
            sin_drift = float(observation[1]) if len(observation) > 1 else 0.0
            norm_speed = float(observation[2]) if len(observation) > 2 else 0.0
        else:
            # Default fallback
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # Calculate desired heading correction
        drift_angle = np.arctan2(sin_drift, cos_drift)
        heading_action = np.clip(drift_angle * 0.6, -1.0, 1.0)
        
        # Speed control
        target_speed = 0.65
        speed_error = target_speed - norm_speed
        speed_action = np.clip(speed_error * 1.8, -1.0, 1.0)
        
        return np.array([heading_action, speed_action], dtype=np.float32)
    
    def world_to_screen(self, x_km: float, y_km: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        adjusted_x = x_km - self.pan_x
        adjusted_y = y_km - self.pan_y
        screen_x = int(self.center_x + adjusted_x * self.scale)
        screen_y = int(self.center_y - adjusted_y * self.scale)
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        world_x = (screen_x - self.center_x) / self.scale + self.pan_x
        world_y = (self.center_y - screen_y) / self.scale + self.pan_y
        return world_x, world_y
    
    def update_zoom(self, zoom_factor: float, center_x: Optional[int] = None, center_y: Optional[int] = None):
        """Update zoom level"""
        # Ensure we have valid integer coordinates
        actual_center_x = center_x if center_x is not None else self.center_x
        actual_center_y = center_y if center_y is not None else self.center_y
        
        world_center_x, world_center_y = self.screen_to_world(actual_center_x, actual_center_y)
        
        old_zoom = self.zoom_level
        self.zoom_level *= zoom_factor
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, self.zoom_level))
        self.scale = self.base_scale * self.zoom_level
        
        if self.zoom_level != old_zoom:
            new_world_center_x, new_world_center_y = self.screen_to_world(actual_center_x, actual_center_y)
            self.pan_x += new_world_center_x - world_center_x
            self.pan_y += new_world_center_y - world_center_y
    
    def reset_view(self):
        """Reset zoom and pan to default"""
        self.zoom_level = 1.0
        self.scale = self.base_scale * self.zoom_level
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.follow_agent = None
    
    def frame_all_aircraft(self, observations: Dict):
        """Auto-zoom and pan to frame all aircraft with padding"""
        if not observations:
            return
        
        # Get all aircraft positions
        positions = []
        for agent in observations.keys():
            x_km, y_km = self.get_agent_position(agent)
            positions.append((x_km, y_km))
        
        if len(positions) < 2:
            self.reset_view()
            return
        
        # Calculate bounding box
        min_x = min(p[0] for p in positions)
        max_x = max(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] for p in positions)
        
        # Add padding (20% on each side)
        padding = 0.2
        range_x = max_x - min_x
        range_y = max_y - min_y
        min_x -= range_x * padding
        max_x += range_x * padding
        min_y -= range_y * padding
        max_y += range_y * padding
        
        # Calculate required zoom to fit all aircraft
        # Account for panel width
        usable_width = self.width - 450  # Leave room for panel
        usable_height = self.height - 100  # Leave room for margins
        
        if range_x > 0 and range_y > 0:
            zoom_x = usable_width / (range_x * self.base_scale)
            zoom_y = usable_height / (range_y * self.base_scale)
            self.zoom_level = min(zoom_x, zoom_y, self.max_zoom)
            self.zoom_level = max(self.zoom_level, self.min_zoom)
            self.scale = self.base_scale * self.zoom_level
        
        # Center on the centroid
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        self.pan_x = center_x
        self.pan_y = center_y
    
    def get_agent_position(self, agent: str) -> Tuple[float, float]:
        """Get agent position in km relative to spawn center"""
        try:
            import bluesky as bs
            acid = agent.upper()
            ac_idx = bs.traf.id2idx(acid)
            if isinstance(ac_idx, list) and len(ac_idx) > 0:
                ac_idx = ac_idx[0]
            
            if isinstance(ac_idx, int) and ac_idx >= 0:
                lat, lon = bs.traf.lat[ac_idx], bs.traf.lon[ac_idx]
                
                # Convert to km relative to scenario center using more accurate method
                scenario_center = self._get_scenario_center()
                spawn_lat = scenario_center.get('lat', 52.0)
                spawn_lon = scenario_center.get('lon', 4.0)
                
                # Use more accurate conversion (matching training environment)
                # 1 degree latitude ≈ 111 km
                # 1 degree longitude ≈ 111 km * cos(latitude)
                y_km = (lat - spawn_lat) * 111.0
                x_km = (lon - spawn_lon) * 111.0 * cos(radians(spawn_lat))  # Use spawn_lat for consistency
                return x_km, y_km
            return 0.0, 0.0
        except Exception as e:
            return 0.0, 0.0
    
    def _get_scenario_center(self) -> Dict[str, float]:
        """Get scenario center from loaded scenario file"""
        try:
            scenario_name = self.render_config.get('scenario', 'canonical_crossing')
            scenario_path = f"scenarios/{scenario_name}.json"
            
            if os.path.exists(scenario_path):
                with open(scenario_path, 'r') as f:
                    scenario_data = json.load(f)
                return scenario_data.get('center', {'lat': 52.0, 'lon': 4.0})
        except:
            pass
        return {'lat': 52.0, 'lon': 4.0}  # Default center
    
    def get_agent_heading_speed(self, agent: str) -> Tuple[float, float]:
        """Get agent heading and speed"""
        try:
            import bluesky as bs
            acid = agent.upper()
            ac_idx = bs.traf.id2idx(acid)
            if isinstance(ac_idx, list) and len(ac_idx) > 0:
                ac_idx = ac_idx[0]
            
            if isinstance(ac_idx, int) and ac_idx >= 0:
                hdg = float(getattr(bs.traf, 'hdg', [0.0])[ac_idx]) if hasattr(bs.traf, 'hdg') else 0.0
                tas = float(getattr(bs.traf, 'tas', [250.0])[ac_idx]) if hasattr(bs.traf, 'tas') else 250.0
                spd = tas * 1.94384  # Convert m/s to kt
                return hdg, spd
            return 0.0, 250.0
        except Exception as e:
            return 0.0, 250.0
    
    def get_waypoint_position(self, agent: str) -> Optional[Tuple[float, float]]:
        """Get waypoint position for agent"""
        try:
            # Access waypoints through the environment hierarchy
            waypoints = None
            if self.env and hasattr(self.env, 'par_env') and hasattr(self.env.par_env, '_agent_waypoints'):
                waypoints = self.env.par_env._agent_waypoints
            
            if waypoints and agent in waypoints:
                wpt_lat, wpt_lon = waypoints[agent]
                scenario_center = self._get_scenario_center()
                spawn_lat = scenario_center.get('lat', 52.0)
                spawn_lon = scenario_center.get('lon', 4.0)
                x_km = (wpt_lon - spawn_lon) * 111.0 * cos(radians(wpt_lat))
                y_km = (wpt_lat - spawn_lat) * 111.0
                return x_km, y_km
        except:
            pass
        return None
    
    def get_agent_color(self, agent: str) -> Tuple[int, int, int]:
        """Get consistent color for each agent"""
        agent_idx = hash(agent) % len(AGENT_COLORS)
        return AGENT_COLORS[agent_idx]
    
    def draw_grid(self):
        """Draw coordinate grid"""
        if not self.show_grid:
            return
        
        base_spacing = 25.0
        if self.zoom_level < 0.5:
            grid_spacing = base_spacing * 2
        elif self.zoom_level > 3.0:
            grid_spacing = base_spacing / 2
        else:
            grid_spacing = base_spacing
        
        pixel_spacing = grid_spacing * self.scale
        
        if pixel_spacing > 10:
            top_left_world = self.screen_to_world(0, 0)
            bottom_right_world = self.screen_to_world(self.width, self.height)
            
            min_x = int(top_left_world[0] // grid_spacing) * grid_spacing - grid_spacing
            max_x = int(bottom_right_world[0] // grid_spacing) * grid_spacing + grid_spacing * 2
            min_y = int(bottom_right_world[1] // grid_spacing) * grid_spacing - grid_spacing
            max_y = int(top_left_world[1] // grid_spacing) * grid_spacing + grid_spacing * 2
            
            # Vertical lines
            x = min_x
            while x <= max_x:
                screen_x, _ = self.world_to_screen(x, 0)
                if -50 <= screen_x <= self.width + 50:
                    pygame.draw.line(self.screen, COLORS['grid'], (screen_x, 0), (screen_x, self.height), 1)
                x += grid_spacing
            
            # Horizontal lines
            y = min_y
            while y <= max_y:
                _, screen_y = self.world_to_screen(0, y)
                if -50 <= screen_y <= self.height + 50:
                    pygame.draw.line(self.screen, COLORS['grid'], (0, screen_y), (self.width, screen_y), 1)
                y += grid_spacing
    
    def draw_scale_compass(self):
        """Draw scale bar (NM) and north arrow for spatial reference"""
        if not self.show_scale_compass:
            return
        
        # Scale bar (bottom-left)
        scale_x = 20
        scale_y = self.height - 60
        
        # Calculate scale bar length (10 NM or appropriate power of 10)
        nm_per_pixel = 1.0 / (self.scale * NM_TO_KM)
        
        # Find nice round number for scale
        scale_nm_options = [1, 2, 5, 10, 20, 50, 100]
        target_pixels = 100  # Aim for ~100px scale bar
        scale_nm = min(scale_nm_options, key=lambda x: abs(x / nm_per_pixel - target_pixels))
        scale_pixels = int(scale_nm / nm_per_pixel)
        
        # Draw scale bar background
        bar_bg = pygame.Surface((scale_pixels + 40, 50), pygame.SRCALPHA)
        bar_bg.fill((25, 35, 45, 220))
        self.screen.blit(bar_bg, (scale_x - 10, scale_y - 10))
        
        # Draw scale bar line
        pygame.draw.line(self.screen, COLORS['text'], (scale_x, scale_y + 15), (scale_x + scale_pixels, scale_y + 15), 3)
        pygame.draw.line(self.screen, COLORS['text'], (scale_x, scale_y + 10), (scale_x, scale_y + 20), 2)
        pygame.draw.line(self.screen, COLORS['text'], (scale_x + scale_pixels, scale_y + 10), (scale_x + scale_pixels, scale_y + 20), 2)
        
        # Draw scale label
        scale_text = f"{scale_nm} NM"
        scale_label = self.small_font.render(scale_text, True, COLORS['text'])
        self.screen.blit(scale_label, (scale_x, scale_y - 5))
        
        # North arrow (bottom-right of scale)
        compass_x = scale_x + scale_pixels + 30
        compass_y = scale_y + 15
        
        # Draw north arrow
        arrow_size = 15
        arrow_points = [
            (compass_x, compass_y - arrow_size),  # Top (north)
            (compass_x - arrow_size // 3, compass_y + arrow_size // 3),  # Left
            (compass_x, compass_y),  # Center
            (compass_x + arrow_size // 3, compass_y + arrow_size // 3),  # Right
        ]
        pygame.draw.polygon(self.screen, (255, 100, 100), arrow_points)
        pygame.draw.polygon(self.screen, COLORS['text'], arrow_points, 2)
        
        # Draw 'N' label
        n_label = self.small_font.render("N", True, COLORS['text'])
        self.screen.blit(n_label, (compass_x - 6, compass_y - arrow_size - 18))
    
    def draw_agent(self, agent: str, x_km: float, y_km: float, heading: float, speed: float):
        """Draw agent aircraft as a plane shape"""
        screen_x, screen_y = self.world_to_screen(x_km, y_km)
        agent_color = self.get_agent_color(agent)
        
        # Aircraft size based on zoom
        size = max(6, int(12 * self.zoom_level))
        hdg_rad = radians(heading - 90)
        
        # Draw plane shape: fuselage + wings
        # Fuselage (main body)
        fuselage_length = size
        fuselage_width = size * 0.3
        
        # Calculate fuselage endpoints
        nose_x = screen_x + fuselage_length * 0.5 * cos(hdg_rad)
        nose_y = screen_y + fuselage_length * 0.5 * sin(hdg_rad)
        tail_x = screen_x - fuselage_length * 0.5 * cos(hdg_rad)
        tail_y = screen_y - fuselage_length * 0.5 * sin(hdg_rad)
        
        # Draw fuselage as a thick line
        pygame.draw.line(self.screen, agent_color, (nose_x, nose_y), (tail_x, tail_y), max(2, int(size * 0.3)))
        
        # Wings (perpendicular to fuselage)
        wing_length = size * 0.8
        wing_x1 = screen_x + wing_length * 0.5 * cos(hdg_rad + radians(90))
        wing_y1 = screen_y + wing_length * 0.5 * sin(hdg_rad + radians(90))
        wing_x2 = screen_x - wing_length * 0.5 * cos(hdg_rad + radians(90))
        wing_y2 = screen_y - wing_length * 0.5 * sin(hdg_rad + radians(90))
        
        pygame.draw.line(self.screen, agent_color, (wing_x1, wing_y1), (wing_x2, wing_y2), max(2, int(size * 0.25)))
        
        # Vertical stabilizer (tail)
        tail_height = size * 0.4
        tail_x1 = tail_x + tail_height * cos(hdg_rad + radians(90))
        tail_y1 = tail_y + tail_height * sin(hdg_rad + radians(90))
        tail_x2 = tail_x - tail_height * cos(hdg_rad + radians(90))
        tail_y2 = tail_y - tail_height * sin(hdg_rad + radians(90))
        
        pygame.draw.line(self.screen, agent_color, (tail_x1, tail_y1), (tail_x2, tail_y2), max(1, int(size * 0.15)))
        
        # Outline the entire aircraft
        pygame.draw.circle(self.screen, COLORS['text'], (int(screen_x), int(screen_y)), max(2, int(size * 0.6)), 1)
        
        # Speed vector
        if self.show_speed_vectors and speed > 0:
            vector_length = min(speed / 6.0, 60)
            end_x = screen_x + vector_length * cos(hdg_rad)
            end_y = screen_y + vector_length * sin(hdg_rad)
            pygame.draw.line(self.screen, COLORS['speed_vector'], 
                           (screen_x, screen_y), (end_x, end_y), 2)
        
        # Labels
        if self.zoom_level > 0.3:
            label_text = f"{agent}"
            if self.zoom_level > 0.6:
                label_text += f" ({speed:.0f}kt)"
            
            text = self.small_font.render(label_text, True, COLORS['text'])
            self.screen.blit(text, (screen_x + 15, screen_y - 15))
    
    def draw_waypoint(self, x_km: float, y_km: float, agent: Optional[str] = None):
        """Draw waypoint with city name"""
        screen_x, screen_y = self.world_to_screen(x_km, y_km)
        color = self.get_agent_color(agent) if agent else COLORS['waypoint']
        
        # Pulsing waypoint
        pulse = int(10 + 3 * sin(time.time() * 3))
        pygame.draw.circle(self.screen, color, (screen_x, screen_y), pulse)
        pygame.draw.circle(self.screen, COLORS['text'], (screen_x, screen_y), pulse, 2)
        pygame.draw.circle(self.screen, COLORS['text'], (screen_x, screen_y), 4)
        
        # City label
        if agent and self.zoom_level > 0.5:
            city_name = self.get_city_name(agent)
            label = f"{self.city_marker} {city_name}"
            text = self.small_font.render(label, True, COLORS['text'])
            text_rect = text.get_rect()
            text_rect.center = (screen_x + 30, screen_y - 15)
            pygame.draw.rect(self.screen, COLORS['panel_bg'], text_rect.inflate(6, 4))
            self.screen.blit(text, text_rect)
    
    def draw_waypoint_line(self, agent_pos: Tuple[float, float], waypoint_pos: Tuple[float, float], agent: str):
        """Draw dashed line from agent to waypoint"""
        if not self.show_waypoint_lines:
            return
            
        agent_screen = self.world_to_screen(*agent_pos)
        waypoint_screen = self.world_to_screen(*waypoint_pos)
        agent_color = self.get_agent_color(agent)
        
        start_x, start_y = agent_screen
        end_x, end_y = waypoint_screen
        
        dx = end_x - start_x
        dy = end_y - start_y
        distance = sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            segments = int(distance / 15)
            for i in range(0, segments, 2):
                if i < segments - 1:
                    seg_start_x = start_x + (dx * i / segments)
                    seg_start_y = start_y + (dy * i / segments)
                    seg_end_x = start_x + (dx * (i + 1) / segments)
                    seg_end_y = start_y + (dy * (i + 1) / segments)
                    pygame.draw.line(self.screen, agent_color, 
                                   (seg_start_x, seg_start_y), (seg_end_x, seg_end_y), 2)
    
    def draw_intrusion_zone(self, x_km: float, y_km: float, radius_km: float = 9.26, is_in_conflict: bool = False, zone_type: str = 'separation'):
        """Draw intrusion protection zone
        
        Args:
            x_km: Agent x position in km
            y_km: Agent y position in km  
            radius_km: Radius of intrusion zone in km (default: 9.26 km = 5.0 NM)
            is_in_conflict: Whether agent is currently in conflict
            zone_type: Type of zone ('separation', 'collision', 'deep_breach')
        """
        if not self.show_intrusion_zones:
            return
            
        screen_x, screen_y = self.world_to_screen(x_km, y_km)
        radius_pixels = int(radius_km * self.scale)
        
        if radius_pixels > 3:
            # Different colors for different zone types
            if zone_type == 'deep_breach':
                base_color = (255, 100, 100) if is_in_conflict else (255, 150, 150)
                alpha = int(120 + 30 * sin(time.time() * 5)) if is_in_conflict else 60
                line_width = 3
            elif zone_type == 'collision':
                base_color = (255, 140, 80) if is_in_conflict else (255, 180, 120)
                alpha = int(100 + 30 * sin(time.time() * 4)) if is_in_conflict else 50
                line_width = 2
            else:  # separation zone
                base_color = (255, 80, 80) if is_in_conflict else COLORS['intrusion_zone']
                alpha = int(150 + 50 * sin(time.time() * 4)) if is_in_conflict else int(80 + 20 * sin(time.time() * 2))
                line_width = 2
            
            zone_color = (*base_color, alpha)
            zone_surface = pygame.Surface((radius_pixels * 2, radius_pixels * 2), pygame.SRCALPHA)
            pygame.draw.circle(zone_surface, zone_color, (radius_pixels, radius_pixels), radius_pixels, line_width)
            self.screen.blit(zone_surface, (screen_x - radius_pixels, screen_y - radius_pixels))
    
    def _get_ring_surface(self, radius_pixels: int, color: Tuple[int, int, int], alpha: int, line_width: int) -> pygame.Surface:
        """Get cached ring surface for performance"""
        cache_key = (radius_pixels, color, alpha, line_width)
        if cache_key not in self._ring_cache:
            ring_surface = pygame.Surface((radius_pixels * 2, radius_pixels * 2), pygame.SRCALPHA)
            ring_color = (*color, alpha)
            pygame.draw.circle(ring_surface, ring_color, (radius_pixels, radius_pixels), radius_pixels, line_width)
            self._ring_cache[cache_key] = ring_surface
        return self._ring_cache[cache_key]
    
    def draw_triple_band_halo(self, agent: str, x_km: float, y_km: float, observations: Dict):
        """Draw triple-band halo system: 5NM (green), 3NM (amber), 1NM (red)
        
        Shows graduated threat levels with pulsing on active violations
        """
        if not self.show_intrusion_zones:
            return
            
        # Calculate minimum separation to other agents
        min_separation = self.calculate_min_separation(agent, observations)
        
        screen_x, screen_y = self.world_to_screen(x_km, y_km)
        
        # Define three threat bands with environment-configured thresholds
        bands = [
            {
                'radius_km': getattr(self, 'separation_km', 9.26),  # 5 NM = 9.26 km
                'threshold_nm': 5.0,
                'color_safe': (50, 255, 50),     # Green
                'color_threat': (255, 180, 50),  # Amber
                'alpha_safe': 100,
                'alpha_threat': 180,
                'width': 2
            },
            {
                'radius_km': getattr(self, 'collision_km', 5.556),  # 3 NM = 5.556 km
                'threshold_nm': 3.0,
                'color_safe': (255, 180, 50),    # Amber
                'color_threat': (255, 100, 50),  # Orange-red
                'alpha_safe': 80,
                'alpha_threat': 200,
                'width': 2
            },
            {
                'radius_km': getattr(self, 'deep_breach_km', 1.852),  # 1 NM = 1.852 km
                'threshold_nm': 1.0,
                'color_safe': (255, 100, 50),    # Orange-red
                'color_threat': (255, 30, 30),   # Critical red
                'alpha_safe': 60,
                'alpha_threat': 220,
                'width': 3
            }
        ]
        
        # Draw bands from outer to inner
        for band in bands:
            radius_pixels = int(band['radius_km'] * self.scale)
            
            if radius_pixels > 3:
                is_violated = min_separation < band['threshold_nm']
                
                # Select color and alpha based on violation status
                if is_violated:
                    base_color = band['color_threat']
                    alpha = band['alpha_threat']
                    # Pulse effect on violated bands
                    pulse = int(30 * sin(time.time() * 6))
                    alpha = max(100, min(255, alpha + pulse))
                else:
                    base_color = band['color_safe']
                    alpha = band['alpha_safe']
                
                line_width = band['width'] if is_violated else band['width'] - 1
                
                # Use cached ring surface for performance
                ring_surface = self._get_ring_surface(radius_pixels, base_color, alpha, line_width)
                self.screen.blit(ring_surface, (screen_x - radius_pixels, screen_y - radius_pixels))
    
    def draw_trajectory(self, agent: str):
        """Draw agent trajectory with enhanced legibility and optional breadcrumbs"""
        if agent in self.agent_trajectories and len(self.agent_trajectories[agent]) > 1:
            points = []
            for x_km, y_km in self.agent_trajectories[agent]:
                screen_x, screen_y = self.world_to_screen(x_km, y_km)
                points.append((screen_x, screen_y))
            
            if len(points) > 1:
                agent_color = self.get_agent_color(agent)
                
                # Draw trajectory lines with enhanced contrast for recent steps
                for i in range(1, len(points)):
                    # Thicken and brighten last 15 steps
                    if i > len(points) - 15:
                        alpha = min(255, int(255 * 0.9))
                        line_width = 3
                    else:
                        alpha = min(255, int(255 * (i / len(points)) * 0.6))
                        line_width = 2
                    
                    if alpha > 30:
                        pygame.draw.line(self.screen, agent_color, points[i-1], points[i], line_width)
                
                # Optional breadcrumbs (step markers)
                if self.show_breadcrumbs:
                    # Show breadcrumbs every 5 steps for last 50 steps
                    recent_points = points[-50:]
                    for i, (sx, sy) in enumerate(recent_points):
                        if i % 5 == 0:
                            # Draw small circle marker
                            breadcrumb_alpha = int(100 + 155 * (i / len(recent_points)))
                            breadcrumb_surface = pygame.Surface((8, 8), pygame.SRCALPHA)
                            pygame.draw.circle(breadcrumb_surface, (*agent_color, breadcrumb_alpha), (4, 4), 3)
                            pygame.draw.circle(breadcrumb_surface, (255, 255, 255, breadcrumb_alpha), (4, 4), 3, 1)
                            self.screen.blit(breadcrumb_surface, (sx - 4, sy - 4))
    
    def track_conflicts(self, observations: Dict):
        """Track conflicts between agents for history display"""
        if len(observations) < 2:
            return
            
        # Check all pairs of agents for conflicts (separation < 5.0 NM)
        agents = list(observations.keys())
        for i, agent1 in enumerate(agents):
            for agent2 in agents[i+1:]:
                sep = self.calculate_min_separation_between_agents(agent1, agent2)
                if sep < 5.0:  # Conflict threshold
                    # Record conflict
                    conflict = (self.step_count, agent1, agent2, sep)
                    self.conflict_history.append(conflict)
                    
                    # Keep only recent conflicts
                    if len(self.conflict_history) > self.max_conflict_history:
                        self.conflict_history.pop(0)
    
    def draw_pairwise_matrix(self, x: int, y: int, observations: Dict) -> int:
        """Draw pairwise separation matrix with headers and hover tooltip"""
        agents = sorted(observations.keys())
        n = len(agents)
        if n < 2:
            return y

        cell = min(26, 360 // n)
        pad  = 18  # space for headers
        txt  = (180, 190, 200)
        grid_x, grid_y = x + pad, y + pad

        # Title row
        self.screen.blit(self.small_font.render("Pair Sep (NM):  rows→, cols↑", True, txt), (x, y))
        y += 18

        # Column headers (A1..An-1)
        for j, a in enumerate(agents[:-1]):
            lab = self.small_font.render(a, True, txt)
            cx  = grid_x + j*cell + cell//2 - lab.get_width()//2
            self.screen.blit(lab, (cx, y-2))

        # Row headers (A2..An)
        for i, a in enumerate(agents[1:], start=1):
            lab = self.small_font.render(a, True, txt)
            cy  = grid_y + i*cell + cell//2 - lab.get_height()//2
            self.screen.blit(lab, (x, cy))

        # Cells (lower triangle)
        hover_text = None
        mx, my = pygame.mouse.get_pos()
        for i in range(n):
            for j in range(i):
                a, b = agents[i], agents[j]
                sep  = self.calculate_min_separation_between_agents(a, b)

                # color by threshold
                if   sep < 3.0: color = (200, 50,  50)
                elif sep < 5.0: color = (255, 150, 50)
                elif sep < 7.0: color = (255, 200, 100)
                else:           color = ( 80, 180, 80)

                cx = grid_x + j*cell
                cy = grid_y + i*cell
                rect = pygame.Rect(cx, cy, cell-2, cell-2)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (60, 70, 80), rect, 1)

                if cell >= 22 and sep < 100:
                    num = self.small_font.render(f"{sep:.0f}", True, (255, 255, 255))
                    self.screen.blit(num, num.get_rect(center=rect.center))

                # hover tooltip
                if rect.collidepoint(mx, my):
                    hover_text = f"{a}–{b}: {sep:.1f} NM"

        if hover_text:
            tip = self.small_font.render(hover_text, True, (240, 240, 250))
            tip_rect = tip.get_rect(topleft=(grid_x, grid_y + n*cell + 6)).inflate(8, 4)
            pygame.draw.rect(self.screen, (30, 40, 50), tip_rect)
            self.screen.blit(tip, (grid_x+4, grid_y + n*cell + 8))

        return grid_y + n*cell + 18

    def calculate_min_separation_between_agents(self, agent1: str, agent2: str) -> float:
        """Calculate separation between two specific agents"""
        try:
            import bluesky as bs
            
            # Get positions for agent1
            acid1 = agent1.upper()
            ac_idx1 = bs.traf.id2idx(acid1)
            if isinstance(ac_idx1, list) and len(ac_idx1) > 0:
                ac_idx1 = ac_idx1[0]
            
            # Get positions for agent2
            acid2 = agent2.upper()
            ac_idx2 = bs.traf.id2idx(acid2)
            if isinstance(ac_idx2, list) and len(ac_idx2) > 0:
                ac_idx2 = ac_idx2[0]
            
            if (isinstance(ac_idx1, int) and ac_idx1 >= 0 and 
                isinstance(ac_idx2, int) and ac_idx2 >= 0):
                
                lat1 = float(bs.traf.lat[ac_idx1])
                lon1 = float(bs.traf.lon[ac_idx1])
                lat2 = float(bs.traf.lat[ac_idx2])
                lon2 = float(bs.traf.lon[ac_idx2])
                
                # Use exact same haversine function as training
                sep_nm = haversine_nm(lat1, lon1, lat2, lon2)
                return sep_nm
                
        except Exception as e:
            pass
        
        return 200.0  # Default large separation

    def draw_info_panel(self, observations: Dict, rewards: Dict, infos: Dict):
        """Draw production-ready info panel with visual hierarchy cards"""
        if not self.show_metrics_panel:
            return
        
        # Adaptive panel width based on screen size (28% of width, min 380px, max 500px)
        panel_width = max(380, min(500, int(self.width * 0.28)))
        panel_x = self.width - panel_width - 10
        panel_y = 10
        panel_height = self.height - 20
        
        # Panel background
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(235)
        panel_surface.fill((20, 28, 36))
        self.screen.blit(panel_surface, (panel_x, panel_y))
        pygame.draw.rect(self.screen, (80, 100, 120), (panel_x, panel_y, panel_width, panel_height), 2)
        
        content_x = panel_x + 15
        content_y = panel_y + 15
        # Calculate line height from font metrics to prevent layout issues
        line_height = max(16, self.small_font.get_linesize())
        
        # === EPISODE CARD ===
        self.draw_card_header(content_x, int(content_y), panel_width - 30, f"= {self.icon_episode}", (40, 60, 80))
        content_y += 25
        
        # Get trained scenario from checkpoint
        trained_scenario = self.render_config.get('scenario', 'Unknown')
        scenario = self.render_config.get('scenario', 'Unknown')
        checkpoint_path = Path(self.checkpoint_path) if self.checkpoint_path else None
        if checkpoint_path and checkpoint_path.exists():
            metadata_path = checkpoint_path / "training_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    trained_scenario = metadata.get("scenario", scenario)
                except:
                    pass
        
        # Key metrics row - apply max width constraint
        model_text = f"Model: {trained_scenario} → Run: {scenario}"
        model_label, _ = self._render_text_clamped(model_text, self.small_font, (200, 210, 220), panel_width - 40)
        self.screen.blit(model_label, (content_x + 5, content_y))
        content_y += line_height
        
        ep_step_text = f"Ep {self.episode_count} | Step {self.step_count} | {self.clock.get_fps():.0f} FPS"
        ep_label, _ = self._render_text_clamped(ep_step_text, self.small_font, (180, 190, 200), panel_width - 40)
        self.screen.blit(ep_label, (content_x + 5, content_y))
        content_y += line_height + 10
        
        # === CONFLICTS CARD ===
        self.draw_card_header(content_x, int(content_y), panel_width - 30, f"! {self.icon_conflicts}", (80, 40, 40))
        content_y += 25
        
        # Calculate global min separation
        global_min_sep = 200.0
        for agent in observations.keys():
            min_sep = self.calculate_min_separation(agent, observations)
            global_min_sep = min(global_min_sep, min_sep)
        
        # Big number display for min separation
        min_sep_color = COLORS['conflict'] if global_min_sep < 5.0 else COLORS['success']
        min_sep_text = f"{global_min_sep:.1f} NM"
        min_sep_label = self.large_font.render(min_sep_text, True, min_sep_color)
        self.screen.blit(min_sep_label, (content_x + 5, content_y))
        
        status_text = "Min Sep" if global_min_sep < 200 else "All Clear"
        status_label = self.small_font.render(status_text, True, (180, 190, 200))
        self.screen.blit(status_label, (content_x + 5, content_y + 22))
        content_y += 45
        
        # Pairwise separation matrix (if multiple agents)
        if self.show_pair_matrix and len(observations) > 1:
            content_y = self.draw_pairwise_matrix(content_x + 5, content_y, observations)
            content_y += 5
        
        # Timeline sparkline
        if self.show_timeline and len(self.min_sep_timeline) > 1:
            content_y = self.draw_timeline_sparkline(content_x + 5, content_y, panel_width - 50)
            content_y += 5
        
        # Conflict history list - with width constraint
        if self.conflict_history:
            for i, (timestep, agent1, agent2, sep) in enumerate(self.conflict_history[-3:]):
                if content_y + line_height > panel_y + panel_height - 150:
                    break
                conflict_text = f"T{timestep:3d}: {agent1}↔{agent2} {sep:.1f}NM"
                conflict_label, _ = self._render_text_clamped(conflict_text, self.small_font, (255, 150, 150), panel_width - 50)
                self.screen.blit(conflict_label, (content_x + 10, content_y))
                content_y += line_height * 0.9
        else:
            no_conflict_label = self.small_font.render("✓ No violations", True, COLORS['success'])
            self.screen.blit(no_conflict_label, (content_x + 10, content_y))
            content_y += line_height
        
        content_y += 10
        
        # === AGENTS CARD ===
        self.draw_card_header(content_x, int(content_y), panel_width - 30, f"> {self.icon_agents}", (40, 70, 50))
        content_y += 25
        
        # Agent status rows
        # Calculate minimum space needed per agent (2 lines + spacing)
        min_agent_space = max(line_height * 2.5, line_height * 2 + 16)
        
        for agent in sorted(observations.keys()):
            # Check if enough space remains for this agent + controls section
            if content_y + min_agent_space > panel_y + panel_height - 120:
                # Not enough space - show truncation indicator
                remaining = len([a for a in sorted(observations.keys()) if a > agent])
                if remaining > 0:
                    truncate_text = f"... +{remaining} more agent{'s' if remaining > 1 else ''}"
                    truncate_label = self.small_font.render(truncate_text, True, (150, 150, 150))
                    self.screen.blit(truncate_label, (content_x + 10, content_y))
                break
            
            # Check if waypoint reached
            is_reached = agent in self.agents_reached_waypoint
            
            reward = rewards.get(agent, 0.0)
            agent_color = self.get_agent_color(agent)
            
            # Agent indicator circle
            pygame.draw.circle(self.screen, agent_color, (content_x + 8, content_y + 7), 5)
            pygame.draw.circle(self.screen, COLORS['text'], (content_x + 8, content_y + 7), 5, 1)
            
            # Agent name - render and get actual width
            agent_text = f"{agent}"
            if is_reached:
                agent_text += " ✓"
                agent_color_text = (100, 255, 100)
            else:
                agent_color_text = (220, 230, 240)
            
            agent_label, agent_width = self._render_text_clamped(agent_text, self.small_font, agent_color_text, 60)
            self.screen.blit(agent_label, (content_x + 20, content_y))
            
            # Calculate dynamic x positions based on actual widths
            # Use font-size-aware spacing (scale with font size for consistency)
            base_spacing = max(8, int(line_height * 0.6))  # Scale spacing with line height
            current_x = content_x + 20 + agent_width + base_spacing
            available_width = panel_width - (current_x - panel_x) - 15  # remaining space in panel
            
            # Compact metrics: R | S | WP | ETA with dynamic spacing
            min_sep = self.calculate_min_separation(agent, observations)
            wp_dist = self.calculate_waypoint_distance(agent)
            eta = self.calculate_eta(agent)
            
            # Reward - adaptive max width based on available space
            if available_width > 40:
                reward_color = COLORS['success'] if reward > 0 else (COLORS['conflict'] if reward < -2 else (180, 190, 200))
                reward_text = f"R:{reward:.1f}"
                reward_max_w = min(max(45, int(available_width * 0.15)), 60)  # 15% of available or max 60px
                reward_label, reward_width = self._render_text_clamped(reward_text, self.small_font, reward_color, reward_max_w)
                self.screen.blit(reward_label, (current_x, content_y))
                current_x += reward_width + base_spacing
                available_width -= (reward_width + base_spacing)
            
            # Separation - adaptive max width
            if min_sep < 200.0 and available_width > 45:
                sep_color = COLORS['conflict'] if min_sep < 5.0 else (COLORS['success'] if min_sep > 7.0 else (255, 180, 50))
                sep_text = f"S:{min_sep:.1f}"
                sep_max_w = min(max(50, int(available_width * 0.18)), 70)  # 18% of available or max 70px
                sep_label, sep_width = self._render_text_clamped(sep_text, self.small_font, sep_color, sep_max_w)
                self.screen.blit(sep_label, (current_x, content_y))
                current_x += sep_width + base_spacing
                available_width -= (sep_width + base_spacing)
            
            # Waypoint distance and ETA - distribute remaining space intelligently
            if wp_dist < float('inf') and not is_reached and available_width > 40:
                wp_color = COLORS['success'] if wp_dist < 10.0 else (180, 190, 200)
                wp_text = f"WP:{wp_dist:.0f}"
                # Reserve space for ETA if present
                wp_max_w = int(available_width * 0.5) if (eta and eta < 3600) else int(available_width * 0.8)
                wp_max_w = min(max(50, wp_max_w), 80)  # Clamp between 50-80px
                wp_label, wp_width = self._render_text_clamped(wp_text, self.small_font, wp_color, wp_max_w)
                self.screen.blit(wp_label, (current_x, content_y))
                current_x += wp_width + base_spacing - 2  # Slightly tighter spacing for ETA
                available_width -= (wp_width + base_spacing - 2)
                
                if eta and eta < 3600 and available_width > 20:
                    eta_min = eta / 60.0
                    eta_text = f"{eta_min:.0f}m" if eta_min >= 1 else f"{eta:.0f}s"
                    eta_max_w = min(max(35, available_width), 50)
                    eta_label, _ = self._render_text_clamped(eta_text, self.small_font, (150, 160, 170), eta_max_w)
                    self.screen.blit(eta_label, (current_x, content_y))
            
            # City destination on second line - with max width
            city_name = self.get_city_name(agent)
            city_text = f"→{city_name}"
            city_label, _ = self._render_text_clamped(city_text, self.small_font, (140, 150, 160), panel_width - 50)
            # Position city text slightly below first line (70% of line height)
            city_offset = max(12, int(line_height * 0.7))
            self.screen.blit(city_label, (content_x + 20, content_y + city_offset))
            
            # Dynamic row spacing based on line height (ensures proper spacing for any font size)
            row_spacing = max(line_height * 2.2, line_height + city_offset + 6)
            content_y += row_spacing
        
        content_y += 5
        
        # === CONTROLS CARD (footer) ===
        controls_y = panel_y + panel_height - 110
        if controls_y > content_y + 10:
            self.draw_card_header(content_x, controls_y, panel_width - 30, f"+ {self.icon_controls}", (45, 45, 55))
            controls_y += 23
            
            controls = [
                "SPACE:Pause . :Step F:Frame B:Crumbs",
                "1-4:Follow ESC:Free R:Reset Q:Quit",
                "T:Traj I:Zones V:Vectors M:Panel",
                "Wheel:Zoom WASD:Pan 0:Reset View"
            ]
            
            for control in controls:
                if controls_y + line_height > panel_y + panel_height - 10:
                    break
                control_label, _ = self._render_text_clamped(control, self.small_font, (130, 140, 150), panel_width - 40)
                self.screen.blit(control_label, (content_x + 5, controls_y))
                controls_y += line_height * 0.85
    
    def draw_card_header(self, x: int, y: int, width: int, title: str, color: Tuple[int, int, int]):
        """Draw a card header with background"""
        pygame.draw.rect(self.screen, color, (x, y, width, 20))
        pygame.draw.rect(self.screen, (100, 110, 120), (x, y, width, 20), 1)
        title_label = self.small_font.render(title, True, (240, 245, 250))
        self.screen.blit(title_label, (x + 5, y + 3))
    
    def _render_text_clamped(self, text: str, font: pygame.font.Font, color: Tuple[int, int, int], 
                            max_width: int) -> Tuple[pygame.Surface, int]:
        """Render text with width constraint. If too wide, truncate with ellipsis.
        Returns (surface, actual_width)"""
        if max_width <= 0:
            return font.render("", True, color), 0
        
        label = font.render(text, True, color)
        width = label.get_width()
        
        if width <= max_width:
            return label, width
        
        # Binary search for optimal truncation point
        ellipsis = "…"
        if len(text) <= 1:
            return font.render(ellipsis, True, color), font.size(ellipsis)[0]
        
        lo, hi = 1, len(text)
        best_text = text[0] + ellipsis
        best_width = font.size(best_text)[0]
        
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid] + ellipsis
            candidate_width = font.size(candidate)[0]
            
            if candidate_width <= max_width:
                best_text = candidate
                best_width = candidate_width
                lo = mid + 1
            else:
                hi = mid - 1
        
        return font.render(best_text, True, color), best_width
    

    
    def draw_timeline_sparkline(self, x: int, y: int, width: int) -> int:
        """Draw min-separation timeline sparkline"""
        if len(self.min_sep_timeline) < 2:
            return y
        
        height = 40
        
        label = self.small_font.render("Min-Sep Timeline:", True, (180, 190, 200))
        self.screen.blit(label, (x, y))
        y += 18
        
        # Draw sparkline background
        pygame.draw.rect(self.screen, (30, 40, 50), (x, y, width, height))
        pygame.draw.rect(self.screen, (70, 80, 90), (x, y, width, height), 1)
        
        # Draw 5 NM threshold line
        threshold_y = y + height - int((5.0 / 10.0) * height)  # Assume 0-10 NM scale
        pygame.draw.line(self.screen, (255, 100, 100, 150), (x, threshold_y), (x + width, threshold_y), 1)
        
        # Draw sparkline
        points = []
        max_sep = max(10.0, max(sep for _, sep in self.min_sep_timeline))
        
        for i, (timestep, sep) in enumerate(self.min_sep_timeline[-width:]):
            px = x + int((i / len(self.min_sep_timeline[-width:])) * width)
            py = y + height - int((min(sep, max_sep) / max_sep) * height)
            points.append((px, py))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, (100, 200, 255), False, points, 2)
        
        return y + height + 5
    
    def handle_events(self, observations: Dict):
        """Handle pygame events"""
        keys = pygame.key.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        
        # Pan with keys (unless following an agent)
        if not self.follow_agent:
            pan_speed = 5.0 / self.scale
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.pan_x -= pan_speed
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.pan_x += pan_speed
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.pan_y += pan_speed
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.pan_y -= pan_speed
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.size
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                self.center_x = self.width // 2
                self.center_y = self.height // 2
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_t:
                    self.show_trajectories = not self.show_trajectories
                elif event.key == pygame.K_i:
                    self.show_intrusion_zones = not self.show_intrusion_zones
                elif event.key == pygame.K_v:
                    self.show_speed_vectors = not self.show_speed_vectors
                elif event.key == pygame.K_l:
                    self.show_waypoint_lines = not self.show_waypoint_lines
                elif event.key == pygame.K_m:
                    self.show_metrics_panel = not self.show_metrics_panel
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key == pygame.K_0:
                    self.reset_view()
                elif event.key == pygame.K_f:
                    self.frame_all_aircraft(observations)
                elif event.key == pygame.K_b:
                    self.show_breadcrumbs = not self.show_breadcrumbs
                elif event.key == pygame.K_PERIOD:
                    if self.paused:
                        self.single_step_advance = True
                elif event.key == pygame.K_1:
                    self.follow_agent = 'A1' if 'A1' in observations else None
                elif event.key == pygame.K_2:
                    self.follow_agent = 'A2' if 'A2' in observations else None
                elif event.key == pygame.K_3:
                    self.follow_agent = 'A3' if 'A3' in observations else None
                elif event.key == pygame.K_4:
                    self.follow_agent = 'A4' if 'A4' in observations else None
                elif event.key == pygame.K_ESCAPE:
                    self.follow_agent = None
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.update_zoom(1.2, *mouse_pos)
                elif event.key == pygame.K_MINUS:
                    self.update_zoom(0.8, *mouse_pos)
                    
            elif event.type == pygame.MOUSEWHEEL:
                zoom_factor = 1.1 if event.y > 0 else 0.9
                self.update_zoom(zoom_factor, *mouse_pos)
    
    def calculate_min_separation(self, agent: str, observations: Dict) -> float:
        """Calculate minimum separation using exact training environment method"""
        try:
            import bluesky as bs
            acid = agent.upper()
            ac_idx = bs.traf.id2idx(acid)
            if isinstance(ac_idx, list) and len(ac_idx) > 0:
                ac_idx = ac_idx[0]
            
            if not (isinstance(ac_idx, int) and ac_idx >= 0):
                return 200.0  # Default if agent not found
            
            try:
                agent_lat = float(bs.traf.lat[ac_idx])
                agent_lon = float(bs.traf.lon[ac_idx])
            except (IndexError, TypeError):
                return 200.0
            
            min_sep = 200.0  # Default if no other aircraft
            
            # Calculate separation to all other agents using exact training method
            for other_agent in observations.keys():
                if other_agent != agent:
                    other_acid = other_agent.upper()
                    other_idx = bs.traf.id2idx(other_acid)
                    if isinstance(other_idx, list) and len(other_idx) > 0:
                        other_idx = other_idx[0]
                    
                    if isinstance(other_idx, int) and other_idx >= 0:
                        try:
                            other_lat = float(bs.traf.lat[other_idx])
                            other_lon = float(bs.traf.lon[other_idx])
                            # Use exact same haversine function as training
                            sep_nm = haversine_nm(agent_lat, agent_lon, other_lat, other_lon)
                            min_sep = min(min_sep, sep_nm)
                        except (IndexError, TypeError):
                            continue
            
            return min_sep
            
        except Exception as e:
            return 200.0  # Fallback
    
    def calculate_waypoint_distance(self, agent: str) -> float:
        """Calculate waypoint distance using exact training environment method"""
        try:
            import bluesky as bs
            acid = agent.upper()
            ac_idx = bs.traf.id2idx(acid)
            if isinstance(ac_idx, list) and len(ac_idx) > 0:
                ac_idx = ac_idx[0]
            
            if not (isinstance(ac_idx, int) and ac_idx >= 0):
                return float('inf')
            
            # Get waypoint from environment
            waypoints = None
            if self.env and hasattr(self.env, 'par_env') and hasattr(self.env.par_env, '_agent_waypoints'):
                waypoints = self.env.par_env._agent_waypoints
            
            if not waypoints or agent not in waypoints:
                return float('inf')
            
            try:
                agent_lat = float(bs.traf.lat[ac_idx])
                agent_lon = float(bs.traf.lon[ac_idx])
                wpt_lat, wpt_lon = waypoints[agent]
                
                # Use exact same haversine function as training
                dist_nm = haversine_nm(agent_lat, agent_lon, wpt_lat, wpt_lon)
                return dist_nm
                
            except (IndexError, TypeError):
                return float('inf')
            
        except Exception as e:
            return float('inf')
    
    def calculate_eta(self, agent: str) -> Optional[float]:
        """Calculate ETA to waypoint in seconds (distance/speed)"""
        wp_dist = self.calculate_waypoint_distance(agent)
        if wp_dist >= float('inf'):
            return None
        
        _, speed = self.get_agent_heading_speed(agent)
        if speed <= 0:
            return None
        
        # Convert: distance (NM) / speed (kt) * 3600 (s/hr) = seconds
        eta_seconds = (wp_dist / speed) * 3600.0
        return eta_seconds

    def get_trajectory_info(self):
        """Get trajectory logging information from environment"""
        try:
            if self.env and hasattr(self.env, 'par_env'):
                env = self.env.par_env
                if hasattr(env, '_traj_rows') and hasattr(env, 'log_trajectories'):
                    return {
                        'enabled': env.log_trajectories,
                        'rows_logged': len(env._traj_rows) if env._traj_rows else 0,
                        'results_dir': getattr(env, 'results_dir', 'unknown'),
                        'episode_id': getattr(env, '_episode_id', 0),
                        'episode_tag': getattr(env, 'episode_tag', None)
                    }
        except:
            pass
        return {'enabled': False, 'rows_logged': 0, 'results_dir': 'unknown', 'episode_id': 0, 'episode_tag': None}

    def set_episode_tag(self, episode_num: int):
        """Set unique episode tag for trajectory CSV naming"""
        try:
            if self.env and hasattr(self.env, 'par_env'):
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                scenario = self.render_config.get('scenario', 'unknown')
                episode_tag = f"vis_ep_{episode_num:03d}_{scenario}_{timestamp}"
                
                # Set the episode_tag in the environment
                self.env.par_env.episode_tag = episode_tag
                return episode_tag
        except Exception as e:
            print(f"⚠️  Warning: Could not set episode tag: {e}")
        return None

    def reset_episode(self):
        """Reset environment for new episode with optional shift application"""
        if hasattr(self, 'gif_recorder') and self.episode_count > 0:
            episode_stats = {
                'steps': self.step_count,
                'waypoints_reached': sum(1 for traj in self.agent_trajectories.values() if len(traj) > 0),
                'total_conflicts': 0
            }
            self.gif_recorder.end_episode(episode_stats)
        
        # Report trajectory logging status before reset
        traj_info = self.get_trajectory_info()
        if traj_info['enabled'] and traj_info['rows_logged'] > 0:
            print(f"📊 Episode {self.episode_count}: Logged {traj_info['rows_logged']} trajectory rows")
        
        self.agent_trajectories = {}
        self.episode_count += 1
        self.step_count = 0
        
        # Clear conflict history for new episode
        self.conflict_history = []
        self.min_sep_timeline = []
        self.agents_reached_waypoint = set()
        
        # Set unique episode tag for this episode
        episode_tag = self.set_episode_tag(self.episode_count)
        if episode_tag:
            print(f"🏷️  Episode {self.episode_count} tag: {episode_tag}")
        
        if hasattr(self, 'gif_recorder'):
            self.gif_recorder.start_episode(self.episode_count)
        
        if self.env is not None:
            # Build reset options with shift configuration if provided
            reset_options = {}
            
            if hasattr(self, 'shift_config') and self.shift_config:
                # Convert shift config to intrashift format
                # Load scenario to get all agent IDs
                scenario_name = self.render_config.get('scenario', 'canonical_crossing')
                scenario_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios", f"{scenario_name}.json")
                
                try:
                    with open(scenario_path, 'r') as f:
                        scenario_data = json.load(f)
                    all_agents = [agent['id'] for agent in scenario_data['agents']]
                    
                    # Import shift creation function from intrashift_tester
                    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'testing'))
                    from intrashift_tester import create_targeted_shift
                    
                    # Create targeted shift configuration
                    targeted_shift = create_targeted_shift(
                        agent_id=self.shift_config['target_agent'],
                        shift_type=self.shift_config['shift_type'],
                        shift_value=self.shift_config['shift_value'],
                        scenario_agents=all_agents,
                        shift_data=None  # No complex shift data for basic shifts
                    )
                    
                    reset_options['targeted_shift'] = targeted_shift
                    
                    # Add empty env_shift (no wind changes for visualization)
                    reset_options['env_shift'] = {
                        'wind_enabled': False,
                        'wind_speed_kt': 0.0,
                        'wind_direction_deg': 0.0
                    }
                    
                    print(f"\\n🔀 Applying shift to episode {self.episode_count}:")
                    print(f"   {self.shift_config.get('description', 'Shift applied')}")
                except Exception as e:
                    print(f"\\n⚠️ Failed to apply shift: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Reset with or without shift options
            if reset_options:
                return self.env.reset(options=reset_options)
            else:
                return self.env.reset()
        else:
            return {}, {}
    
    def run_visualization(self, max_episodes: int = 10, fps: int = 8):
        """Run the visualization loop"""
        try:
            # Initial reset
            reset_result = self.reset_episode()
            if isinstance(reset_result, tuple) and len(reset_result) >= 2:
                observations, infos = reset_result
            else:
                observations, infos = {}, {}
            rewards = {agent: 0.0 for agent in observations.keys()}
            
            scenario_name = self.render_config.get('scenario', 'Unknown')
            algo_name = self.render_config.get('algo', 'Unknown')
            print(f"🚀 Starting {'trained model' if self.model_loaded else 'rule-based'} visualization")
            print(f"📋 Scenario: {scenario_name}")
            print(f"🤖 Algorithm: {algo_name}")
            
            # Main loop
            while self.running and self.episode_count <= max_episodes:
                self.handle_events(observations)
                
                if (not self.paused or self.single_step_advance) and observations:
                    if self.single_step_advance:
                        self.single_step_advance = False  # Clear single-step flag
                    # Get actions
                    actions = {}
                    for agent in observations.keys():
                        obs = observations[agent]
                        action = self.get_action(agent, obs)
                        actions[agent] = action
                    
                    # Step environment
                    try:
                        if self.env is not None:
                            new_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                            observations = new_observations
                            self.step_count += 1
                            
                            # Track conflicts for history display
                            self.track_conflicts(observations)
                            
                            # Track min-sep timeline for sparkline
                            global_min_sep = 200.0
                            for agent in observations.keys():
                                min_sep = self.calculate_min_separation(agent, observations)
                                global_min_sep = min(global_min_sep, min_sep)
                            self.min_sep_timeline.append((self.step_count, global_min_sep))
                            if len(self.min_sep_timeline) > self.max_timeline_length:
                                self.min_sep_timeline.pop(0)
                            
                            # Check for waypoints reached
                            for agent, info in infos.items():
                                if info.get('reached_waypoint', False):
                                    self.agents_reached_waypoint.add(agent)
                        else:
                            raise Exception("Environment not initialized")
                        
                    except Exception as e:
                        print(f"⚠️ Step error: {e}")
                        reset_result = self.reset_episode()
                        if isinstance(reset_result, tuple) and len(reset_result) >= 2:
                            observations, infos = reset_result
                        else:
                            observations, infos = {}, {}
                        rewards = {agent: 0.0 for agent in observations.keys()}
                        continue
                    
                    # Update trajectories
                    for agent in observations.keys():
                        x_km, y_km = self.get_agent_position(agent)
                        if agent not in self.agent_trajectories:
                            self.agent_trajectories[agent] = []
                        self.agent_trajectories[agent].append((x_km, y_km))
                        
                        if len(self.agent_trajectories[agent]) > 150:
                            self.agent_trajectories[agent].pop(0)
                    
                    # Check episode end
                    if (all(terminations.values()) or all(truncations.values()) or 
                        len(observations) == 0 or self.step_count > 800):
                        
                        waypoints_reached = sum(1 for info in infos.values() if info.get('reached_waypoint', False))
                        print(f"📊 Episode {self.episode_count}: {self.step_count} steps, {waypoints_reached} waypoints reached")
                        
                        if self.episode_count < max_episodes:
                            import time
                            time.sleep(1.5)
                            reset_result = self.reset_episode()
                            if isinstance(reset_result, tuple) and len(reset_result) >= 2:
                                observations, infos = reset_result
                            else:
                                observations, infos = {}, {}
                            rewards = {agent: 0.0 for agent in observations.keys()}
                        else:
                            break
                
                # Rendering
                self.screen.fill(COLORS['background'])
                self.draw_grid()
                self.draw_scale_compass()
                
                # Apply smooth camera follow if following an agent
                if self.follow_agent and self.follow_agent in observations:
                    target_x, target_y = self.get_agent_position(self.follow_agent)
                    # Smooth interpolation (20% per frame)
                    self.pan_x += (target_x - self.pan_x) * 0.2
                    self.pan_y += (target_y - self.pan_y) * 0.2
                
                # Draw all elements
                for agent in observations.keys():
                    x_km, y_km = self.get_agent_position(agent)
                    heading, speed = self.get_agent_heading_speed(agent)
                    
                    # Draw trajectory
                    if self.show_trajectories:
                        self.draw_trajectory(agent)
                    
                    # Draw triple-band halo system (5NM green, 3NM amber, 1NM red)
                    self.draw_triple_band_halo(agent, x_km, y_km, observations)
                    
                    # Draw additional intrusion zones if enabled and zoomed in
                    if self.zoom_level > 1.2:
                        # Draw intrusion zone using actual environment separation distance
                        current_conflicts = infos.get(agent, {}).get('current_violations', 0)
                        if isinstance(current_conflicts, bool):
                            current_conflicts = 1 if current_conflicts else 0
                        
                        # Draw collision zone (3 NM) - more critical
                        collision_radius = getattr(self, 'collision_km', 5.556)
                        self.draw_intrusion_zone(x_km, y_km, radius_km=collision_radius, 
                                               is_in_conflict=(current_conflicts > 0), 
                                               zone_type='collision')
                        
                        # Draw deep breach zone (1 NM) - most critical
                        if self.zoom_level > 2.0:
                            deep_breach_radius = getattr(self, 'deep_breach_km', 1.852)
                            self.draw_intrusion_zone(x_km, y_km, radius_km=deep_breach_radius, 
                                                   is_in_conflict=(current_conflicts > 0),
                                                   zone_type='deep_breach')
                    
                    # Draw waypoint and connection
                    wpt_pos = self.get_waypoint_position(agent)
                    if wpt_pos:
                        self.draw_waypoint(wpt_pos[0], wpt_pos[1], agent)
                        self.draw_waypoint_line((x_km, y_km), wpt_pos, agent)
                    
                    # Draw agent
                    self.draw_agent(agent, x_km, y_km, heading, speed)
                
                # Draw UI
                self.draw_info_panel(observations, rewards, infos)
                
                pygame.display.flip()
                
                # Capture frame
                if hasattr(self, 'gif_recorder'):
                    self.gif_recorder.capture_frame(self.screen)
                
                self.clock.tick(fps)
            
            print(f"\\n🎉 Visualization completed! Ran {self.episode_count} episodes.")
            
            # Final trajectory summary
            traj_info = self.get_trajectory_info()
            if traj_info['enabled']:
                print(f"📊 Trajectory logging summary:")
                print(f"   Episodes completed: {self.episode_count}")
                print(f"   Results directory: {traj_info['results_dir']}")
                
                # Show all trajectory CSV files created during this run
                results_dir = traj_info['results_dir']
                if os.path.exists(results_dir):
                    # Look for CSV files with timestamps from this session
                    import time
                    current_date = time.strftime("%Y%m%d")
                    csv_files = [f for f in os.listdir(results_dir) 
                               if f.endswith('.csv') and 'traj' in f and current_date in f]
                    
                    if csv_files:
                        print(f"   📁 Trajectory CSV files created this session:")
                        total_size = 0
                        for csv_file in sorted(csv_files):
                            file_path = os.path.join(results_dir, csv_file)
                            if os.path.exists(file_path):
                                file_size = os.path.getsize(file_path)
                                total_size += file_size
                                # Extract episode number from filename
                                episode_num = "?"
                                if "vis_ep_" in csv_file:
                                    try:
                                        episode_num = csv_file.split("vis_ep_")[1].split("_")[0]
                                    except:
                                        pass
                                print(f"      • Episode {episode_num}: {csv_file} ({file_size:,} bytes)")
                        print(f"   📊 Total data: {total_size:,} bytes across {len(csv_files)} episodes")
                    else:
                        print(f"   ⚠️  No trajectory CSV files found (may have been saved with different naming)")
            
        except KeyboardInterrupt:
            print("\\n⏹️ Visualization interrupted by user")
        except Exception as e:
            print(f"\\n❌ Error during visualization: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Final GIF save
            if hasattr(self, 'gif_recorder'):
                episode_stats = {
                    'steps': self.step_count,
                    'waypoints_reached': sum(1 for traj in self.agent_trajectories.values() if len(traj) > 0),
                    'total_conflicts': 0
                }
                self.gif_recorder.end_episode(episode_stats)
            
            pygame.quit()
            try:
                import ray
                if ray.is_initialized():
                    ray.shutdown()
            except:
                pass


def main():
    """Main function with training standards compatibility"""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Agent Air Traffic Control Visualization (Training Standards Compatible)",
        epilog="Examples:\\n"
               "  python visualize_trained_model_updated.py --scenario t_formation --checkpoint models/PPO_t_formation_20250928_095428\\n"
               "  python visualize_trained_model_updated.py --scenario head_on --checkpoint path/to/checkpoint --episodes 10",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Core training-standard parameters (REQUIRED)
    parser.add_argument("--scenario", "-s", type=str, required=True,
                       choices=["head_on", "t_formation", "parallel", "converging", "canonical_crossing",
                               "chase_2x2", "chase_3p1", "chase_4all",
                               "cross_2x2", "cross_3p1", "cross_4all", 
                               "merge_2x2", "merge_3p1", "merge_4all"],
                       help="Scenario type to visualize (required, matching training standards)")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Path to trained model checkpoint directory (required)")
    parser.add_argument("--algo", "-a", type=str, default="PPO", choices=["PPO", "SAC", "IMPALA"],
                       help="Algorithm type for checkpoint validation (default: PPO)")
    
    # Visualization parameters
    parser.add_argument("--episodes", "-e", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second")
    parser.add_argument("--width", type=int, default=1600, help="Window width")
    parser.add_argument("--height", type=int, default=1200, help="Window height")
    
    # GIF recording options
    parser.add_argument("--record-gifs", action="store_true", default=True, 
                       help="Record episodes as GIFs (default: True)")
    parser.add_argument("--no-record-gifs", action="store_false", dest="record_gifs",
                       help="Disable GIF recording")
    parser.add_argument("--gif-output-dir", default="episode_gifs",
                       help="Directory to save GIF files (default: episode_gifs)")
    parser.add_argument("--gif-name", default=None,
                       help="Custom name for GIF file (without .gif extension). If not provided, uses episode naming.")
    
    # Advanced options
    parser.add_argument("--max-steps", type=int, default=200, 
                       help="Maximum steps per episode (trained for 150, using 200 as buffer)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Trajectory logging options
    parser.add_argument("--log-trajectories", action="store_true", default=True,
                       help="Enable trajectory CSV logging (default: True)")
    parser.add_argument("--no-log-trajectories", action="store_false", dest="log_trajectories",
                       help="Disable trajectory CSV logging")
    parser.add_argument("--results-dir", default="vis_results",
                       help="Directory to save trajectory CSV files (default: vis_results)")
    
    # Shift configuration for LOS visualization
    parser.add_argument("--apply-shift", type=str, default=None,
                       help="Apply intrashift configuration (JSON string with target_agent, shift_type, shift_value, etc.)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for episode generation (default: 42)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Enhanced Multi-Agent Air Traffic Control Visualization")
    print("="*80)
    print(f"German Cities: Dresden, Berlin, Hamburg, Munich, Frankfurt, Cologne")
    print(f"Scenario: {args.scenario}")
    print(f"Algorithm: {args.algo}")
    
    # Validate and resolve checkpoint path
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint path does not exist: {checkpoint_path}")
        return 1
    
    # Validate checkpoint contains expected scenario (if metadata available)
    metadata_path = checkpoint_path / "training_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            trained_scenario = metadata.get("scenario", "unknown")
            trained_algo = metadata.get("algorithm", "unknown")
            
            if trained_scenario != args.scenario:
                print(f"WARNING: Checkpoint trained on '{trained_scenario}' but visualizing '{args.scenario}'")
                print(f"   This may result in suboptimal performance.")
            
            if trained_algo != args.algo:
                print(f"WARNING: Checkpoint uses algorithm '{trained_algo}' but specified '{args.algo}'")
                args.algo = trained_algo  # Auto-correct algorithm type
                print(f"   Auto-corrected to use '{trained_algo}'")
            
            print(f"Checkpoint validation: {trained_algo} model trained on {trained_scenario}")
        except Exception as e:
            print(f"WARNING: Could not read checkpoint metadata: {e}")
    
    print(f"Loading trained model from: {checkpoint_path}")
    
    # Parse shift configuration if provided
    shift_config = None
    if args.apply_shift:
        try:
            shift_config = json.loads(args.apply_shift)
            print(f"\n🔀 Shift Configuration Loaded:")
            print(f"   Target Agent: {shift_config.get('target_agent', 'N/A')}")
            print(f"   Shift Type: {shift_config.get('shift_type', 'N/A')}")
            print(f"   Shift Value: {shift_config.get('shift_value', 'N/A')}")
            print(f"   Range: {shift_config.get('shift_range', 'N/A')}")
            print(f"   Description: {shift_config.get('description', 'N/A')}")
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse shift configuration: {e}")
            shift_config = None
    
    # Configuration
    render_config = {
        'scenario': args.scenario,
        'algo': args.algo,
        'width': args.width,
        'height': args.height,
        'max_steps': args.max_steps,
        'record_gifs': args.record_gifs,
        'gif_output_dir': args.gif_output_dir,
        'gif_name': args.gif_name,
        'verbose': args.verbose,
        'log_trajectories': args.log_trajectories,
        'results_dir': args.results_dir,
        'shift_config': shift_config,
        'seed': args.seed,
    }
    
    print(f"Episodes: {args.episodes}")
    print(f"FPS: {args.fps}")
    print(f"Resolution: {args.width}x{args.height}")
    
    if args.record_gifs and PIL_AVAILABLE:
        print(f"GIF recording enabled. Output: {args.gif_output_dir}/")
    elif args.record_gifs and not PIL_AVAILABLE:
        print(f"GIF recording requested but PIL not available. Install with: pip install Pillow")
    else:
        print(f"GIF recording disabled")
    
    if args.log_trajectories:
        print(f"Trajectory logging enabled. Output: {args.results_dir}/")
    else:
        print(f"Trajectory logging disabled")
    
    print("\\nControls:")
    print("   SPACE: Pause/Resume | H: Help | R: Reset | Q: Quit")
    print("   Mouse Wheel: Zoom | WASD: Pan | T/I/V: Toggle elements")
    print("="*80 + "\\n")
    
    try:
        visualizer = TrainedModelVisualizer(str(checkpoint_path), render_config)
        visualizer.run_visualization(max_episodes=args.episodes, fps=args.fps)
        return 0
    except Exception as e:
        print(f"ERROR: Failed to start visualization: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())