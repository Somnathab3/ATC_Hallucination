#!/usr/bin/env python3
"""
Enhanced Multi-Agent Air Traffic Control Visualization (Training Standards Compatible)

This script loads a trained RLlib model and visualizes it with pygame rendering.
Updated to match training CLI standards with scenario type and checkpoint validation.

Usage:
    python visualize_trained_model_updated.py --scenario t_formation --checkpoint F:\ATC_Hallucination\models\PPO_t_formation_20250928_095428
    python visualize_trained_model_updated.py --scenario head_on --checkpoint path/to/checkpoint --episodes 10
    python visualize_trained_model_updated.py --scenario canonical_crossing --checkpoint path/to/checkpoint --no-record-gifs
"""

import os
import sys
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

# Add local path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    
    def __init__(self, output_dir: str = "episode_gifs", enabled: bool = True):
        self.enabled = enabled and PIL_AVAILABLE
        self.output_dir = Path(output_dir)
        self.frames = []
        self.current_episode = 0
        self.recording = False
        
        if self.enabled:
            self.output_dir.mkdir(exist_ok=True)
            print(f"📹 GIF recording enabled. Output directory: {self.output_dir}")

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
        
        # Fonts
        base_font_size = max(20, min(28, self.height // 40))
        self.font = pygame.font.Font(None, base_font_size)
        self.small_font = pygame.font.Font(None, base_font_size - 6)
        self.large_font = pygame.font.Font(None, base_font_size + 6)
        self.title_font = pygame.font.Font(None, base_font_size + 12)
        
        # Visualization state
        self.running = True
        self.paused = False
        self.step_by_step = False
        self.show_trajectories = True
        self.show_intrusion_zones = True
        self.show_speed_vectors = True
        self.show_waypoint_lines = True
        self.show_metrics_panel = True
        self.show_help = False
        self.show_grid = True
        
        # Zoom and pan
        self.base_scale = 4.0
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
        
        # GIF recorder
        self.gif_recorder = EpisodeGIFRecorder(
            output_dir=render_config.get('gif_output_dir', 'episode_gifs'),
            enabled=render_config.get('record_gifs', True)
        )
        
        # Load model and create environment
        self.algorithm = None
        self.env = None
        self.model_loaded = False
        self._load_model()
    
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
        # Use absolute path to prevent scenario generation issues
        scenario_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios", f"{scenario_name}.json")
        
        if not os.path.exists(scenario_path):
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
        
        env_config = {
            "scenario_path": scenario_path,
            "action_delay_steps": 0,
            "max_episode_steps": self.render_config.get('max_steps', 100),  # Match training
            "separation_nm": 5.0,
            "log_trajectories": self.render_config.get('log_trajectories', True),  # Enable trajectory logging
            "seed": 42,
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
            # Use absolute path to prevent scenario generation issues
            scenario_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios", f"{scenario_name}.json")
            
            if not os.path.exists(scenario_path):
                raise FileNotFoundError(f"Scenario file not found: {scenario_path}")
            
            # Use same configuration as main environment for consistency
            env_config = {
                "scenario_path": scenario_path,
                "action_delay_steps": 0,
                "max_episode_steps": self.render_config.get('max_steps', 100),
                "separation_nm": 5.0,
                "log_trajectories": self.render_config.get('log_trajectories', True),  # Enable trajectory logging
                "seed": 42,
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
                result = self.algorithm.compute_single_action(observation, policy_id="shared_policy")
                
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
    
    def draw_agent(self, agent: str, x_km: float, y_km: float, heading: float, speed: float):
        """Draw agent aircraft"""
        screen_x, screen_y = self.world_to_screen(x_km, y_km)
        agent_color = self.get_agent_color(agent)
        
        # Aircraft size based on zoom
        size = max(6, int(12 * self.zoom_level))
        hdg_rad = radians(heading - 90)
        
        # Triangle points
        nose_x = screen_x + size * cos(hdg_rad)
        nose_y = screen_y + size * sin(hdg_rad)
        
        left_x = screen_x + size * 0.7 * cos(hdg_rad + 2.2)
        left_y = screen_y + size * 0.7 * sin(hdg_rad + 2.2)
        
        right_x = screen_x + size * 0.7 * cos(hdg_rad - 2.2)
        right_y = screen_y + size * 0.7 * sin(hdg_rad - 2.2)
        
        points = [(nose_x, nose_y), (left_x, left_y), (right_x, right_y)]
        
        # Draw aircraft
        pygame.draw.polygon(self.screen, agent_color, points)
        pygame.draw.polygon(self.screen, COLORS['text'], points, 2)
        
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
            label = f"🏙️ {city_name}"
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
    
    def draw_trajectory(self, agent: str):
        """Draw agent trajectory"""
        if agent in self.agent_trajectories and len(self.agent_trajectories[agent]) > 1:
            points = []
            for x_km, y_km in self.agent_trajectories[agent]:
                screen_x, screen_y = self.world_to_screen(x_km, y_km)
                points.append((screen_x, screen_y))
            
            if len(points) > 1:
                agent_color = self.get_agent_color(agent)
                for i in range(1, len(points)):
                    alpha = min(255, int(255 * (i / len(points)) * 0.8))
                    if alpha > 30:
                        pygame.draw.line(self.screen, agent_color, points[i-1], points[i], 2)
    
    def draw_info_panel(self, observations: Dict, rewards: Dict, infos: Dict):
        """Draw information panel"""
        if not self.show_metrics_panel:
            return
        
        panel_x = self.width - 380
        panel_y = 10
        panel_width = 360
        panel_height = self.height - 20
        
        # Panel background
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(220)
        panel_surface.fill(COLORS['panel_bg'])
        self.screen.blit(panel_surface, (panel_x, panel_y))
        pygame.draw.rect(self.screen, COLORS['text'], (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Content
        content_x = panel_x + 15
        content_y = panel_y + 15
        line_height = 22
        
        # Title
        model_type = "TRAINED MODEL" if self.model_loaded else "RULE-BASED"
        algo_type = self.render_config.get('algo', 'Unknown')
        scenario = self.render_config.get('scenario', 'Unknown')
        
        title = f"🤖 {model_type} ({algo_type})"
        title_text = self.large_font.render(title, True, COLORS['success' if self.model_loaded else 'text'])
        self.screen.blit(title_text, (content_x, content_y))
        content_y += line_height
        
        scenario_text = f"📋 Scenario: {scenario}"
        text = self.font.render(scenario_text, True, COLORS['text'])
        self.screen.blit(text, (content_x, content_y))
        content_y += line_height * 1.5
        
        # Episode info
        episode_text = f"Episode: {self.episode_count} | Step: {self.step_count}"
        text = self.font.render(episode_text, True, COLORS['text'])
        self.screen.blit(text, (content_x, content_y))
        content_y += line_height
        
        # Trajectory logging status
        traj_info = self.get_trajectory_info()
        if traj_info['enabled']:
            traj_text = f"📊 Traj: {traj_info['rows_logged']} rows logged"
            traj_color = COLORS['success'] if traj_info['rows_logged'] > 0 else COLORS['text']
        else:
            traj_text = "📊 Trajectory logging: OFF"
            traj_color = COLORS['text']
        
        text = self.small_font.render(traj_text, True, traj_color)
        self.screen.blit(text, (content_x, content_y))
        content_y += line_height * 1.5
        
        # Agent status
        for agent in sorted(observations.keys()):
            reward = rewards.get(agent, 0.0)
            info = infos.get(agent, {})
            agent_color = self.get_agent_color(agent)
            
            # Agent indicator
            pygame.draw.circle(self.screen, agent_color, (content_x + 8, content_y + 10), 8)
            pygame.draw.circle(self.screen, COLORS['text'], (content_x + 8, content_y + 10), 8, 2)
            
            # Agent info
            agent_text = f"{agent}:"
            text = self.small_font.render(agent_text, True, COLORS['text'])
            self.screen.blit(text, (content_x + 25, content_y))
            
            # Reward
            if reward > 0:
                reward_color = COLORS['success']
            elif reward > -2:
                reward_color = COLORS['text']
            else:
                reward_color = COLORS['conflict']
            
            reward_text = f"R: {reward:.2f}"
            reward_label = self.small_font.render(reward_text, True, reward_color)
            self.screen.blit(reward_label, (content_x + 25, content_y + 15))
            
            # Min separation (using exact training calculation)
            min_sep = self.calculate_min_separation(agent, observations)
            if min_sep < 200.0:
                sep_color = COLORS['conflict'] if min_sep < 5.0 else COLORS['text']
                sep_text = f"Sep: {min_sep:.1f}NM"
                sep_label = self.small_font.render(sep_text, True, sep_color)
                self.screen.blit(sep_label, (content_x + 200, content_y))
            
            # Waypoint distance (using exact training calculation)
            wp_dist = self.calculate_waypoint_distance(agent)
            if wp_dist < float('inf'):
                wp_text = f"WP: {wp_dist:.1f}NM"
                wp_label = self.small_font.render(wp_text, True, COLORS['text'])
                self.screen.blit(wp_label, (content_x + 200, content_y + 15))
            
            # City destination
            city_name = self.get_city_name(agent)
            city_text = f"→{city_name}"
            city_label = self.small_font.render(city_text, True, COLORS['text'])
            self.screen.blit(city_label, (content_x + 25, content_y + 30))
            
            content_y += line_height * 2.5
        
        # Controls
        content_y += line_height
        controls_title = self.font.render("CONTROLS", True, COLORS['text'])
        self.screen.blit(controls_title, (content_x, content_y))
        content_y += line_height
        
        controls = [
            "SPACE: Pause/Resume",
            "R: Reset episode", 
            "T: Toggle trajectories",
            "I: Toggle zones",
            "V: Toggle vectors",
            "Mouse Wheel: Zoom",
            "WASD: Pan view",
            "H: Help",
            "Q: Quit"
        ]
        
        for control in controls:
            text = self.small_font.render(control, True, COLORS['text'])
            self.screen.blit(text, (content_x, content_y))
            content_y += line_height * 0.7
    
    def handle_events(self):
        """Handle pygame events"""
        keys = pygame.key.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        
        # Pan with keys
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
        """Reset environment for new episode"""
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
        
        # Set unique episode tag for this episode
        episode_tag = self.set_episode_tag(self.episode_count)
        if episode_tag:
            print(f"🏷️  Episode {self.episode_count} tag: {episode_tag}")
        
        if hasattr(self, 'gif_recorder'):
            self.gif_recorder.start_episode(self.episode_count)
        
        if self.env is not None:
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
                self.handle_events()
                
                if not self.paused and observations:
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
                
                # Draw all elements
                for agent in observations.keys():
                    x_km, y_km = self.get_agent_position(agent)
                    heading, speed = self.get_agent_heading_speed(agent)
                    
                    # Draw trajectory
                    if self.show_trajectories:
                        self.draw_trajectory(agent)
                    
                    # Draw intrusion zone using actual environment separation distance
                    current_conflicts = infos.get(agent, {}).get('current_violations', 0)
                    if isinstance(current_conflicts, bool):
                        current_conflicts = 1 if current_conflicts else 0
                    
                    # Draw multiple zone rings for better visualization
                    separation_radius = getattr(self, 'separation_km', 9.26)
                    collision_radius = getattr(self, 'collision_km', 5.556)
                    deep_breach_radius = getattr(self, 'deep_breach_km', 1.852)
                    
                    # Draw outermost separation zone (5 NM) - main intrusion zone
                    self.draw_intrusion_zone(x_km, y_km, radius_km=separation_radius, is_in_conflict=(current_conflicts > 0))
                    
                    # Optionally draw inner zones if zoomed in enough
                    if self.zoom_level > 0.8:
                        # Draw collision zone (3 NM) - more critical
                        self.draw_intrusion_zone(x_km, y_km, radius_km=collision_radius, 
                                               is_in_conflict=(current_conflicts > 0), 
                                               zone_type='collision')
                        
                        # Draw deep breach zone (1 NM) - most critical
                        if self.zoom_level > 1.5:
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
                       choices=["head_on", "t_formation", "parallel", "converging", "canonical_crossing"],
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
    
    # Advanced options
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Trajectory logging options
    parser.add_argument("--log-trajectories", action="store_true", default=True,
                       help="Enable trajectory CSV logging (default: True)")
    parser.add_argument("--no-log-trajectories", action="store_false", dest="log_trajectories",
                       help="Disable trajectory CSV logging")
    parser.add_argument("--results-dir", default="vis_results",
                       help="Directory to save trajectory CSV files (default: vis_results)")
    
    args = parser.parse_args()
    
    print(f"🎬 Enhanced Multi-Agent Air Traffic Control Visualization")
    print(f"🏙️ German Cities: Dresden, Berlin, Hamburg, Munich, Frankfurt, Cologne")
    print(f"📋 Scenario: {args.scenario}")
    print(f"🤖 Algorithm: {args.algo}")
    
    # Validate and resolve checkpoint path
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint path does not exist: {checkpoint_path}")
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
                print(f"⚠️ WARNING: Checkpoint trained on '{trained_scenario}' but visualizing '{args.scenario}'")
                print(f"   This may result in suboptimal performance.")
            
            if trained_algo != args.algo:
                print(f"⚠️ WARNING: Checkpoint uses algorithm '{trained_algo}' but specified '{args.algo}'")
                args.algo = trained_algo  # Auto-correct algorithm type
                print(f"   Auto-corrected to use '{trained_algo}'")
            
            print(f"✅ Checkpoint validation: {trained_algo} model trained on {trained_scenario}")
        except Exception as e:
            print(f"⚠️ Could not read checkpoint metadata: {e}")
    
    print(f"📁 Loading trained model from: {checkpoint_path}")
    
    # Configuration
    render_config = {
        'scenario': args.scenario,
        'algo': args.algo,
        'width': args.width,
        'height': args.height,
        'max_steps': args.max_steps,
        'record_gifs': args.record_gifs,
        'gif_output_dir': args.gif_output_dir,
        'verbose': args.verbose,
        'log_trajectories': args.log_trajectories,
        'results_dir': args.results_dir,
    }
    
    print(f"🎞️ Episodes: {args.episodes}")
    print(f"⚡ FPS: {args.fps}")
    print(f"📺 Resolution: {args.width}x{args.height}")
    
    if args.record_gifs and PIL_AVAILABLE:
        print(f"📹 GIF recording enabled. Output: {args.gif_output_dir}/")
    elif args.record_gifs and not PIL_AVAILABLE:
        print(f"📹 GIF recording requested but PIL not available. Install with: pip install Pillow")
    else:
        print(f"📹 GIF recording disabled")
    
    if args.log_trajectories:
        print(f"📊 Trajectory logging enabled. Output: {args.results_dir}/")
    else:
        print(f"📊 Trajectory logging disabled")
    
    print("\\n🎮 Controls:")
    print("   SPACE: Pause/Resume | H: Help | R: Reset | Q: Quit")
    print("   Mouse Wheel: Zoom | WASD: Pan | T/I/V: Toggle elements\\n")
    
    try:
        visualizer = TrainedModelVisualizer(str(checkpoint_path), render_config)
        visualizer.run_visualization(max_episodes=args.episodes, fps=args.fps)
        return 0
    except Exception as e:
        print(f"❌ Failed to start visualization: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())