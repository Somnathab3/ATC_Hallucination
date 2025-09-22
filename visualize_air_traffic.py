#!/usr/bin/env python3
"""
Enhanced Air Traffic Control Visualization
Shows trained PPO/SAC models in action with conflict detection and waypoint tracking
"""

import os
import sys
import time
import json
import math
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    import pygame
    import pygame.freetype
except ImportError:
    print("❌ Error: pygame not installed. Please run: pip install pygame")
    sys.exit(1)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    import ray
    from ray.rllib.algorithms.ppo import PPO
    from ray.rllib.algorithms.sac import SAC
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env
    from src.environment.marl_collision_env_minimal import MARLCollisionEnv
    import bluesky as bs
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please make sure all dependencies are installed")
    sys.exit(1)


class AirTrafficVisualizer:
    """Enhanced visualization for multi-agent air traffic control"""
    
    def __init__(self, checkpoint_path: Optional[str] = None, scenario_path: Optional[str] = None):
        # Auto-detect checkpoint and scenario if not provided
        if checkpoint_path is None:
            # Look for the parallel scenario trained model
            parallel_models_dir = "models/results_20250922_181059_Parallel/models"
            if os.path.exists(parallel_models_dir):
                checkpoint_path = parallel_models_dir
            else:
                raise FileNotFoundError("No trained model found. Please specify --checkpoint path")
        
        if scenario_path is None:
            scenario_path = "scenarios/parallel.json"
            
        # Verify scenario file exists
        if not os.path.exists(scenario_path):
            print(f"⚠️  Scenario file not found: {scenario_path}, using default")
            
        self.checkpoint_path = checkpoint_path
        self.scenario_path = scenario_path
        
        # Display settings
        self.screen_width = 1400
        self.screen_height = 1000
        self.fps = 8
        
        # Colors
        self.COLORS = {
            'background': (20, 24, 32),
            'aircraft': (0, 255, 0),
            'aircraft_conflict': (255, 100, 100),
            'aircraft_collision': (255, 0, 0),
            'waypoint': (255, 255, 0),
            'trail': (100, 150, 255),
            'separation_circle': (255, 255, 255),
            'conflict_zone': (255, 100, 100),
            'text': (255, 255, 255),
            'panel': (40, 44, 52),
            'grid': (60, 64, 72),
        }
        
        # Visualization state
        self.scale = 0.5  # nm per pixel (zoomed in more for better view)
        self.zoom_factor = 1.2  # More gradual zoom steps
        self.min_scale = 0.1  # Maximum zoom in
        self.max_scale = 5.0   # Maximum zoom out
        self.center_lat = 52.0
        self.center_lon = 5.0
        self.show_trails = True
        self.show_separation_circles = True
        self.show_info_panel = True
        self.show_grid = True
        self.paused = False
        
        # Data tracking
        self.aircraft_trails = {}
        self.max_trail_length = 50
        self.episode_count = 0
        self.step_count = 0
        self.conflicts_detected = 0
        self.collisions_detected = 0
        self.waypoints_reached = 0
        self.waypoint_threshold_nm = 5.0  # Distance threshold for waypoint completion
        self.agents_at_waypoint = set()  # Track which agents have reached waypoints
        
        # Debug tracking
        self.model_actions_used = 0
        self.rule_actions_used = 0
        self.action_debug = True
        self.last_obs_sample = None
        self.last_actions_sample = None
        
        # Initialize pygame
        pygame.init()
        pygame.freetype.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("🛩️ Multi-Agent Air Traffic Control - Enhanced Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.freetype.SysFont("Arial", 14)
        self.font_large = pygame.freetype.SysFont("Arial", 18, bold=True)
        
        # Load model and environment
        self._load_model_and_env()
        
    def _load_model_and_env(self):
        """Load the trained model and environment"""
        try:
            print(f"🔄 Loading model from: {self.checkpoint_path}")
            
            # Initialize Ray
            if not ray.is_initialized():
                ray.init(local_mode=True, log_to_driver=False, configure_logging=False)
            
            # Register environment
            register_env("marl_collision_env_v0", 
                        lambda cfg: ParallelPettingZooEnv(MARLCollisionEnv(cfg)))
            
            # Detect algorithm type
            algo_type = self._detect_algorithm_type()
            print(f"🤖 Algorithm: {algo_type}")
            
            # Convert Windows path to proper format
            checkpoint_path = os.path.abspath(self.checkpoint_path).replace('\\', '/')
            
            # Load algorithm with configuration matching training
            config = (PPO.get_default_config()
                     .environment(env="marl_collision_env_v0")
                     .framework("torch")
                     .resources(num_gpus=0)
                     .env_runners(num_env_runners=0)
                     .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False))
            
            if algo_type == "SAC":
                self.algorithm = SAC(config=config)
            else:
                self.algorithm = PPO(config=config)
            
            self.algorithm.restore(checkpoint_path)
            
            # Detect policy ID from trained model
            try:
                # Check if the algorithm has multiagent configuration
                config = getattr(self.algorithm, 'config', None)
                if config and hasattr(config, 'multi_agent'):
                    # New API style
                    multi_agent_config = getattr(config, 'multi_agent', {})
                    policies = getattr(multi_agent_config, 'policies', {})
                    policy_ids = list(policies.keys()) if policies else ["default_policy"]
                elif config and hasattr(config, 'multiagent'):
                    # Old API style
                    multiagent_config = getattr(config, 'multiagent', {})
                    policies = multiagent_config.get('policies', {})
                    policy_ids = list(policies.keys()) if policies else ["default_policy"]
                else:
                    # Single agent setup
                    policy_ids = ["default_policy"]
            except Exception as e:
                print(f"Warning: Could not detect policies, using default: {e}")
                policy_ids = ["default_policy"]
            
            print(f"Available policies: {policy_ids}")
            
            # Use the correct policy name from training (parallel scenario uses shared_policy)
            self.primary_policy = 'shared_policy' if 'shared_policy' in policy_ids else policy_ids[0] if policy_ids else 'default_policy'
            print(f"Using policy: {self.primary_policy}")
            
            # Create environment for visualization with SAME config as training
            env_config = {
                "scenario_path": self.scenario_path if os.path.exists(self.scenario_path) else None,
                "action_delay_steps": 0,
                "max_episode_steps": 100,  # MATCH TRAINING EXACTLY
                "separation_nm": 5.0,
                "log_trajectories": True,  # Match training
                "results_dir": "vis_temp",
                "seed": 42,
                
                # Team shaping knobs (EXACT SAME AS TRAINING)
                "team_coordination_weight": 0.2,
                "team_gamma": 0.99,
                "team_share_mode": "responsibility",
                "team_ema": 0.001,
                "team_cap": 0.005,
                "team_anneal": 1.0,
                "team_neighbor_threshold_km": 10.0,
            }
            
            self.env = ParallelPettingZooEnv(MARLCollisionEnv(env_config))
            self.model_loaded = True
            print("✅ Model and environment loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Creating fallback environment...")
            self._create_fallback_env()
    
    def _detect_algorithm_type(self) -> str:
        """Detect if checkpoint is PPO or SAC"""
        try:
            checkpoint_file = os.path.join(self.checkpoint_path, "rllib_checkpoint.json")
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    return data.get("type", "PPO")
        except Exception:
            pass
        return "PPO"  # Default
    
    def _create_fallback_env(self):
        """Create environment without trained model (rule-based policy)"""
        try:
            # Use SAME config as training for consistency
            env_config = {
                "scenario_path": self.scenario_path if os.path.exists(self.scenario_path) else None,
                "action_delay_steps": 0,
                "max_episode_steps": 100,  # MATCH TRAINING EXACTLY
                "separation_nm": 5.0,
                "log_trajectories": True,  # Match training
                "results_dir": "vis_temp",
                "seed": 42,
                
                # Team shaping knobs (EXACT SAME AS TRAINING)
                "team_coordination_weight": 0.2,
                "team_gamma": 0.99,
                "team_share_mode": "responsibility",
                "team_ema": 0.001,
                "team_cap": 0.005,
                "team_anneal": 1.0,
                "team_neighbor_threshold_km": 10.0,
            }
            self.env = ParallelPettingZooEnv(MARLCollisionEnv(env_config))
            self.algorithm = None
            self.model_loaded = False
            self.primary_policy = "rule_based"
            print("✅ Fallback environment created (rule-based policy)")
        except Exception as e:
            print(f"❌ Failed to create fallback environment: {e}")
            sys.exit(1)
    
    def lat_lon_to_screen(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert lat/lon to screen coordinates"""
        # Simple equirectangular projection
        x_offset = (lon - self.center_lon) * 60.0 / self.scale  # 60 nm per degree
        y_offset = (self.center_lat - lat) * 60.0 / self.scale
        
        x = self.screen_width // 2 + int(x_offset)
        y = self.screen_height // 2 + int(y_offset)
        return x, y
    
    def screen_to_lat_lon(self, x: int, y: int) -> Tuple[float, float]:
        """Convert screen coordinates to lat/lon"""
        x_offset = (x - self.screen_width // 2) * self.scale / 60.0
        y_offset = (self.screen_height // 2 - y) * self.scale / 60.0
        
        lon = self.center_lon + x_offset
        lat = self.center_lat + y_offset
        return lat, lon
    
    def draw_grid(self):
        """Draw coordinate grid"""
        if not self.show_grid:
            return
            
        grid_spacing = max(10, int(self.scale * 10))  # Grid every 10nm
        
        # Vertical lines
        for x in range(0, self.screen_width, grid_spacing):
            pygame.draw.line(self.screen, self.COLORS['grid'], (x, 0), (x, self.screen_height), 1)
        
        # Horizontal lines
        for y in range(0, self.screen_height, grid_spacing):
            pygame.draw.line(self.screen, self.COLORS['grid'], (0, y), (self.screen_width, y), 1)
    
    def draw_aircraft(self, agent_data: Dict):
        """Draw aircraft with conflict/collision indication"""
        aid = agent_data['id']
        lat = agent_data['lat']
        lon = agent_data['lon']
        hdg = agent_data.get('hdg', 0)
        conflict = agent_data.get('conflict', False)
        collision = agent_data.get('collision', False)
        at_waypoint = aid in self.agents_at_waypoint
        
        x, y = self.lat_lon_to_screen(lat, lon)
        
        # Choose color based on status
        if collision:
            color = self.COLORS['aircraft_collision']
            radius = 8
        elif conflict:
            color = self.COLORS['aircraft_conflict']
            radius = 6
        elif at_waypoint:
            color = (0, 255, 255)  # Cyan for waypoint reached
            radius = 7
        else:
            color = self.COLORS['aircraft']
            radius = 5
        
        # Draw aircraft circle
        pygame.draw.circle(self.screen, color, (x, y), radius)
        pygame.draw.circle(self.screen, self.COLORS['text'], (x, y), radius, 2)
        
        # Draw heading indicator (small line)
        hdg_rad = math.radians(90 - hdg)  # Convert to math coordinates
        end_x = x + int(15 * math.cos(hdg_rad))
        end_y = y - int(15 * math.sin(hdg_rad))
        pygame.draw.line(self.screen, color, (x, y), (end_x, end_y), 2)
        
        # Draw aircraft ID
        text_surface, _ = self.font.render(aid, self.COLORS['text'])
        self.screen.blit(text_surface, (x + 10, y - 20))
        
        # Update trail
        if aid not in self.aircraft_trails:
            self.aircraft_trails[aid] = []
        
        self.aircraft_trails[aid].append((x, y))
        if len(self.aircraft_trails[aid]) > self.max_trail_length:
            self.aircraft_trails[aid].pop(0)
    
    def draw_trails(self):
        """Draw aircraft trails"""
        if not self.show_trails:
            return
            
        for aid, trail in self.aircraft_trails.items():
            if len(trail) > 1:
                for i in range(1, len(trail)):
                    alpha = int(255 * i / len(trail))
                    color = (*self.COLORS['trail'][:3], alpha)
                    
                    # Create surface for alpha blending
                    trail_surf = pygame.Surface((2, 2))
                    trail_surf.set_alpha(alpha)
                    trail_surf.fill(self.COLORS['trail'])
                    
                    pygame.draw.line(self.screen, self.COLORS['trail'], 
                                   trail[i-1], trail[i], 2)
    
    def draw_waypoints(self, waypoints: Dict):
        """Draw waypoints for all aircraft"""
        for aid, (wpt_lat, wpt_lon) in waypoints.items():
            x, y = self.lat_lon_to_screen(wpt_lat, wpt_lon)
            
            # Draw waypoint as yellow diamond
            diamond_points = [
                (x, y - 8),     # top
                (x + 8, y),     # right
                (x, y + 8),     # bottom
                (x - 8, y)      # left
            ]
            pygame.draw.polygon(self.screen, self.COLORS['waypoint'], diamond_points)
            pygame.draw.polygon(self.screen, self.COLORS['text'], diamond_points, 2)
            
            # Draw waypoint ID
            text_surface, _ = self.font.render(f"WP-{aid}", self.COLORS['text'])
            self.screen.blit(text_surface, (x + 12, y - 8))
    
    def draw_separation_circles(self, aircraft_data: List[Dict]):
        """Draw separation circles around aircraft"""
        if not self.show_separation_circles:
            return
            
        separation_nm = 5.0  # 5 NM separation
        separation_pixels = int(separation_nm / self.scale)
        
        for aircraft in aircraft_data:
            if aircraft.get('conflict', False):
                x, y = self.lat_lon_to_screen(aircraft['lat'], aircraft['lon'])
                pygame.draw.circle(self.screen, self.COLORS['conflict_zone'], 
                                 (x, y), separation_pixels, 2)
    
    def draw_info_panel(self, aircraft_data: List[Dict], step_reward: float):
        """Draw information panel"""
        if not self.show_info_panel:
            return
            
        panel_width = 350
        panel_height = self.screen_height
        panel_x = self.screen_width - panel_width
        
        # Draw panel background
        panel_rect = pygame.Rect(panel_x, 0, panel_width, panel_height)
        pygame.draw.rect(self.screen, self.COLORS['panel'], panel_rect)
        pygame.draw.rect(self.screen, self.COLORS['text'], panel_rect, 2)
        
        y_offset = 20
        line_height = 25
        
        # Title
        title_surface, _ = self.font_large.render("🛩️ Air Traffic Control", self.COLORS['text'])
        self.screen.blit(title_surface, (panel_x + 10, y_offset))
        y_offset += 40
        
        # Episode info
        total_actions = self.model_actions_used + self.rule_actions_used
        model_pct = (self.model_actions_used / max(1, total_actions)) * 100 if total_actions > 0 else 0
        
        info_lines = [
            f"Episode: {self.episode_count}",
            f"Step: {self.step_count}",
            f"Reward: {step_reward:.1f}",
            "",
            f"Model Status: {'✅ LOADED' if self.model_loaded else '❌ FALLBACK'}",
            f"Actions: {model_pct:.1f}% model, {100-model_pct:.1f}% rule",
            f"Policy: {getattr(self, 'primary_policy', 'unknown')}",
            "",
            f"Zoom: {self.scale:.2f} nm/pixel",
            f"View: {1/self.scale:.1f}x detail",
            "",
            f"Total Conflicts: {self.conflicts_detected}",
            f"Total Collisions: {self.collisions_detected}",
            f"Waypoints Reached: {self.waypoints_reached}",
            "",
            "Aircraft Status:",
        ]
        
        for line in info_lines:
            if line:
                text_surface, _ = self.font.render(line, self.COLORS['text'])
                self.screen.blit(text_surface, (panel_x + 10, y_offset))
            y_offset += line_height
        
        # Aircraft details
        for aircraft in aircraft_data:
            aid = aircraft['id']
            at_waypoint = aid in self.agents_at_waypoint
            
            if aircraft.get('collision'):
                status = "� COLLISION"
            elif aircraft.get('conflict'):
                status = "🟡 CONFLICT"
            elif at_waypoint:
                status = "🎯 AT WAYPOINT"
            else:
                status = "🟢 NORMAL"
            
            lines = [
                f"  {aid}: {status}",
                f"    Pos: {aircraft['lat']:.2f}, {aircraft['lon']:.2f}",
                f"    Hdg: {aircraft.get('hdg', 0):.0f}°",
                f"    Spd: {aircraft.get('speed', 0):.0f} kt",
                ""
            ]
            
            for line in lines:
                if line:
                    text_surface, _ = self.font.render(line, self.COLORS['text'])
                    self.screen.blit(text_surface, (panel_x + 10, y_offset))
                y_offset += line_height
        
        # Controls
        y_offset = self.screen_height - 200
        controls = [
            "Controls:",
            "SPACE: Pause/Resume",
            "R: Reset Episode", 
            "T: Toggle Trails",
            "S: Toggle Separation",
            "I: Toggle Info Panel",
            "G: Toggle Grid",
            "+: Zoom In (More Detail)",
            "-: Zoom Out (Less Detail)",
            "Q/ESC: Quit"
        ]
        
        for line in controls:
            text_surface, _ = self.font.render(line, self.COLORS['text'])
            self.screen.blit(text_surface, (panel_x + 10, y_offset))
            y_offset += line_height
    
    def get_aircraft_data(self, observations: Dict) -> List[Dict]:
        """Extract aircraft data from BlueSky"""
        aircraft_data = []
        
        try:
            # Get data from BlueSky traffic
            for i, acid in enumerate(bs.traf.id):
                if i < len(bs.traf.lat):
                    aircraft_info = {
                        'id': acid,
                        'lat': float(bs.traf.lat[i]),
                        'lon': float(bs.traf.lon[i]),
                        'hdg': float(getattr(bs.traf, 'hdg', [0] * len(bs.traf.id))[i]),
                        'speed': float(getattr(bs.traf, 'tas', [150] * len(bs.traf.id))[i] * 1.94384),  # m/s to kt
                        'alt': float(bs.traf.alt[i] * 3.28084),  # m to ft
                        'conflict': False,
                        'collision': False
                    }
                    aircraft_data.append(aircraft_info)
        except (IndexError, AttributeError) as e:
            print(f"Warning: Error accessing BlueSky data: {e}")
        
        # Calculate conflicts and collisions
        for i, ac1 in enumerate(aircraft_data):
            for j, ac2 in enumerate(aircraft_data[i+1:], i+1):
                dist_nm = self._haversine_distance(ac1['lat'], ac1['lon'], ac2['lat'], ac2['lon'])
                
                if dist_nm < 0.3:  # Collision threshold
                    ac1['collision'] = True
                    ac2['collision'] = True
                    if not getattr(self, '_collision_logged', False):
                        self.collisions_detected += 1
                        self._collision_logged = True
                elif dist_nm < 5.0:  # Conflict threshold
                    ac1['conflict'] = True
                    ac2['conflict'] = True
                    if not getattr(self, '_conflict_logged', False):
                        self.conflicts_detected += 1
                        self._conflict_logged = True
        
        return aircraft_data
    
    def get_waypoints(self) -> Dict[str, Tuple[float, float]]:
        """Get waypoint data from environment"""
        waypoints = {}
        try:
            # Access waypoints from the environment
            if hasattr(self.env, 'par_env') and hasattr(self.env.par_env, '_agent_waypoints'):
                waypoints = self.env.par_env._agent_waypoints
        except Exception as e:
            print(f"Warning: Could not access waypoints: {e}")
        
        return waypoints
    
    def check_waypoint_completion(self, aircraft_data: List[Dict], waypoints: Dict[str, Tuple[float, float]]) -> bool:
        """Check if any agents have reached their waypoints and should terminate episode"""
        episode_should_end = False
        
        for aircraft in aircraft_data:
            agent_id = aircraft['id']
            if agent_id in waypoints:
                wpt_lat, wpt_lon = waypoints[agent_id]
                
                # Calculate distance to waypoint
                dist_to_waypoint = self._haversine_distance(
                    aircraft['lat'], aircraft['lon'], wpt_lat, wpt_lon
                )
                
                # Check if agent reached waypoint
                if dist_to_waypoint <= self.waypoint_threshold_nm:
                    if agent_id not in self.agents_at_waypoint:
                        self.agents_at_waypoint.add(agent_id)
                        self.waypoints_reached += 1
                        print(f"🎯 Agent {agent_id} reached waypoint! Distance: {dist_to_waypoint:.2f} nm")
                        
                        # Check if all agents have reached waypoints (mission complete)
                        if len(self.agents_at_waypoint) >= len(waypoints):
                            print(f"🏆 All agents reached waypoints! Episode complete.")
                            episode_should_end = True
        
        return episode_should_end
    
    def _get_rule_based_actions(self, obs: Dict) -> Dict:
        """Generate rule-based actions for fallback"""
        actions = {}
        for agent_id, observation in obs.items():
            if isinstance(observation, (list, np.ndarray)) and len(observation) >= 3:
                # Extract heading error and speed from observation
                cos_drift = observation[0]
                sin_drift = observation[1]
                norm_speed = observation[2]
                
                # Calculate desired heading correction
                drift_angle = np.arctan2(sin_drift, cos_drift)
                heading_action = np.clip(drift_angle * 0.5, -1.0, 1.0)
                
                # Speed control: maintain cruise speed
                target_speed = 0.7
                speed_error = target_speed - norm_speed
                speed_action = np.clip(speed_error * 1.5, -1.0, 1.0)
                
                actions[agent_id] = [heading_action, speed_action]
            else:
                # Random small actions if observation format is unexpected
                actions[agent_id] = [np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1)]
        
        return actions
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in nautical miles"""
        from math import radians, sin, cos, sqrt, atan2
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        R_nm = 3440.065  # Earth radius in nautical miles
        return R_nm * c
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_episode()
                elif event.key == pygame.K_t:
                    self.show_trails = not self.show_trails
                elif event.key == pygame.K_s:
                    self.show_separation_circles = not self.show_separation_circles
                elif event.key == pygame.K_i:
                    self.show_info_panel = not self.show_info_panel
                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # Zoom in (smaller scale = more detail)
                    self.scale = max(self.min_scale, self.scale / self.zoom_factor)
                    print(f"Zoomed in: {self.scale:.2f} nm/pixel")
                elif event.key == pygame.K_MINUS:
                    # Zoom out (larger scale = less detail)
                    self.scale = min(self.max_scale, self.scale * self.zoom_factor)
                    print(f"Zoomed out: {self.scale:.2f} nm/pixel")
        
        return True
    
    def reset_episode(self):
        """Reset the environment for a new episode"""
        try:
            obs, _ = self.env.reset()
            self.episode_count += 1
            self.step_count = 0
            self.aircraft_trails.clear()
            self._conflict_logged = False
            self._collision_logged = False
            
            # Reset waypoint tracking
            self.agents_at_waypoint.clear()
            
            # Reset debug counters
            self.model_actions_used = 0
            self.rule_actions_used = 0
            
            print(f"🔄 Episode {self.episode_count} started")
            return obs
        except Exception as e:
            print(f"Error resetting environment: {e}")
            return {}
    
    def run_visualization(self, max_episodes: int = 5):
        """Main visualization loop"""
        print(f"🎬 Starting visualization (max {max_episodes} episodes)")
        print("Controls: SPACE=Pause, R=Reset, T=Trails, S=Separation, I=Info, G=Grid, Q=Quit")
        
        obs = self.reset_episode()
        running = True
        total_reward = 0
        
        while running and self.episode_count <= max_episodes:
            dt = self.clock.tick(self.fps)
            
            # Handle events
            running = self.handle_events()
            if not running:
                break
            
            # Skip simulation steps if paused
            if self.paused:
                self.screen.fill(self.COLORS['background'])
                
                # Draw paused message
                pause_surface, _ = self.font_large.render("⏸️ PAUSED", self.COLORS['text'])
                pause_rect = pause_surface.get_rect(center=(self.screen_width//2, self.screen_height//2))
                self.screen.blit(pause_surface, pause_rect)
                
                pygame.display.flip()
                continue
            
            # Get actions from model or rule-based policy
            if self.model_loaded and self.algorithm and obs:
                actions = {}
                model_success_count = 0
                rule_fallback_count = 0
                
                for agent_id in obs.keys():
                    try:
                        # Use the correct policy name from training
                        action = self.algorithm.compute_single_action(
                            obs[agent_id], policy_id=self.primary_policy
                        )
                        # Handle both tuple and array returns
                        if isinstance(action, tuple):
                            action = action[0]
                        actions[agent_id] = action
                        model_success_count += 1
                        
                        # Debug first few actions
                        if self.action_debug and self.step_count < 5:
                            print(f"Model action for {agent_id}: {action} (obs shape: {np.array(obs[agent_id]).shape})")
                        
                    except Exception as e:
                        # Fallback to rule-based for this agent
                        rule_actions = self._get_rule_based_actions({agent_id: obs[agent_id]})
                        actions[agent_id] = rule_actions[agent_id]
                        rule_fallback_count += 1
                        
                        if self.action_debug and self.step_count < 5:
                            print(f"Rule action for {agent_id}: {actions[agent_id]} (model failed: {e})")
                
                # Track action source statistics
                self.model_actions_used += model_success_count
                self.rule_actions_used += rule_fallback_count
                
                # Periodic debug output
                if self.step_count % 50 == 0:
                    total_actions = self.model_actions_used + self.rule_actions_used
                    model_pct = (self.model_actions_used / max(1, total_actions)) * 100
                    print(f"Step {self.step_count}: {model_pct:.1f}% model actions, {100-model_pct:.1f}% rule-based")
                
            else:
                # Simple rule-based policy
                actions = self._get_rule_based_actions(obs)
                self.rule_actions_used += len(obs) if obs else 0
            
            # Step environment
            try:
                obs, rewards, terminations, truncations, infos = self.env.step(actions)
                self.step_count += 1
                step_reward = sum(rewards.values()) if rewards else 0
                total_reward += step_reward
                
                # Debug reward breakdown for first few steps
                if self.action_debug and self.step_count <= 10:
                    print(f"Step {self.step_count} rewards: {rewards}")
                    print(f"Step {self.step_count} total: {step_reward:.1f}, cumulative: {total_reward:.1f}")
                    if infos:
                        for agent_id, info in infos.items():
                            if isinstance(info, dict) and 'reward_breakdown' in info:
                                print(f"  {agent_id} breakdown: {info['reward_breakdown']}")
                
                # Get current aircraft and waypoint data for termination check
                aircraft_data = self.get_aircraft_data(obs)
                waypoints = self.get_waypoints()
                
                # Check for waypoint completion termination
                waypoint_complete = self.check_waypoint_completion(aircraft_data, waypoints)
                
                # Check for episode end
                if any(terminations.values()) or any(truncations.values()) or waypoint_complete:
                    end_reason = "waypoint completion" if waypoint_complete else "environment termination"
                    print(f"Episode {self.episode_count} ended at step {self.step_count} ({end_reason}). Total reward: {total_reward:.1f}")
                    obs = self.reset_episode()
                    total_reward = 0
                    
            except Exception as e:
                print(f"Error stepping environment: {e}")
                obs = self.reset_episode()
                step_reward = 0
            
            # Draw everything
            self.screen.fill(self.COLORS['background'])
            
            # Use aircraft and waypoint data from termination check (if available)
            if 'aircraft_data' not in locals():
                aircraft_data = self.get_aircraft_data(obs)
            if 'waypoints' not in locals():
                waypoints = self.get_waypoints()
            
            # Draw components
            self.draw_grid()
            self.draw_trails()
            self.draw_separation_circles(aircraft_data)
            self.draw_waypoints(waypoints)
            
            for aircraft in aircraft_data:
                self.draw_aircraft(aircraft)
            
            self.draw_info_panel(aircraft_data, step_reward)
            
            # Update display
            pygame.display.flip()
        
        print(f"🎬 Visualization completed after {self.episode_count} episodes")
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Air Traffic Control Visualization")
    parser.add_argument("--checkpoint", type=str, required=False,
                       help="Path to trained model checkpoint (auto-detects if not provided)")
    parser.add_argument("--scenario", type=str, 
                       default="scenarios/parallel.json",
                       help="Path to scenario file")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to visualize")
    
    args = parser.parse_args()
    
    # Check if files exist (only if provided)
    if args.checkpoint and not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.scenario):
        print(f"⚠️  Scenario not found: {args.scenario}, will use default scenario")
    
    # Create and run visualizer
    try:
        visualizer = AirTrafficVisualizer(args.checkpoint, args.scenario)
        visualizer.run_visualization(args.episodes)
    except KeyboardInterrupt:
        print("\n🛑 Visualization interrupted by user")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()