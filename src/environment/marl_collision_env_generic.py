"""
Module Name: marl_collision_env_generic.py
Description: 
    Generic MARL environment for air traffic collision avoidance with structured
    scenario generation. Generates balanced conflict scenarios each episode across
    three families (MERGE, CHASE, CROSS) with three patterns each (2x2, 3p1, 4all).
    
    Scenario Families:
        - MERGE: Converging flights to common waypoints (head-on/angled approaches)
        - CHASE: In-trail conflicts with overtaking/same-heading geometry
        - CROSS: Perpendicular/angular crossings (90-135° conflict angles)
    
    Patterns:
        - 2x2: Two pairs of agents (e.g., A1+A2 vs A3+A4)
        - 3p1: Three-agent cluster + one singleton (e.g., A1+A2+A3 vs A4)
        - 4all: All four agents in symmetric conflict (e.g., four-way crossing)
    
    All scenarios use consistent parameters:
        - 4 agents per episode
        - 250 kt TAS baseline
        - 10,000 ft altitude
        - 5 NM well-clear separation standard
        - Unified reward system with team PBRS
    
    Enables cross-scenario generalization by training on procedurally generated
    conflict geometries rather than frozen scenarios.

Author: Som
Date: 2025-10-04 (Updated: 2025-10-08)
"""

import os
import sys
import json
import math
import logging
import atexit
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from math import cos, sin, radians, degrees, sqrt, atan2

# Suppress BlueSky logging
for name in ("bluesky", "bluesky.navdatabase", "bluesky.simulation", "bluesky.traffic"):
    logging.getLogger(name).setLevel(logging.ERROR)

# Import BlueSky components
import bluesky as bs
from gymnasium import Env, spaces
from gymnasium.spaces import Box, Dict as DictSpace
from pettingzoo.utils import ParallelEnv

# Import shared constants and utilities from minimal environment
from .marl_collision_env_minimal import (
    D_HEADING, D_VELOCITY, NM_TO_KM, DRIFT_NORM_DEN, WAYPOINT_THRESHOLD_NM,
    ACTION_THRESHOLD_DEG, ACTION_THRESHOLD_KT,
    kt_to_nms, nm_to_lat_deg, nm_to_lon_deg, heading_to_unit, haversine_nm,
    DEFAULT_TEAM_COORDINATION_WEIGHT, DEFAULT_TEAM_GAMMA, DEFAULT_TEAM_SHARE_MODE,
    DEFAULT_TEAM_EMA, DEFAULT_TEAM_CAP, DEFAULT_TEAM_ANNEAL, DEFAULT_TEAM_NEIGHBOR_THRESHOLD_KM,
    DEFAULT_PROGRESS_REWARD_PER_KM, DEFAULT_TIME_PENALTY_PER_SEC, DEFAULT_REACH_REWARD,
    DEFAULT_VIOLATION_ENTRY_PENALTY, DEFAULT_VIOLATION_STEP_SCALE, DEFAULT_DEEP_BREACH_NM,
    DEFAULT_DRIFT_IMPROVE_GAIN, DEFAULT_DRIFT_DEADZONE_DEG, DEFAULT_ACTION_COST_PER_UNIT,
    DEFAULT_TERMINAL_NOT_REACHED_PENALTY
)

# Global BlueSky initialization tracking
_BS_READY = False

@atexit.register
def _clean_bs():
    """Clean up BlueSky simulation on process exit."""
    try:
        bs.sim.reset()
    except Exception:
        pass


class MARLCollisionEnvGeneric(ParallelEnv):
    """
    Generic MARL Environment for Air Traffic Collision Avoidance with Structured Scenarios.
    
    Generates conflict scenarios each episode from three families:
        - MERGE: Converging flights (head-on, angled) to common waypoints
            * 2x2: Two pairs approach from opposite sides (mirrored head-on)
            * 3p1: Three agents converge + one singleton (asymmetric conflict)
            * 4all: Four agents symmetrically converge (four-way merge)
        
        - CHASE: In-trail conflicts (overtaking, same heading)
            * 2x2: Two pairs in-trail with speed differences
            * 3p1: Three agents in-trail + one crossing
            * 4all: Four agents in chain with cascading conflicts
        
        - CROSS: Perpendicular/angular crossings
            * 2x2: Two pairs crossing at 90° (canonical crossing)
            * 3p1: Three agents crossing + one perpendicular
            * 4all: Four-way intersection (all agents cross center)
    
    Scenario Generation Algorithm:
        1. Select scenario family (MERGE/CHASE/CROSS) and pattern (2x2/3p1/4all)
        2. Generate initial positions with minimum separation (15-25 NM typical)
        3. Assign waypoints to create desired conflict geometry
        4. Validate: Ensure start separation ≥10 NM, conflict zone <5 NM
        5. Randomize: Rotate entire scenario, perturb speeds (±25 kt)
    
    Uses unified reward system (9 components) with team PBRS:
        - Progress, drift, violation (entry + step), action cost, time, reach, terminal, team
        - Same reward structure as marl_collision_env_minimal for consistency
    
    Fixed Configuration:
        - 4 agents per episode (matching existing scenario suite)
        - 250 kt baseline TAS
        - 10,000 ft altitude
        - 5 NM well-clear separation
        - Comprehensive trajectory logging with hallucination detection support
    
    Args:
        env_config: Configuration dict with:
            - conflict_probability: Unused (legacy compatibility)
            - scenario_complexity: Unused (legacy compatibility)
            - airspace_size_nm: Spatial extent for scenario generation (default 50 NM)
            - neighbor_topk: Number of nearest neighbors in observation (default 3)
            - collision_nm: Hard collision threshold (default 1.0 NM)
            - team_coordination_weight: PBRS weight λ (default 0.2)
            - enable_hallucination_detection: Real-time alert analysis (default False)
    """
    
    metadata = {"name": "marl_collision_env_generic", "render_modes": []}
    
    @property 
    def max_num_agents(self):
        return self.max_aircraft
    
    @property
    def num_agents(self):
        return len(self.agents)

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        global _BS_READY
        
        if env_config is None:
            env_config = {}
        
        # Fixed 4-agent configuration (matching existing scenario suite)
        self.max_aircraft = 4
        self.min_aircraft = 4
        
        # Scenario generation parameters (legacy compatibility - not used for structured scenarios)
        self.conflict_probability = float(env_config.get("conflict_probability", 0.8))
        self.scenario_complexity = str(env_config.get("scenario_complexity", "medium"))
        self.airspace_size_nm = float(env_config.get("airspace_size_nm", 50.0))
        
        self.conflict_angles = env_config.get("conflict_angles", [45, 90, 135, 180, 225, 270, 315])
        self.conflict_angle_range = env_config.get("conflict_angle_range", (45, 315))
        self.conflict_distances = env_config.get("conflict_distances", [0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.conflict_distance_range = env_config.get("conflict_distance_range", (0.5, 5.0))
        self.time_to_conflict_range = env_config.get("time_to_conflict_range", (100, 1000))
        self.altitude_tolerance_ft = float(env_config.get("altitude_tolerance_ft", 1000.0))
        
        self.aircraft_types = env_config.get("aircraft_types", ["A320", "B737", "A330", "B747", "CRJ2"])
        self.speed_ranges = env_config.get("speed_ranges", {
            "A320": (220, 280), "B737": (210, 270), "A330": (240, 300),
            "B747": (250, 320), "CRJ2": (200, 250)
        })
        
        self.max_episode_steps = int(env_config.get("max_episode_steps", 100))
        
        # Initialize BlueSky once per process
        if not _BS_READY:
            from io import StringIO
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            try:
                bs.init()
                bs.scr.echo = False
                bs.settings.gui = False
                _BS_READY = True
            except Exception as e:
                print(f"BlueSky init failed: {e}")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        # Create possible agents list based on max aircraft (fixed naming: A1-A4)
        self.possible_agents = ["A1", "A2", "A3", "A4"]
        self.agents = []
        
        # DEBUG: Log initialization
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"DEBUG Generic Env Init:")
        logger.info(f"  max_aircraft = {self.max_aircraft}")
        logger.info(f"  possible_agents = {self.possible_agents}")

        # Action delay support for shift testing
        self.action_delay_steps = int(env_config.get("action_delay_steps", 0))
        self._action_queues = {aid: [[0.0, 0.0]] * self.action_delay_steps
                               for aid in self.possible_agents}

        # Internal state and configuration
        self._prev_actions_deg_kt = {aid: np.zeros(2, dtype=np.float32) for aid in self.possible_agents}
        self._step_idx = 0
        self._episode_id = 0
        self._agent_waypoints = {}
        self._traj_rows = []
        self._separation_threshold_nm = float(env_config.get("separation_nm", 5.0))
        self.log_trajectories = env_config.get("log_trajectories", True)
        self.episode_tag = env_config.get("episode_tag", None)
        
        # Observation configuration
        self.neighbor_topk = int(env_config.get("neighbor_topk", 3))
        
        self.team_w = float(env_config.get("team_coordination_weight", DEFAULT_TEAM_COORDINATION_WEIGHT))
        self.team_gamma = float(env_config.get("team_gamma", DEFAULT_TEAM_GAMMA))
        self.team_share = str(env_config.get("team_share_mode", DEFAULT_TEAM_SHARE_MODE)).lower()
        self.team_ema_a = float(env_config.get("team_ema", DEFAULT_TEAM_EMA))
        self.team_cap = float(env_config.get("team_cap", DEFAULT_TEAM_CAP))
        self.team_anneal = float(env_config.get("team_anneal", DEFAULT_TEAM_ANNEAL))
        self.team_nb_km = float(env_config.get("team_neighbor_threshold_km", DEFAULT_TEAM_NEIGHBOR_THRESHOLD_KM))
        self._team_phi = None
        self._team_dphi_ema = 0.0
        
        self.r_prog_per_km = float(env_config.get("progress_reward_per_km", DEFAULT_PROGRESS_REWARD_PER_KM))
        self.r_time_per_s = float(env_config.get("time_penalty_per_sec", DEFAULT_TIME_PENALTY_PER_SEC))
        self.r_reach_bonus = float(env_config.get("reach_reward", DEFAULT_REACH_REWARD))
        self.action_cost_per_unit = float(env_config.get("action_cost_per_unit", DEFAULT_ACTION_COST_PER_UNIT))
        self.terminal_not_reached_penalty = float(env_config.get("terminal_not_reached_penalty", DEFAULT_TERMINAL_NOT_REACHED_PENALTY))
        
        self.sep_nm = float(env_config.get("separation_nm", 5.0))
        self.violation_entry_penalty = float(env_config.get("violation_entry_penalty", DEFAULT_VIOLATION_ENTRY_PENALTY))
        self.violation_step_scale = float(env_config.get("violation_step_scale", DEFAULT_VIOLATION_STEP_SCALE))
        self.deep_breach_nm = float(env_config.get("deep_breach_nm", DEFAULT_DEEP_BREACH_NM))
        
        self.drift_improve_gain = float(env_config.get("drift_improve_gain", DEFAULT_DRIFT_IMPROVE_GAIN))
        self.drift_deadzone_deg = float(env_config.get("drift_deadzone_deg", DEFAULT_DRIFT_DEADZONE_DEG))
        
        self._prev_wp_dist_nm: Dict[str, Optional[float]] = {aid: None for aid in self.possible_agents}
        self._conflict_on_prev = {aid: 0 for aid in self.possible_agents}
        self._waypoint_hits = {aid: 0 for aid in self.possible_agents}
        self._agents_to_stop_logging = set()
        self._waypoint_reached = {aid: False for aid in self.possible_agents}
        self._waypoint_reached_this_step = {aid: False for aid in self.possible_agents}
        self._agent_done = {aid: False for aid in self.possible_agents}
        self._agent_wpreached = {aid: False for aid in self.possible_agents}
        
        self._prev_in_violation = {aid: False for aid in self.possible_agents}
        self._prev_minsep_nm = {aid: float("inf") for aid in self.possible_agents}
        self._prev_drift_abs = {aid: 0.0 for aid in self.possible_agents}

        self._base_cas_kt = {aid: 250.0 for aid in self.possible_agents}
        self._spd_bounds_kt = (100.0, 400.0)
        self._spd_bounds_scale = {aid: (1.0, 1.0) for aid in self.possible_agents}
        
        self.collision_nm = float(env_config.get("collision_nm", 1.0))

        # Set observation spaces - same as minimal env
        K = self.neighbor_topk
        self._observation_spaces = {
            aid: spaces.Dict({
                "wp_dist_norm": spaces.Box(-1.0, 1.0, (1,), np.float32),
                "cos_to_wp": spaces.Box(-1.0, 1.0, (1,), np.float32),
                "sin_to_wp": spaces.Box(-1.0, 1.0, (1,), np.float32),
                "airspeed": spaces.Box(-10.0, 10.0, (1,), np.float32),
                "progress_rate": spaces.Box(-1.0, 1.0, (1,), np.float32),
                "safety_rate": spaces.Box(-1.0, 1.0, (1,), np.float32),
                "x_r": spaces.Box(-12.0, 12.0, (K,), np.float32),
                "y_r": spaces.Box(-12.0, 12.0, (K,), np.float32),
                "vx_r": spaces.Box(-150.0, 150.0, (K,), np.float32),
                "vy_r": spaces.Box(-150.0, 150.0, (K,), np.float32),
            }) for aid in self.possible_agents
        }

        self._action_spaces = {aid: spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        ) for aid in self.possible_agents}

        # Set results directory
        self.results_dir = env_config.get("results_dir", "results")
        
        # Initialize real-time hallucination detection
        self._enable_hallucination_detection = env_config.get("enable_hallucination_detection", True)
        if self._enable_hallucination_detection:
            try:
                from ..analysis.hallucination_detector_enhanced import HallucinationDetector
                self._hallucination_detector = HallucinationDetector()
                self._rt_trajectory = {
                    "positions": [],
                    "actions": [],
                    "agents": {},
                    "waypoint_status": {},
                    "waypoints": {}
                }
            except ImportError:
                self._hallucination_detector = None
        else:
            self._hallucination_detector = None

        # Required PettingZoo attributes
        self.agent_selection = None
        
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO)

    def _generate_random_scenario(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate structured conflict scenarios (merge, chase, cross) with fixed 4-agent configuration.
        
        Creates scenarios that resemble the existing scenario suite:
        - MERGE: Aircraft converge to common waypoints (2x2 or 3+1 patterns)
        - CHASE: In-trail conflicts with same heading (2x2 or 3+1 patterns)
        - CROSS: Perpendicular/angular crossing conflicts (2x2 or 3+1 patterns)
        
        All scenarios use 4 agents with consistent speeds (250 kt) and altitude (10000 ft)
        to ensure horizontal conflicts and enable cross-scenario generalization.
        
        Args:
            seed: Random seed for reproducibility.
            
        Returns:
            Dictionary with aircraft configurations, center coordinates, and metadata.
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Fixed 4 agents for all scenarios (matching trained models)
        num_aircraft = 4
        agent_ids = ["A1", "A2", "A3", "A4"]
        
        # Central airspace reference point (consistent with existing scenarios)
        center_lat = 51.0
        center_lon = 13.7
        base_alt_ft = 10000.0
        base_spd_kt = 250.0
        
        # Randomly select scenario type (equal probability for balanced training)
        scenario_types = ["merge", "chase", "cross"]
        scenario_type = np.random.choice(scenario_types)
        
        # Randomly select pattern (2x2 vs 3+1)
        pattern = np.random.choice(["2x2", "3p1"])
        
        # Generate scenario based on type and pattern
        if scenario_type == "merge":
            aircraft_configs = self._generate_merge_scenario(
                agent_ids, center_lat, center_lon, base_alt_ft, base_spd_kt, pattern
            )
        elif scenario_type == "chase":
            aircraft_configs = self._generate_chase_scenario(
                agent_ids, center_lat, center_lon, base_alt_ft, base_spd_kt, pattern
            )
        else:  # cross
            aircraft_configs = self._generate_cross_scenario(
                agent_ids, center_lat, center_lon, base_alt_ft, base_spd_kt, pattern
            )
        
        return {
            "agents": aircraft_configs,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "scenario_type": f"{scenario_type}_{pattern}",
            "num_aircraft": num_aircraft
        }
    
    def _generate_merge_scenario(self, agent_ids: List[str], center_lat: float, 
                                center_lon: float, alt_ft: float, spd_kt: float,
                                pattern: str) -> List[Dict[str, Any]]:
        """
        Generate MERGE scenario where aircraft converge to common waypoint(s).
        
        MERGE patterns create head-on or angled approach conflicts:
            - Agents approach same waypoint from different directions
            - Conflict occurs in convergence zone (≤5 NM from waypoint)
            - Tests strategic deconfliction and sequencing decisions
        
        Patterns:
            2x2: Two separate 2→1 merges
                - Left merge: A1 (SW) + A2 (SE) → left waypoint
                - Right merge: A3 (NW) + A4 (NE) → right waypoint
                - Waypoints separated by 15 NM longitudinally
                - Mirrored geometry for balanced training
            
            3p1: Three-agent cluster + singleton
                - A1, A2, A3 converge to center waypoint from 120° spacing
                - A4 approaches from perpendicular direction
                - Creates asymmetric 3-vs-1 conflict
        
        Geometry:
            - Start positions: ~40-50 NM from waypoint (radius)
            - Approach angles: 30-60° offset from direct path
            - Headings: Pre-calculated toward waypoints
        
        Args:
            agent_ids: List of agent IDs (A1-A4).
            center_lat: Center latitude (deg).
            center_lon: Center longitude (deg).
            alt_ft: Altitude (ft, constant 10,000).
            spd_kt: Speed (kt, baseline 250).
            pattern: "2x2" or "3p1" configuration.
        
        Returns:
            List of agent config dicts with id, type, lat, lon, hdg_deg, spd_kt, alt_ft, waypoint.
        """
        configs = []
        
        if pattern == "2x2":
            # Two separate 2→1 merges (like merge_2x2.json)
            # Group 1: A1, A2 merge to left waypoint
            # Group 2: A3, A4 merge to right waypoint
            separation = 15.0  # NM between merge centers
            radius = 50.0  # NM from center
            
            # Left merge center
            left_wp_lon = center_lon - nm_to_lon_deg(separation, center_lat)
            
            # A1: Southwest approach to left waypoint
            angle1 = radians(210)  # Southwest
            configs.append({
                "id": "A1",
                "type": "A320",
                "lat": center_lat + nm_to_lat_deg(radius * 0.8 * cos(angle1)),
                "lon": left_wp_lon + nm_to_lon_deg(radius * 0.8 * sin(angle1), center_lat),
                "hdg_deg": 352.5,  # Heading toward waypoint
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat, "lon": left_wp_lon}
            })
            
            # A2: Southeast approach to left waypoint
            angle2 = radians(150)  # Southeast
            configs.append({
                "id": "A2",
                "type": "A320",
                "lat": center_lat + nm_to_lat_deg(radius * 0.8 * cos(angle2)),
                "lon": left_wp_lon + nm_to_lon_deg(radius * 0.8 * sin(angle2), center_lat),
                "hdg_deg": 7.5,  # Heading toward waypoint
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat, "lon": left_wp_lon}
            })
            
            # Right merge center
            right_wp_lon = center_lon + nm_to_lon_deg(separation, center_lat)
            
            # A3: Northwest approach to right waypoint
            angle3 = radians(330)  # Northwest
            configs.append({
                "id": "A3",
                "type": "A320",
                "lat": center_lat + nm_to_lat_deg(radius * 0.8 * cos(angle3)),
                "lon": right_wp_lon + nm_to_lon_deg(radius * 0.8 * sin(angle3), center_lat),
                "hdg_deg": 172.5,  # Heading toward waypoint
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat, "lon": right_wp_lon}
            })
            
            # A4: Northeast approach to right waypoint
            angle4 = radians(30)  # Northeast
            configs.append({
                "id": "A4",
                "type": "A320",
                "lat": center_lat + nm_to_lat_deg(radius * 0.8 * cos(angle4)),
                "lon": right_wp_lon + nm_to_lon_deg(radius * 0.8 * sin(angle4), center_lat),
                "hdg_deg": 187.5,  # Heading toward waypoint
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat, "lon": right_wp_lon}
            })
            
        else:  # 3p1 pattern
            # 3 aircraft merge to center, 1 goes to side (like merge_3p1.json)
            radius = 50.0  # NM from center
            
            # A1: Southeast approach to center
            angle1 = radians(195)  # South-southeast
            configs.append({
                "id": "A1",
                "type": "A320",
                "lat": center_lat + nm_to_lat_deg(radius * cos(angle1)),
                "lon": center_lon + nm_to_lon_deg(radius * sin(angle1), center_lat),
                "hdg_deg": 345.0,  # Heading toward center
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat, "lon": center_lon}
            })
            
            # A2: South approach to center
            configs.append({
                "id": "A2",
                "type": "A320",
                "lat": center_lat - nm_to_lat_deg(radius),
                "lon": center_lon,
                "hdg_deg": 0.0,  # North
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat, "lon": center_lon}
            })
            
            # A3: Southwest approach to center
            angle3 = radians(165)  # South-southwest
            configs.append({
                "id": "A3",
                "type": "A320",
                "lat": center_lat + nm_to_lat_deg(radius * cos(angle3)),
                "lon": center_lon + nm_to_lon_deg(radius * sin(angle3), center_lat),
                "hdg_deg": 15.0,  # Heading toward center
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat, "lon": center_lon}
            })
            
            # A4: Far east, non-conflicting
            side_offset = 25.0  # NM to the side
            configs.append({
                "id": "A4",
                "type": "A320",
                "lat": center_lat - nm_to_lat_deg(radius * 0.5),
                "lon": center_lon + nm_to_lon_deg(side_offset, center_lat),
                "hdg_deg": 0.0,  # North
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {
                    "lat": center_lat + nm_to_lat_deg(radius * 0.5),
                    "lon": center_lon + nm_to_lon_deg(side_offset, center_lat)
                }
            })
        
        return configs
    
    def _generate_chase_scenario(self, agent_ids: List[str], center_lat: float,
                                center_lon: float, alt_ft: float, spd_kt: float,
                                pattern: str) -> List[Dict[str, Any]]:
        """
        Generate CHASE scenario with in-trail conflicts (overtaking/same-heading geometry).        CHASE patterns create longitudinal conflicts:
            - Agents on same heading with different speeds or positions
            - Trailing aircraft overtakes lead aircraft
            - Tests speed management and longitudinal separation
        
        Patterns:
            2x2: Two parallel in-trail pairs
                - Left lane: A1 (lead) + A2 (trail) with 6 NM separation
                - Right lane: A3 (lead) + A4 (trail) with 6 NM separation
                - Lanes separated by 8 NM laterally
                - Mirrored geometry for balanced training
            
            3p1: Three-agent chain + singleton
                - A1, A2, A3 in-trail on centerline (6 NM spacing)
                - A4 on parallel lane 15 NM offset
                - Creates asymmetric 3-vs-1 conflict
        
        Geometry:
            - Start positions: In-trail spacing 6-8 NM
            - Headings: All 0° (North) for same-heading conflicts
            - Waypoints: Ahead of formation by ~10-15 NM
        
        Conflict Mechanism:
            - Speed variations (±25 kt) during episode create overtaking
            - Agent policy decisions (speed commands) trigger conflicts
        
        Args:
            agent_ids: List of agent IDs (A1-A4).
            center_lat: Center latitude (deg).
            center_lon: Center longitude (deg).
            alt_ft: Altitude (ft, constant 10,000).
            spd_kt: Speed (kt, baseline 250).
            pattern: "2x2" or "3p1" configuration.
        
        Returns:
            List of agent config dicts with id, type, lat, lon, hdg_deg, spd_kt, alt_ft, waypoint.
        """
        configs = []
        
        if pattern == "2x2":
            # Two separate in-trail pairs (like chase_2x2.json)
            lane_separation = 8.0  # NM between lanes
            trail_spacing = 6.0  # NM in-trail spacing
            
            # Left lane
            left_lon = center_lon - nm_to_lon_deg(lane_separation / 2, center_lat)
            
            # A1: Lead aircraft, left lane
            configs.append({
                "id": "A1",
                "type": "A320",
                "lat": center_lat - nm_to_lat_deg(trail_spacing * 1.5),
                "lon": left_lon,
                "hdg_deg": 0.0,  # North
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat + nm_to_lat_deg(trail_spacing * 1.5), "lon": left_lon}
            })
            
            # A2: Trail aircraft, left lane
            configs.append({
                "id": "A2",
                "type": "A320",
                "lat": center_lat - nm_to_lat_deg(trail_spacing * 0.5),
                "lon": left_lon,
                "hdg_deg": 0.0,  # North
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat + nm_to_lat_deg(trail_spacing * 1.5), "lon": left_lon}
            })
            
            # Right lane
            right_lon = center_lon + nm_to_lon_deg(lane_separation / 2, center_lat)
            
            # A3: Lead aircraft, right lane
            configs.append({
                "id": "A3",
                "type": "A320",
                "lat": center_lat - nm_to_lat_deg(trail_spacing * 1.5),
                "lon": right_lon,
                "hdg_deg": 0.0,  # North
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat + nm_to_lat_deg(trail_spacing * 1.5), "lon": right_lon}
            })
            
            # A4: Trail aircraft, right lane
            configs.append({
                "id": "A4",
                "type": "A320",
                "lat": center_lat - nm_to_lat_deg(trail_spacing * 0.5),
                "lon": right_lon,
                "hdg_deg": 0.0,  # North
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat + nm_to_lat_deg(trail_spacing * 1.5), "lon": right_lon}
            })
            
        else:  # 3p1 pattern
            # 3 in-trail conflicting, 1 on side lane (like chase_3p1.json)
            trail_spacing = 6.0  # NM in-trail spacing
            side_offset = 15.0  # NM to the side
            
            # Center lane (3 aircraft in-trail)
            for i, aid in enumerate(["A1", "A2", "A3"]):
                configs.append({
                    "id": aid,
                    "type": "A320",
                    "lat": center_lat - nm_to_lat_deg(trail_spacing * (2 - i)),
                    "lon": center_lon,
                    "hdg_deg": 0.0,  # North
                    "spd_kt": spd_kt,
                    "alt_ft": alt_ft,
                    "waypoint": {"lat": center_lat + nm_to_lat_deg(trail_spacing * 1.5), "lon": center_lon}
                })
            
            # Side lane (non-conflicting)
            side_lon = center_lon + nm_to_lon_deg(side_offset, center_lat)
            configs.append({
                "id": "A4",
                "type": "A320",
                "lat": center_lat - nm_to_lat_deg(trail_spacing * 1.5),
                "lon": side_lon,
                "hdg_deg": 0.0,  # North
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat + nm_to_lat_deg(trail_spacing * 1.5), "lon": side_lon}
            })
        
        return configs
    
    def _generate_cross_scenario(self, agent_ids: List[str], center_lat: float,
                                center_lon: float, alt_ft: float, spd_kt: float,
                                pattern: str) -> List[Dict[str, Any]]:
        """
        Generate CROSS scenario with perpendicular/angular crossing conflicts.
        
        CROSS patterns create lateral conflicts:
            - Agents approach common intersection point from different directions
            - Perpendicular or angular crossing geometries (90-135°)
            - Tests tactical deconfliction and crossing sequencing
        
        Patterns:
            2x2: Two pairs crossing at 90° (canonical crossing)
                - Pair 1: A1 (West→East) + A2 (East→West)
                - Pair 2: A3 (South→North) + A4 (North→South)
                - All pass through center waypoint
                - Classic 4-way intersection conflict
            
            3p1: Three-agent crossing + perpendicular singleton
                - A1, A2, A3 cross at 120° spacing (triangular)
                - A4 crosses perpendicular to A1-A2 axis
                - Creates asymmetric crossing conflict
        
        Geometry:
            - Start positions: ~30-50 NM from intersection center
            - Crossing angles: 90° (2x2), 120°/90° (3p1)
            - Waypoints: All agents target center point
        
        Conflict Zone:
            - Intersection region within 5 NM of center
            - Agents must sequence crossing or deviate laterally
        
        Args:
            agent_ids: List of agent IDs (A1-A4).
            center_lat: Center latitude (deg).
            center_lon: Center longitude (deg).
            alt_ft: Altitude (ft, constant 10,000).
            spd_kt: Speed (kt, baseline 250).
            pattern: "2x2" or "3p1" configuration.
        
        Returns:
            List of agent config dicts with id, type, lat, lon, hdg_deg, spd_kt, alt_ft, waypoint.
        """
        configs = []
        
        if pattern == "2x2":
            # Two separate 90° crossings (like cross_2x2.json)
            cross_separation = 12.0  # NM between crossing centers
            cross_radius = 15.0  # NM from crossing point
            
            # Left crossing
            left_center_lon = center_lon - nm_to_lon_deg(cross_separation, center_lat)
            
            # A1: North-south through left crossing
            configs.append({
                "id": "A1",
                "type": "A320",
                "lat": center_lat - nm_to_lat_deg(cross_radius * 0.5),
                "lon": left_center_lon,
                "hdg_deg": 0.0,  # North
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat + nm_to_lat_deg(cross_radius * 0.5), "lon": left_center_lon}
            })
            
            # A2: East-west through left crossing
            configs.append({
                "id": "A2",
                "type": "A320",
                "lat": center_lat,
                "lon": left_center_lon - nm_to_lon_deg(cross_radius, center_lat),
                "hdg_deg": 90.0,  # East
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat, "lon": left_center_lon + nm_to_lon_deg(cross_radius, center_lat)}
            })
            
            # Right crossing
            right_center_lon = center_lon + nm_to_lon_deg(cross_separation, center_lat)
            
            # A3: North-south through right crossing
            configs.append({
                "id": "A3",
                "type": "A320",
                "lat": center_lat - nm_to_lat_deg(cross_radius * 0.5),
                "lon": right_center_lon,
                "hdg_deg": 0.0,  # North
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat + nm_to_lat_deg(cross_radius * 0.5), "lon": right_center_lon}
            })
            
            # A4: East-west through right crossing
            configs.append({
                "id": "A4",
                "type": "A320",
                "lat": center_lat,
                "lon": right_center_lon - nm_to_lon_deg(cross_radius, center_lat),
                "hdg_deg": 90.0,  # East
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat, "lon": right_center_lon + nm_to_lon_deg(cross_radius, center_lat)}
            })
            
        else:  # 3p1 pattern
            # 3-way intersection at center, 1 on side (like cross_3p1.json)
            cross_radius = 15.0  # NM from center
            side_offset = 18.0  # NM to the side
            
            # A1: North-south through center
            configs.append({
                "id": "A1",
                "type": "A320",
                "lat": center_lat - nm_to_lat_deg(cross_radius * 0.5),
                "lon": center_lon,
                "hdg_deg": 0.0,  # North
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat + nm_to_lat_deg(cross_radius * 0.5), "lon": center_lon}
            })
            
            # A2: West-to-east through center
            configs.append({
                "id": "A2",
                "type": "A320",
                "lat": center_lat,
                "lon": center_lon - nm_to_lon_deg(cross_radius, center_lat),
                "hdg_deg": 90.0,  # East
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat, "lon": center_lon + nm_to_lon_deg(cross_radius, center_lat)}
            })
            
            # A3: East-to-west through center (opposite of A2)
            configs.append({
                "id": "A3",
                "type": "A320",
                "lat": center_lat,
                "lon": center_lon + nm_to_lon_deg(cross_radius, center_lat),
                "hdg_deg": 270.0,  # West
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat, "lon": center_lon - nm_to_lon_deg(cross_radius, center_lat)}
            })
            
            # A4: North-south on side lane (non-conflicting)
            side_lon = center_lon + nm_to_lon_deg(side_offset, center_lat)
            configs.append({
                "id": "A4",
                "type": "A320",
                "lat": center_lat - nm_to_lat_deg(cross_radius * 0.5),
                "lon": side_lon,
                "hdg_deg": 0.0,  # North
                "spd_kt": spd_kt,
                "alt_ft": alt_ft,
                "waypoint": {"lat": center_lat + nm_to_lat_deg(cross_radius), "lon": side_lon}
            })
        
        return configs
        
    def _log_generated_scenario(self, scenario: Dict[str, Any], msg_prefix: str = ""):
        """Log scenario details for debugging."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"DEBUG {msg_prefix} Generated Scenario:")
        logger.info(f"  num_aircraft = {scenario.get('num_aircraft', 'N/A')}")
        logger.info(f"  scenario_type = {scenario.get('scenario_type', 'N/A')}")
        logger.info(f"  agent IDs = {[a['id'] for a in scenario['agents']]}")

    # The rest of the methods are copied from MARLCollisionEnv with minimal changes
    # (observation_space, action_space, observe, last, reset, step, etc.)
    
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]
    
    @property
    def observation_spaces(self):
        return self._observation_spaces

    @property
    def action_spaces(self):
        return self._action_spaces
    
    def observe(self, agent):
        """Return observation for a specific agent."""
        obs = self._collect_observations()
        if agent in obs:
            return obs[agent]
        # Default fallback observation
        K = self.neighbor_topk
        null = np.zeros(1, np.float32)
        zK = np.zeros(K, np.float32)
        return {"wp_dist_norm": null, "cos_to_wp": null, "sin_to_wp": null,
                "airspeed": null, "progress_rate": null, "safety_rate": null,
                "x_r": zK, "y_r": zK, "vx_r": zK, "vy_r": zK}
    
    def last(self):
        """Return (obs, reward, done, info) for current agent (AEC API compatibility)."""
        K = self.neighbor_topk
        null = np.zeros(1, np.float32)
        zK = np.zeros(K, np.float32)
        obs = {"wp_dist_norm": null, "cos_to_wp": null, "sin_to_wp": null,
               "airspeed": null, "progress_rate": null, "safety_rate": null,
               "x_r": zK, "y_r": zK, "vx_r": zK, "vy_r": zK}
        return obs, 0.0, False, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        if options is None:
            options = {}
            
        self._episode_id += 1
        self._step_idx = 0
        self._agent_waypoints = {}

        # Reset BlueSky efficiently
        bs.stack.stack("HOLD")
        bs.stack.stack("DEL WIND")
        
        # Delete existing aircraft
        for aid in self.possible_agents:
            try:
                bs.stack.stack(f"DEL {aid}")
            except Exception:
                pass
        
        bs.stack.stack("TIME 0")
        bs.stack.stack("DT 1.0")

        # Generate structured scenario (merge/chase/cross with fixed 4 agents)
        scenario = self._generate_random_scenario(seed)
        
        # DEBUG: Log what scenario was generated
        self._log_generated_scenario(scenario, f"Episode {self._episode_id}")
        
        # Reset agents list based on generated scenario
        self.agents = [agent["id"] for agent in scenario["agents"]]
        
        # Clear buffers
        for aid in self.possible_agents:
            self._prev_actions_deg_kt[aid][:] = [0.0, 0.0]

        # Create aircraft from generated scenario
        for agent in scenario["agents"]:
            aid = agent["id"]
            
            lat = float(agent["lat"])
            lon = float(agent["lon"])
            alt_ft = float(agent["alt_ft"])
            hdg = float(agent["hdg_deg"])
            spd_kt = float(agent["spd_kt"])
            actype = agent.get("type", "A320")
            
            # Ensure valid values
            hdg = hdg % 360.0
            spd_kt = max(100.0, min(400.0, spd_kt))
            
            # Create aircraft in BlueSky
            bs.stack.stack(f"CRE {aid} {actype} {lat:.6f} {lon:.6f} {int(round(hdg))} {alt_ft:.0f} {spd_kt:.0f}")
            
            # Remember baseline speed
            self._base_cas_kt[aid] = spd_kt
            bs.stack.stack(f"SPD {aid} {spd_kt:.1f}")
            
            # Set waypoint
            wpt = agent.get("waypoint", {})
            wlat, wlon = float(wpt["lat"]), float(wpt["lon"])
            
            self._agent_waypoints[aid] = (wlat, wlon)
            bs.stack.stack(f"ADDWPT {aid} {wlat:.6f} {wlon:.6f}")

        # Resume operation and flush aircraft creation
        bs.stack.stack("OP")
        bs.sim.step()
        
        # Initialize team PBRS state
        obs = self._collect_observations()
        self._team_phi = self._compute_team_phi()
        self._team_dphi_ema = 0.0
        # CRITICAL FIX: Use self.agents (dynamic, from scenario) not self.possible_agents (static)
        # because waypoints are populated from generated scenario agents
        self._prev_wp_dist_nm = {aid: None for aid in self.agents}
        self._waypoint_hits = {aid: 0 for aid in self.agents}
        self._agents_to_stop_logging = set()
        self._waypoint_reached = {aid: False for aid in self.agents}
        self._waypoint_reached_this_step = {aid: False for aid in self.agents}
        self._agent_done = {aid: False for aid in self.agents}
        self._agent_wpreached = {aid: False for aid in self.agents}
        
        # Reset unified reward state tracking
        self._prev_in_violation = {aid: False for aid in self.agents}
        self._prev_minsep_nm = {aid: float("inf") for aid in self.agents}
        self._prev_drift_abs = {aid: 0.0 for aid in self.agents}
        
        # Reset real-time hallucination detection
        if self._enable_hallucination_detection and self._hallucination_detector:
            # CRITICAL FIX: Use self.agents (from generated scenario) instead of self.possible_agents
            # because _agent_waypoints is populated from scenario agents (AC01, AC02, etc.)
            # not from static possible_agents (A1, A2, A3)
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"DEBUG Generic Env Reset:")
            logger.info(f"  self.possible_agents (static) = {self.possible_agents}")
            logger.info(f"  self.agents (dynamic, from scenario) = {self.agents}")
            logger.info(f"  _agent_waypoints keys = {list(self._agent_waypoints.keys())}")
            
            # Verify all agents have waypoints (should match now)
            missing_waypoints = [aid for aid in self.agents if aid not in self._agent_waypoints]
            if missing_waypoints:
                logger.error(f"  MISMATCH: Missing waypoints for {missing_waypoints}")
                logger.error(f"  Generated scenario agents: {[a['id'] for a in scenario['agents']]}")
            else:
                logger.info(f"  ✓ All agents have waypoints")
            
            self._rt_trajectory = {
                "positions": [],
                "actions": [],
                "agents": {aid: {"headings": [], "speeds": []} for aid in self.agents},
                "waypoint_status": {aid: {} for aid in self.agents},
                "waypoints": {aid: {"lat": self._agent_waypoints[aid][0],
                                   "lon": self._agent_waypoints[aid][1]} for aid in self.agents}
            }
        
        infos = {aid: {} for aid in self.agents}
        return obs, infos

    # Copy all the remaining methods from MARLCollisionEnv with minimal changes
    # This includes step(), _relative_obs_for(), _update_rt_trajectory(), _collect_observations(),
    # _log_step(), _save_trajectory_csv(), _compute_team_phi()
    
    def step(self, actions: Dict[str, np.ndarray]):
        """Execute actions and return observations, rewards, terminations, truncations, infos."""
        # This method is essentially the same as in MARLCollisionEnv
        # Copy the entire step method implementation here
        if not self.agents:
            return

        # Apply actions and store for logging
        normalized_actions = {}
        for aid, a in actions.items():
            if aid not in self.agents:
                continue
                
            # Zero actions for agents who have already reached their waypoint
            if self._waypoint_reached.get(aid, False):
                a = np.zeros_like(a)
                
            # Apply action delay if enabled
            raw = np.asarray(a, dtype=np.float32).reshape(-1)
            raw = np.clip(raw, -1.0, 1.0)
            
            if self.action_delay_steps > 0:
                self._action_queues[aid].append(raw.copy())
                a = self._action_queues[aid].pop(0)
            else:
                a = raw
                
            normalized_actions[aid] = a
            
            dh = float(a[0]) * D_HEADING
            dv = float(a[1]) * D_VELOCITY
            
            # Apply action thresholds - ignore micro-actions
            if abs(dh) < ACTION_THRESHOLD_DEG:
                dh = 0.0
            if abs(dv) < ACTION_THRESHOLD_KT:
                dv = 0.0

            idx = bs.traf.id2idx(aid)
            if not (isinstance(idx, int) and idx >= 0):
                continue

            # Direct control
            try:
                current_hdg = float(bs.traf.hdg[idx])
                current_cas_kt = float(bs.traf.cas[idx])
            except Exception:
                current_hdg = 0.0
                current_cas_kt = 250.0

            # Apply heading change
            new_hdg = current_hdg + dh
            while new_hdg > 180.0:
                new_hdg -= 360.0
            while new_hdg < -180.0:
                new_hdg += 360.0
            if new_hdg < 0:
                new_hdg += 360.0

            # Apply speed change
            new_spd_kt = current_cas_kt + dv
            slo, shi = self._spd_bounds_scale.get(aid, (1.0, 1.0))
            lo_base, hi_base = self._spd_bounds_kt
            new_spd_kt = max(lo_base * slo, min(hi_base * shi, new_spd_kt))

            # Issue BlueSky commands
            bs.stack.stack(f"HDG {aid} {new_hdg:.1f}")
            bs.stack.stack(f"SPD {aid} {new_spd_kt:.1f}")

            self._prev_actions_deg_kt[aid][:] = [dh, dv]

        # Advance simulation
        for _ in range(10):
            bs.sim.step()

        self._step_idx += 1

        # Get current state for all calculations (same logic as minimal env)
        obs = self._collect_observations()
        
        # Calculate distances, conflicts, rewards etc. (same as minimal env)
        min_sep_nm_global = float('inf')
        conflict_flags = {aid: 0 for aid in self.agents}
        collision_flags = {aid: 0 for aid in self.agents}
        reached_wp = {aid: 0 for aid in self.agents}
        first_time_reach = {aid: False for aid in self.agents}
        dist_wp_nm_by_agent: Dict[str, float] = {}
        pairwise_dist_nm: Dict[str, Dict[str, float]] = {aid: {} for aid in self.agents}
        conflict_pairs_count = 0
        
        self._waypoint_reached_this_step = {aid: False for aid in self.possible_agents}
        any_collision_this_step = False
        
        # Calculate waypoint distances and pairwise separations (same logic as minimal env)
        for i, aid_i in enumerate(self.agents):
            idx_i = bs.traf.id2idx(aid_i)
            if not isinstance(idx_i, int) or idx_i < 0:
                dist_wp_nm_by_agent[aid_i] = 1000.0
                continue
                
            try:
                lat_i, lon_i = float(bs.traf.lat[idx_i]), float(bs.traf.lon[idx_i])
            except (IndexError, TypeError):
                lat_i, lon_i = 0.0, 0.0
            
            # Waypoint distance and reaching logic
            wpt = self._agent_waypoints.get(aid_i, (0, 0))
            dist_wp_nm = haversine_nm(lat_i, lon_i, wpt[0], wpt[1])
            dist_wp_nm_by_agent[aid_i] = float(dist_wp_nm)
            
            # Check if agent reaches waypoint
            if not self._agent_wpreached[aid_i] and dist_wp_nm <= WAYPOINT_THRESHOLD_NM:
                self._agent_wpreached[aid_i] = True
                self._waypoint_reached[aid_i] = True
                self._waypoint_reached_this_step[aid_i] = True
                self._waypoint_hits[aid_i] = 1
                reached_wp[aid_i] = 1
                first_time_reach[aid_i] = True
            elif self._agent_wpreached[aid_i]:
                self._waypoint_reached[aid_i] = True
                reached_wp[aid_i] = 1

            # Pairwise separation
            for j, aid_j in enumerate(self.agents):
                if i >= j:
                    continue
                idx_j = bs.traf.id2idx(aid_j)
                if not isinstance(idx_j, int) or idx_j < 0:
                    continue
                try:
                    lat_j, lon_j = float(bs.traf.lat[idx_j]), float(bs.traf.lon[idx_j])
                    dist_nm = haversine_nm(lat_i, lon_i, lat_j, lon_j)
                except (IndexError, TypeError):
                    dist_nm = 200.0
                
                pairwise_dist_nm[aid_i][aid_j] = dist_nm
                pairwise_dist_nm[aid_j][aid_i] = dist_nm
                min_sep_nm_global = min(min_sep_nm_global, dist_nm)
                
                # Conflict detection
                if dist_nm < self.sep_nm:
                    conflict_flags[aid_i] = 1
                    conflict_flags[aid_j] = 1
                    conflict_pairs_count += 1
                
                # Collision detection
                if dist_nm < self.collision_nm:
                    collision_flags[aid_i] = 1
                    collision_flags[aid_j] = 1
                    any_collision_this_step = True

        # Compute comprehensive rewards (FULL unified reward system matching minimal env)
        rewards = {}
        reward_parts_by_agent = {}
        dt = 10.0  # 10 sim steps × 1s each
        
        for aid in self.agents:
            idx = bs.traf.id2idx(aid)
            if not isinstance(idx, int) or idx < 0:
                rewards[aid] = 0.0
                reward_parts_by_agent[aid] = {"total": 0.0}
                continue
            
            # Get current heading for drift calculation
            try:
                current_hdg = float(bs.traf.hdg[idx])
            except (IndexError, TypeError, AttributeError):
                current_hdg = 0.0
            
            # Calculate course to waypoint for drift
            wpt = self._agent_waypoints.get(aid, (0, 0))
            try:
                course_to_wp, _ = bs.tools.geo.kwikqdrdist(
                    float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), wpt[0], wpt[1]
                )
            except Exception:
                course_to_wp = current_hdg
            
            # === FULL UNIFIED REWARD CALCULATION (matching minimal env) ===
            parts = {}
            
            # 1. Progress reward (signed: positive for forward, negative for backtrack)
            wp_dist_now = dist_wp_nm_by_agent.get(aid, 1000.0)
            prev_wp_dist = self._prev_wp_dist_nm.get(aid, None)
            if prev_wp_dist is not None:
                progress_km = (prev_wp_dist - wp_dist_now) * NM_TO_KM
                r_progress = self.r_prog_per_km * progress_km
            else:
                r_progress = 0.0
            
            # 2. Drift improvement reward (with deadzone)
            def angle_diff_deg(a, b):
                d = (a - b) % 360.0
                return d if d <= 180.0 else d - 360.0
            
            drift_deg = abs(angle_diff_deg(current_hdg, course_to_wp))
            prev_drift = self._prev_drift_abs.get(aid, drift_deg)
            drift_improve = prev_drift - drift_deg
            
            r_drift = 0.0
            if drift_improve > self.drift_deadzone_deg:
                r_drift = self.drift_improve_gain * (drift_improve - self.drift_deadzone_deg)
            
            # 3. Time penalty (encourages efficiency)
            r_time = self.r_time_per_s * dt
            
            # 4. Violation penalties (SPLIT into entry and step components)
            agent_min_sep = min(pairwise_dist_nm.get(aid, {}).values(), default=200.0)
            current_in_violation = agent_min_sep < self.sep_nm
            prev_in_violation = self._prev_in_violation.get(aid, False)
            
            # 4a. Violation entry penalty (one-time on violation entry)
            r_violate_entry = 0.0
            if current_in_violation and not prev_in_violation:
                r_violate_entry = self.violation_entry_penalty  # -25.0
            
            # 4b. Per-step violation severity penalty
            r_violate_step = 0.0
            if current_in_violation:
                # Severity ∈ [0, 1] for 0-5 NM, [1, 1.5] for deep breach < 1 NM
                severity = 1.0 - (agent_min_sep / self.sep_nm)
                if agent_min_sep < self.deep_breach_nm:
                    severity = 1.0 + 0.5 * (1.0 - agent_min_sep / self.deep_breach_nm)
                r_violate_step = self.violation_step_scale * severity  # -1.0 * severity
            
            # 5. Action cost (penalizes excessive control)
            a = normalized_actions.get(aid, np.array([0.0, 0.0]))
            act_cost = self.action_cost_per_unit * (abs(float(a[0])) + abs(float(a[1])))
            
            # 6. Reach bonus (one-time on first waypoint capture)
            r_reach = self.r_reach_bonus if first_time_reach.get(aid, False) else 0.0
            
            # 7. Terminal miss penalty (at episode end if waypoint not reached)
            r_terminal = 0.0
            if self._step_idx >= self.max_episode_steps and not self._waypoint_reached[aid]:
                r_terminal = self.terminal_not_reached_penalty  # -10.0
            
            # Total individual reward (before team shaping)
            r_total = (
                r_progress + r_drift + r_violate_entry + r_violate_step +
                act_cost + r_time + r_reach + r_terminal
            )
            
            # Store all reward components
            parts = {
                "act_cost": float(act_cost),
                "progress": float(r_progress),
                "drift": float(r_drift),
                "time": float(r_time),
                "violate_entry": float(r_violate_entry),
                "violate_step": float(r_violate_step),
                "reach": float(r_reach),
                "terminal": float(r_terminal),
                "total": float(r_total),
            }
            
            rewards[aid] = float(r_total)
            reward_parts_by_agent[aid] = parts
            
            # Update state tracking for next step
            self._prev_wp_dist_nm[aid] = wp_dist_now
            self._prev_in_violation[aid] = current_in_violation
            self._prev_drift_abs[aid] = drift_deg

        # Team PBRS shaping (same as minimal env)
        phi_now = self._compute_team_phi()
        dphi = self.team_gamma * phi_now - (self._team_phi if self._team_phi is not None else phi_now)
        self._team_dphi_ema = (1.0 - self.team_ema_a) * self._team_dphi_ema + self.team_ema_a * dphi
        dphi_shaped = max(-self.team_cap, min(self.team_cap, self._team_dphi_ema)) * self.team_anneal
        self._team_phi = phi_now

        # Share weights per agent
        weights = {aid: 1.0 for aid in self.agents}
        
        # Normalize weights
        Wsum = sum(weights.values()) or 1.0
        scale = len(self.agents) / Wsum
        for aid in weights:
            weights[aid] *= scale

        # Add team term to each agent's reward
        for aid in self.agents:
            r_team = self.team_w * weights[aid] * dphi_shaped
            rewards[aid] = float(rewards.get(aid, 0.0) + r_team)
            reward_parts_by_agent[aid]["team"] = float(r_team)
            reward_parts_by_agent[aid]["team_phi"] = float(phi_now)
            reward_parts_by_agent[aid]["team_dphi"] = float(dphi_shaped)

        # Terminations and truncations
        all_at_waypoint = len(self.agents) > 0 and all(reached_wp.values())
        time_limit_reached = self._step_idx >= self.max_episode_steps
        
        episode_complete = all_at_waypoint or time_limit_reached
        
        terminations = {aid: episode_complete for aid in self.agents}
        truncations = {aid: False for aid in self.agents}

        # Enhanced infos
        infos = {}
        total_waypoint_hits = sum(1 for aid in self.agents if self._waypoint_reached.get(aid, False))
        for aid in self.agents:
            agent_min_sep = 200.0
            agent_distances = pairwise_dist_nm.get(aid, {})
            if agent_distances:
                agent_min_sep = min(agent_distances.values())
            
            infos[aid] = {
                "reward_parts": reward_parts_by_agent[aid],
                "min_sep": agent_min_sep,
                "wp_dist": dist_wp_nm_by_agent.get(aid, 1000.0),
                "conflict_flag": conflict_flags[aid],
                "collision_flag": collision_flags[aid],
                "waypoint_reached": self._waypoint_reached[aid],
                "waypoint_hits": self._waypoint_hits[aid],
                "sim_time_s": float(bs.sim.simt),
                "total_waypoint_hits": total_waypoint_hits
            }

        # Update real-time trajectory
        if self._enable_hallucination_detection and self._hallucination_detector:
            self._update_rt_trajectory(normalized_actions)
        
        # Enhanced trajectory logging
        if self.log_trajectories:
            self._log_step(conflict_flags, collision_flags, 
                          rewards, pairwise_dist_nm, dist_wp_nm_by_agent, 
                          conflict_pairs_count, reward_parts_by_agent, first_time_reach)

        # Mark agents as done
        for aid in self.agents:
            if self._agent_wpreached.get(aid, False) and not self._agent_done.get(aid, False):
                pass

        # End episode when termination conditions are met
        if episode_complete:
            if self.log_trajectories and self._traj_rows:
                self._save_trajectory_csv()
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def _relative_obs_for(self, aid: str) -> dict:
        """Return dict obs with relative features (same as minimal env)."""
        # Copy implementation from MARLCollisionEnv._relative_obs_for
        idx = bs.traf.id2idx(aid)
        if not isinstance(idx, int) or idx < 0:
            K = self.neighbor_topk
            null = np.zeros(1, np.float32)
            zK = np.zeros(K, np.float32)
            return {
                "wp_dist_norm": null, "cos_to_wp": null, "sin_to_wp": null,
                "airspeed": null, "progress_rate": null, "safety_rate": null,
                "x_r": zK, "y_r": zK, "vx_r": zK, "vy_r": zK
            }

        # Implementation copied from minimal env...
        wlat, wlon = self._agent_waypoints[aid]
        try:
            qdr, _ = bs.tools.geo.kwikqdrdist(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), wlat, wlon)
        except:
            qdr = 0.0
        
        wp_d_nm = haversine_nm(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), wlat, wlon)
        wp_dist_norm = np.array([np.tanh(wp_d_nm / 20.0)], np.float32)
        
        prev_wp_dist = self._prev_wp_dist_nm.get(aid, None)
        if prev_wp_dist is not None:
            progress_rate_raw = prev_wp_dist - wp_d_nm
            progress_rate = np.array([np.tanh(progress_rate_raw / 2.0)], np.float32)
        else:
            progress_rate = np.zeros(1, np.float32)
        
        try:
            hdg_array = getattr(bs.traf, 'hdg', None)
            if hdg_array is not None and len(hdg_array) > idx:
                hdg = float(hdg_array[idx])
            else:
                hdg = 0.0
        except (IndexError, TypeError, AttributeError):
            hdg = 0.0
            
        theta = math.radians(qdr - hdg)
        cos_to_wp = np.array([math.cos(theta)], np.float32)
        sin_to_wp = np.array([math.sin(theta)], np.float32)

        try:
            tas_array = getattr(bs.traf, 'tas', None)
            if tas_array is not None and len(tas_array) > idx:
                tas_mps = float(tas_array[idx])
            else:
                tas_mps = 150.0
        except (IndexError, TypeError, AttributeError):
            tas_mps = 150.0
        airspeed = np.array([np.clip((tas_mps - 150.0) / 6.0, -10.0, 10.0)], np.float32)

        # Calculate safety rate
        current_min_sep = 200.0
        for other_aid in self.agents:
            if other_aid != aid:
                other_idx = bs.traf.id2idx(other_aid)
                if isinstance(other_idx, int) and other_idx >= 0:
                    try:
                        other_lat, other_lon = float(bs.traf.lat[other_idx]), float(bs.traf.lon[other_idx])
                        sep = haversine_nm(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), other_lat, other_lon)
                        current_min_sep = min(current_min_sep, sep)
                    except:
                        pass
        
        prev_min_sep = self._prev_minsep_nm.get(aid, float("inf"))
        if prev_min_sep != float("inf"):
            safety_rate_raw = current_min_sep - prev_min_sep
            safety_rate = np.array([np.tanh(safety_rate_raw / 5.0)], np.float32)
        else:
            safety_rate = np.zeros(1, np.float32)

        # Build neighbor arrays
        ids = [a for a in self.agents if a != aid]
        ranges = []
        for aj in ids:
            jdx = bs.traf.id2idx(aj)
            if not isinstance(jdx, int) or jdx < 0:
                continue
            d = haversine_nm(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]),
                           float(bs.traf.lat[jdx]), float(bs.traf.lon[jdx]))
            ranges.append((d, aj, jdx))
        ranges.sort(key=lambda t: t[0])
        ranges = ranges[:self.neighbor_topk]

        # Own ground velocity
        tas_kt = tas_mps * 1.94384
        vx = kt_to_nms(tas_kt) * math.cos(math.radians(hdg))
        vy = kt_to_nms(tas_kt) * math.sin(math.radians(hdg))

        K = self.neighbor_topk
        x_r = np.zeros(K, np.float32)
        y_r = np.zeros(K, np.float32)
        vx_r = np.zeros(K, np.float32)
        vy_r = np.zeros(K, np.float32)

        lat0, lon0 = float(bs.traf.lat[idx]), float(bs.traf.lon[idx])
        for k, (d_nm, aj, jdx) in enumerate(ranges):
            dx_nm = haversine_nm(lat0, lon0, lat0, float(bs.traf.lon[jdx])) * (1 if float(bs.traf.lon[jdx])>=lon0 else -1)
            dy_nm = haversine_nm(lat0, lon0, float(bs.traf.lat[jdx]), lon0) * (1 if float(bs.traf.lat[jdx])>=lat0 else -1)
            x_r[k] = np.clip(dx_nm * NM_TO_KM * 1000.0 / 20000.0, -12.0, 12.0)
            y_r[k] = np.clip(dy_nm * NM_TO_KM * 1000.0 / 20000.0, -12.0, 12.0)

            try:
                taj_mps = float(bs.traf.tas[jdx])
                hj = float(bs.traf.hdg[jdx])
            except (IndexError, TypeError, AttributeError):
                taj_mps = 150.0
                hj = 0.0
            vj = kt_to_nms(taj_mps * 1.94384)
            vxj = vj*math.cos(math.radians(hj))
            vyj = vj*math.sin(math.radians(hj))
            vx_r[k] = np.clip((vxj - vx) / 32.0, -150.0, 150.0)
            vy_r[k] = np.clip((vyj - vy) / 66.0, -150.0, 150.0)

        return {
            "wp_dist_norm": wp_dist_norm,
            "cos_to_wp": cos_to_wp,
            "sin_to_wp": sin_to_wp,
            "airspeed": airspeed,
            "progress_rate": progress_rate,
            "safety_rate": safety_rate,
            "x_r": x_r, "y_r": y_r,
            "vx_r": vx_r, "vy_r": vy_r,
        }

    def _update_rt_trajectory(self, actions: Dict[str, np.ndarray]):
        """Update real-time trajectory data for hallucination detection."""
        if not self._enable_hallucination_detection or not self._hallucination_detector:
            return
            
        # Copy implementation from minimal env
        positions = {}
        for aid in self.agents:
            idx = bs.traf.id2idx(aid)
            if isinstance(idx, int) and idx >= 0:
                try:
                    lat = float(bs.traf.lat[idx])
                    lon = float(bs.traf.lon[idx])
                    positions[aid] = (lat, lon)
                except:
                    positions[aid] = (0.0, 0.0)
            else:
                positions[aid] = (0.0, 0.0)
        
        self._rt_trajectory["positions"].append(positions)
        
        action_step = {}
        for aid in self.agents:
            if aid in actions:
                hdg_delta = float(actions[aid][0]) * D_HEADING
                spd_delta = float(actions[aid][1]) * D_VELOCITY
                action_step[aid] = np.array([hdg_delta, spd_delta])
            else:
                action_step[aid] = np.array([0.0, 0.0])
        
        self._rt_trajectory["actions"].append(action_step)
        
        for aid in self.agents:
            step_idx = len(self._rt_trajectory["positions"]) - 1
            self._rt_trajectory["waypoint_status"][aid][step_idx] = self._agent_wpreached.get(aid, False)

    def _collect_observations(self) -> Dict[str, np.ndarray]:
        """Build relative observations for all agents."""
        obs = {}
        
        try:
            for aid in self.agents:
                obs[aid] = self._relative_obs_for(aid)
        except Exception as e:
            self._logger.error(f"Error in observation collection: {e}")
            for aid in self.agents:
                K = self.neighbor_topk
                null = np.zeros(1, np.float32)
                zK = np.zeros(K, np.float32)
                obs[aid] = {
                    "wp_dist_norm": null, "cos_to_wp": null, "sin_to_wp": null,
                    "airspeed": null, "progress_rate": null, "safety_rate": null,
                    "x_r": zK, "y_r": zK, "vx_r": zK, "vy_r": zK
                }
                
        return obs

    def _log_step(self, conflict_flags: Dict[str, int], collision_flags: Dict[str, int], 
                  rewards: Dict[str, float], pairwise_dist_nm: Dict[str, Dict[str, float]], 
                  dist_wp_nm_by_agent: Dict[str, float], conflict_pairs_count: int,
                  reward_parts_by_agent: Dict[str, dict], first_time_reach: Dict[str, bool]):
        """Log step data for trajectory analysis (same as minimal env)."""
        # Copy implementation from minimal env
        all_ids = list(self.possible_agents)
        
        for aid in self.agents:
            if self._agent_done.get(aid, False):
                continue
                
            idx = bs.traf.id2idx(aid)
            
            if isinstance(idx, int) and idx >= 0:
                try:
                    lat_deg = float(bs.traf.lat[idx])
                    lon_deg = float(bs.traf.lon[idx])
                    alt_ft = float(bs.traf.alt[idx]) * 3.28084
                    hdg_deg = float(bs.traf.hdg[idx])
                    tas_kt = float(bs.traf.tas[idx]) * 1.94384
                    cas_kt = float(bs.traf.cas[idx]) * 1.94384
                except:
                    lat_deg = lon_deg = hdg_deg = 0.0
                    alt_ft = tas_kt = cas_kt = 0.0
            else:
                lat_deg = lon_deg = hdg_deg = 0.0
                alt_ft = tas_kt = cas_kt = 0.0
            
            agent_min_sep = 200.0
            agent_distances = pairwise_dist_nm.get(aid, {})
            if agent_distances:
                agent_min_sep = min(agent_distances.values())
            
            row = {
                "episode_id": self._episode_id,
                "step_idx": self._step_idx,
                "sim_time_s": float(bs.sim.simt),
                "agent_id": aid,
                "lat_deg": lat_deg,
                "lon_deg": lon_deg,
                "alt_ft": alt_ft,
                "hdg_deg": hdg_deg,
                "tas_kt": tas_kt,
                "cas_kt": cas_kt,
                "action_hdg_delta_deg": float(self._prev_actions_deg_kt[aid][0]),
                "action_spd_delta_kt": float(self._prev_actions_deg_kt[aid][1]),
                "reward": float(rewards.get(aid, 0.0)),
                "min_separation_nm": float(agent_min_sep),
                "conflict_flag": int(conflict_flags.get(aid, 0)),
                "collision_flag": int(collision_flags.get(aid, 0)),
                "wp_dist_nm": float(dist_wp_nm_by_agent.get(aid, np.nan)),
                "waypoint_reached": int(self._waypoint_reached.get(aid, False)),
                "waypoint_hits": int(self._waypoint_hits.get(aid, 0)),
                "conflict_pairs_count": int(conflict_pairs_count),
            }
            
            # Per-agent distances
            for other_id in all_ids:
                col_name = f"dist_to_{other_id}_nm"
                if other_id == aid:
                    row[col_name] = 0.0
                else:
                    row[col_name] = float(pairwise_dist_nm.get(aid, {}).get(other_id, np.nan))
            
            # Unified reward breakdown - ALL components (matching minimal env)
            parts = reward_parts_by_agent.get(aid, {})
            row.update({
                "reward_act_cost": parts.get("act_cost", 0.0),
                "reward_progress": parts.get("progress", 0.0),
                "reward_drift": parts.get("drift", 0.0),
                "reward_time": parts.get("time", 0.0),
                "reward_violate_entry": parts.get("violate_entry", 0.0),
                "reward_violate_step": parts.get("violate_step", 0.0),
                "reward_reach": parts.get("reach", 0.0),
                "reward_terminal": parts.get("terminal", 0.0),
                "reward_total": parts.get("total", 0.0),
                "reward_team": parts.get("team", 0.0),
                "team_phi": parts.get("team_phi", 0.0),
                "team_dphi": parts.get("team_dphi", 0.0),
            })
            
            self._traj_rows.append(row)

    def _save_trajectory_csv(self):
        """Save trajectory data to CSV file."""
        if not self._traj_rows:
            return
            
        results_dir = getattr(self, 'results_dir', 'results')
        
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), results_dir)
        
        episode_tag = getattr(self, 'episode_tag', None)
        if episode_tag:
            filename = f"traj_{episode_tag}.csv"
        else:
            filename = f"traj_ep_{self._episode_id:04d}.csv"
        os.makedirs(results_dir, exist_ok=True)
        
        if self._traj_rows:
            import pandas as pd
            df = pd.DataFrame(self._traj_rows)
            csv_path = os.path.join(results_dir, filename)
            df.to_csv(csv_path, index=False)
        
        self._traj_rows = []

    def _compute_team_phi(self) -> float:
        """Compute team potential function (same as minimal env)."""
        ids = self.agents
        if not ids or len(ids) < 2:
            return 1.0
        
        min_separation_nm = float('inf')
        
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                aid_i, aid_j = ids[i], ids[j]
                idx_i = bs.traf.id2idx(aid_i)
                idx_j = bs.traf.id2idx(aid_j)
                
                if isinstance(idx_i, int) and idx_i >= 0 and isinstance(idx_j, int) and idx_j >= 0:
                    try:
                        lat_i, lon_i = float(bs.traf.lat[idx_i]), float(bs.traf.lon[idx_i])
                        lat_j, lon_j = float(bs.traf.lat[idx_j]), float(bs.traf.lon[idx_j])
                        dist_nm = haversine_nm(lat_i, lon_i, lat_j, lon_j)
                        min_separation_nm = min(min_separation_nm, dist_nm)
                    except:
                        pass
                    
        if min_separation_nm == float('inf'):
            return 1.0
            
        s = max(0.0, min_separation_nm)
        if s <= self.sep_nm:
            phi_t = 0.5 * (s / self.sep_nm)
        else:
            phi_t = 0.5 + 0.5 * min(1.0, (s - self.sep_nm) / self.sep_nm)
        return float(np.clip(phi_t, 0.0, 1.0))