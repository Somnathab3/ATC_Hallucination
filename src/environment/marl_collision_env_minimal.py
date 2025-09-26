# Minimal Multi-Agent Reinforcement Learning Environment for Air Traffic Collision Avoidance`

import os
import csv
import json
import math
import atexit
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

# Suppress BlueSky logs immediately
for name in ("bluesky", "bluesky.navdatabase", "bluesky.simulation", "bluesky.traffic"):
    logging.getLogger(name).setLevel(logging.ERROR)

import gymnasium
from gymnasium import Env, spaces
from gymnasium.spaces import Box, Dict as DictSpace
from pettingzoo.utils import ParallelEnv
import bluesky as bs


# Action scaling constants (similar to reference method)
D_HEADING = 18.0  # degrees per unit action  
D_VELOCITY = 10.0  # knots per unit action

# ---- Team coordination reward defaults (PBRS-style) ----
DEFAULT_TEAM_COORDINATION_WEIGHT      = 0.2
DEFAULT_TEAM_GAMMA                    = 0.99
DEFAULT_TEAM_SHARE_MODE               = "responsibility"   # "even" | "responsibility" | "neighbor"
DEFAULT_TEAM_EMA                      = 0.001
DEFAULT_TEAM_CAP                      = 0.01
DEFAULT_TEAM_ANNEAL                   = 1.0
DEFAULT_TEAM_NEIGHBOR_THRESHOLD_KM    = 10.0

# ---- Individual reward defaults (rebalanced) ----
DEFAULT_DRIFT_PENALTY_PER_SEC         = -0.1
DEFAULT_PROGRESS_REWARD_PER_KM        =  0.02
DEFAULT_BACKTRACK_PENALTY_PER_KM      = -0.1
DEFAULT_TIME_PENALTY_PER_SEC          = -0.0005
DEFAULT_REACH_REWARD                  = 10.0      # meaningful waypoint completion
DEFAULT_INTRUSION_PENALTY             = -50.0
DEFAULT_CONFLICT_DWELL_PENALTY_PER_SEC= -0.1

# Geometry/normalization constants
NM_TO_KM = 1.852
DRIFT_NORM_DEN = 180.0                 # scale |drift| (deg) to [0..1]
WAYPOINT_THRESHOLD_NM = 1.0            # Distance threshold for waypoint completion (changed from 5.0 to 1.0)


def kt_to_nms(kt):
    """Convert knots to nautical miles per second."""
    return kt / 3600.0


# BlueSky initialization flag (once per process)
_BS_READY = False


# Add proper BlueSky cleanup on exit to prevent warnings
@atexit.register
def _clean_bs():
    try:
        bs.sim.reset()
    except Exception:
        pass


def nm_to_lat_deg(nm: float) -> float:
    return nm / 60.0


def nm_to_lon_deg(nm: float, lat_deg: float) -> float:
    return nm / (60.0 * max(1e-6, math.cos(math.radians(lat_deg))))


def heading_to_unit(heading_deg: float):
    """Return unit vector (east, north) for given heading (degrees, 0=N, 90=E in BlueSky)."""
    rad = math.radians(90.0 - heading_deg)
    return (math.cos(rad), math.sin(rad))


def haversine_nm(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    """Compute great-circle distance in nautical miles between two lat/lon points."""
    from math import radians, sin, cos, sqrt, atan2
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1_deg, lon1_deg, lat2_deg, lon2_deg])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    R_nm = 3440.065  # Earth radius in nautical miles
    return R_nm * c


class MARLCollisionEnv(ParallelEnv):
    metadata = {"name": "marl_collision_env", "render_modes": []}
    
    @property 
    def max_num_agents(self):
        return len(self.possible_agents)
    
    @property
    def num_agents(self):
        return len(self.agents)

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        global _BS_READY
        
        if env_config is None:
            env_config = {}
        
        self.scenario_path = env_config.get("scenario_path", None)
        if not self.scenario_path or not os.path.exists(self.scenario_path):
            raise FileNotFoundError(f"scenario_path not found: {self.scenario_path}")
        
        self.max_episode_steps = int(env_config.get("max_episode_steps", 100))
        
        # Init BlueSky once per process (worker-safe with better error handling)
        if not _BS_READY:
            import sys
            import time
            from io import StringIO
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            try:
                # Add small delay to avoid simultaneous init conflicts
                time.sleep(0.1)
                bs.init(mode='sim', detached=True)
                bs.stack.stack("DT 1.0")
                bs.stack.stack("CASMACHTHR 0")  # interpret SPD inputs as CAS knots only
                _BS_READY = True
                print(f"BlueSky initialized successfully for process {os.getpid()}")
            except Exception as e:
                print(f"BlueSky initialization failed: {e}")
                raise RuntimeError(f"Failed to initialize BlueSky: {e}")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        # Load scenario FIRST to get possible_agents
        with open(self.scenario_path, "r", encoding="utf-8") as f:
            self._scenario_template = json.load(f)

        self.possible_agents = [a["id"] for a in self._scenario_template["agents"]]
        self.agents = []

        # Action delay support for shift testing
        self.action_delay_steps = int(env_config.get("action_delay_steps", 0))
        self._action_queues = {aid: [[0.0, 0.0]] * self.action_delay_steps
                               for aid in self.possible_agents}

        # Internal state and configuration (BEFORE spaces)
        self._prev_actions_deg_kt = {aid: np.zeros(2, dtype=np.float32) for aid in self.possible_agents}
        self._step_idx = 0
        self._episode_id = 0
        self._agent_waypoints = {}  # Store waypoints for each agent
        self._traj_rows = []  # Enhanced trajectory logging
        self._separation_threshold_nm = float(env_config.get("separation_nm", 5.0))  # NM for conflict detection
        self.log_trajectories = env_config.get("log_trajectories", True)
        self.episode_tag = env_config.get("episode_tag", None)  # Optional episode tag for CSV naming
        
        # Observation mode configuration (BEFORE space creation)
        self.obs_mode          = str(env_config.get("obs_mode", "relative")).lower()  # "relative" | "absolute"
        self.neighbor_topk     = int(env_config.get("neighbor_topk", 3))
        
        # Team coordination (PBRS) configuration
        self.team_w         = float(env_config.get("team_coordination_weight", DEFAULT_TEAM_COORDINATION_WEIGHT))
        self.team_gamma     = float(env_config.get("team_gamma", DEFAULT_TEAM_GAMMA))
        self.team_share     = str(env_config.get("team_share_mode", DEFAULT_TEAM_SHARE_MODE)).lower()
        self.team_ema_a     = float(env_config.get("team_ema", DEFAULT_TEAM_EMA))
        self.team_cap       = float(env_config.get("team_cap", DEFAULT_TEAM_CAP))
        self.team_anneal    = float(env_config.get("team_anneal", DEFAULT_TEAM_ANNEAL))
        self.team_nb_km     = float(env_config.get("team_neighbor_threshold_km", DEFAULT_TEAM_NEIGHBOR_THRESHOLD_KM))
        self._team_phi      = None
        self._team_dphi_ema = 0.0
        
        # Individual reward configuration (allow override via env_config)
        self.r_drift_per_s     = float(env_config.get("drift_penalty_per_sec", DEFAULT_DRIFT_PENALTY_PER_SEC))
        self.r_prog_per_km     = float(env_config.get("progress_reward_per_km", DEFAULT_PROGRESS_REWARD_PER_KM))
        self.r_back_per_km     = float(env_config.get("backtrack_penalty_per_km", DEFAULT_BACKTRACK_PENALTY_PER_KM))
        self.r_time_per_s      = float(env_config.get("time_penalty_per_sec", DEFAULT_TIME_PENALTY_PER_SEC))
        self.r_reach_bonus     = float(env_config.get("reach_reward", DEFAULT_REACH_REWARD))
        self.r_intrusion       = float(env_config.get("intrusion_penalty", DEFAULT_INTRUSION_PENALTY))
        self.r_dwell_per_s     = float(env_config.get("conflict_dwell_penalty_per_sec", DEFAULT_CONFLICT_DWELL_PENALTY_PER_SEC))
        self.action_cost_per_unit = float(env_config.get("action_cost_per_unit", -0.01))
        self.terminal_not_reached_penalty = float(env_config.get("terminal_not_reached_penalty", -10.0))
        
        # Per-episode memory
        self._prev_wp_dist_nm: Dict[str, Optional[float]] = {aid: None for aid in self.possible_agents}
        self._conflict_on_prev = {aid: 0 for aid in self.possible_agents}
        self._waypoint_hits = {aid: 0 for aid in self.possible_agents}  # Track cumulative waypoint completions
        self._agents_to_stop_logging = set()  # Track agents that should no longer be logged
        self._waypoint_reached = {aid: False for aid in self.possible_agents}  # Track if agent has reached waypoint this episode
        self._waypoint_reached_this_step = {aid: False for aid in self.possible_agents}  # Track if agent reached waypoint THIS step
        self._agent_done = {aid: False for aid in self.possible_agents}  # Track if agent is completely done
        self._agent_wpreached = {aid: False for aid in self.possible_agents}  # Persistent waypoint reached flag

        # Baselines captured from the scenario (per agent)
        self._base_cas_kt = {aid: 250.0 for aid in self.possible_agents}  # overwritten on reset
        self._spd_bounds_kt = (100.0, 400.0)  # more reasonable CAS window for training
        # Action scaling now handled by constants D_HEADING and D_VELOCITY
        
        # Collision detection settings (used for rewards but not termination)
        self.collision_nm = float(env_config.get("collision_nm", 1.0))
        # Note: collision_debounce_steps removed since no collision-based termination

        # Set observation spaces based on mode
        if self.obs_mode == "relative":
            K = self.neighbor_topk
            self._observation_spaces = {
                aid: spaces.Dict({
                    # Goal signals
                    "wp_dist_norm": spaces.Box(-1.0, 1.0, (1,), np.float32),
                    "cos_to_wp":    spaces.Box(-1.0, 1.0, (1,), np.float32),
                    "sin_to_wp":    spaces.Box(-1.0, 1.0, (1,), np.float32),

                    # Ownship
                    "airspeed":     spaces.Box(-np.inf, np.inf, (1,), np.float32),

                    # Neighbors (top-K): relative position & velocity
                    "x_r":          spaces.Box(-np.inf, np.inf, (K,), np.float32),
                    "y_r":          spaces.Box(-np.inf, np.inf, (K,), np.float32),
                    "vx_r":         spaces.Box(-np.inf, np.inf, (K,), np.float32),
                    "vy_r":         spaces.Box(-np.inf, np.inf, (K,), np.float32),
                }) for aid in self.possible_agents
            }
        else:
            # Legacy absolute observation space
            low  = np.array([-90, -180,   0,   0,     0,   0, 0], dtype=np.float32)
            high = np.array([ 90,  180, 360, 600, 60000, 200, 1], dtype=np.float32)
            obs_dim = 7
            self._observation_spaces = {aid: spaces.Box(
                low=low, high=high, shape=(obs_dim,), dtype=np.float32
            ) for aid in self.possible_agents}

        self._action_spaces = {aid: spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        ) for aid in self.possible_agents}

        # Set results directory (can be overridden by trainer)
        self.results_dir = env_config.get("results_dir", "results")
        
        # Initialize real-time hallucination detection
        self._enable_hallucination_detection = env_config.get("enable_hallucination_detection", True)
        if self._enable_hallucination_detection:
            # Import here to avoid circular imports
            try:
                from src.analysis.hallucination_detector_enhanced import HallucinationDetector
                self._hallucination_detector = HallucinationDetector(
                    action_period_s=10.0,  # Match environment timestep
                    res_window_s=60.0,
                    horizon_s=300.0,
                    action_thresh=(3.0, 5.0)  # deg, kt thresholds
                )
                # Store trajectory data for real-time analysis
                self._rt_trajectory = {
                    "positions": [],
                    "actions": [],
                    "agents": {aid: {"headings": [], "speeds": []} for aid in self.possible_agents},
                    "waypoint_status": {aid: {} for aid in self.possible_agents}
                }
            except ImportError:
                self._enable_hallucination_detection = False
                self._hallucination_detector = None
        else:
            self._hallucination_detector = None

        # Required PettingZoo attributes
        self.agent_selection = None
        
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO)

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]
    
    # Required PettingZoo ParallelEnv properties for RLlib
    @property
    def observation_spaces(self):
        return self._observation_spaces

    @property
    def action_spaces(self):
        return self._action_spaces
    
    def observe(self, agent):
        """Return observation for a specific agent (used by wrapper)."""
        obs = self._collect_observations()
        if agent in obs:
            return obs[agent]
        if self.obs_mode == "relative":
            K = self.neighbor_topk
            null = np.zeros(1, np.float32); zK = np.zeros(K, np.float32)
            return {"wp_dist_norm": null, "cos_to_wp": null, "sin_to_wp": null,
                    "airspeed": null, "x_r": zK, "y_r": zK, "vx_r": zK, "vy_r": zK}
        # absolute fallback
        return np.array([0, 0, 0, 100, 10000, 100, 0], dtype=np.float32)
    
    def last(self):
        """Return (obs, reward, done, info) for current agent (AEC API compatibility)."""
        return np.array([0, 0, 0, 100, 10000, 100, 0], dtype=np.float32), 0.0, False, {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            pass  # deterministic by design
        if options is None:
            options = {}
            
        # Extract shift parameters from options
        shift = options.get("shift", {})
        targeted_shift = options.get("targeted_shift", {})
        
        # Legacy unison shift support
        pos_nm = float(shift.get("position_nm_delta", 0.0))
        hdg_dg = float(shift.get("heading_deg_delta", 0.0))
        spd_dkt = float(shift.get("speed_kt_delta", 0.0))
            
        self._episode_id += 1
        self._step_idx = 0
        self._agent_waypoints = {}

        # Reset BlueSky efficiently (avoid full RESET which reloads nav DB)
        bs.stack.stack("HOLD")       # pause simulation
        # Delete existing aircraft
        for aid in self.possible_agents:
            try:
                bs.stack.stack(f"DEL {aid}")
            except Exception:
                pass
        bs.stack.stack("TIME 0")     # reset sim clock
        bs.stack.stack("DT 1.0")     # set sim tick to 1 second
        bs.stack.stack("OP")         # resume operation

        # Load scenario and create aircraft
        if not self.scenario_path:
            raise ValueError("scenario_path not set")
            
        with open(self.scenario_path, "r", encoding="utf-8") as f:
            scen = json.load(f)

        # Reset agents list based on scenario
        self.agents = []
        
        # Clear buffers
        for aid in self.possible_agents:
            self._prev_actions_deg_kt[aid][:] = [0.0, 0.0]

        # Create aircraft from scenario
        for agent in scen["agents"]:
            aid = agent["id"]
            self.agents.append(aid)  # Add to agents list
            
            # Apply shifts deterministically - check targeted shifts first
            lat = float(agent["lat"])
            lon = float(agent["lon"])
            alt_ft = float(agent["alt_ft"])
            hdg = float(agent["hdg_deg"])
            spd_kt = float(agent["spd_kt"])
            actype = agent.get("type", "A320")
            
            # Apply targeted shifts for this specific agent
            if aid in targeted_shift:
                agent_shifts = targeted_shift[aid]
                hdg += float(agent_shifts.get("heading_deg_delta", 0.0))
                spd_kt += float(agent_shifts.get("speed_kt_delta", 0.0))
                
                # Position shifts - separate lat/lon deltas for more flexibility
                lat += float(agent_shifts.get("position_lat_delta", 0.0))
                lon += float(agent_shifts.get("position_lon_delta", 0.0))
                
                # Aircraft type shift - override default aircraft type if specified
                actype = agent_shifts.get("aircraft_type", actype)
                
                # Legacy position shift along track (for backward compatibility)
                pos_delta = float(agent_shifts.get("position_nm_delta", 0.0))
                if pos_delta != 0.0:
                    def nm_to_lat(nm): return nm/60.0
                    def nm_to_lon(nm, lat): return nm/(60.0 * max(1e-6, math.cos(math.radians(lat))))
                    ex = math.cos(math.radians(90.0 - hdg))
                    ny = math.sin(math.radians(90.0 - hdg))
                    east_nm, north_nm = pos_delta * ex, pos_delta * ny
                    lat += nm_to_lat(north_nm)
                    lon += nm_to_lon(east_nm, lat)
            else:
                # Apply legacy unison shifts if no targeted shift specified
                hdg += hdg_dg
                spd_kt += spd_dkt
                
                # Apply position shift along initial track
                if pos_nm != 0.0:
                    def nm_to_lat(nm): return nm/60.0
                    def nm_to_lon(nm, lat): return nm/(60.0 * max(1e-6, math.cos(math.radians(lat))))
                    ex = math.cos(math.radians(90.0 - hdg))
                    ny = math.sin(math.radians(90.0 - hdg))
                    east_nm, north_nm = pos_nm * ex, pos_nm * ny
                    lat += nm_to_lat(north_nm)
                    lon += nm_to_lon(east_nm, lat)
            
            # Ensure valid values
            hdg = hdg % 360.0
            spd_kt = max(100.0, min(400.0, spd_kt))  # Reasonable speed limits
            
            # Fixed BlueSky command format: CRE acid type lat lon hdg alt spd
            bs.stack.stack(f"CRE {aid} {actype} {lat:.6f} {lon:.6f} {int(round(hdg))} {alt_ft:.0f} {spd_kt:.0f}")
            
            # Set autothrottle to track SPD targets
           # bs.stack.stack(f"THR {aid} AUTO")
            
            # Remember each agent's baseline speed from scenario (CAS ~= spd_kt there)
            self._base_cas_kt[aid] = spd_kt
            # (Optional) Explicitly fix SPD mode at CAS knots via SPD cmd once
            bs.stack.stack(f"SPD {aid} {spd_kt:.1f}")
            
            wpt = agent.get("waypoint", {})
            wlat, wlon = float(wpt["lat"]), float(wpt["lon"])
            
            # Apply waypoint shifts if specified in targeted_shift
            if aid in targeted_shift:
                agent_shifts = targeted_shift[aid]
                wlat += float(agent_shifts.get("waypoint_lat_delta", 0.0))
                wlon += float(agent_shifts.get("waypoint_lon_delta", 0.0))
            
            self._agent_waypoints[aid] = (wlat, wlon)
            bs.stack.stack(f"ADDWPT {aid} {wlat:.6f} {wlon:.6f}")

        bs.sim.step()
        
        # Initialize team PBRS state
        obs = self._collect_observations()
        self._team_phi = self._compute_team_phi()
        self._team_dphi_ema = 0.0
        self._prev_wp_dist_nm = {aid: None for aid in self.possible_agents}
        self._waypoint_hits = {aid: 0 for aid in self.possible_agents}  # Reset waypoint hit counters
        self._agents_to_stop_logging = set()  # Reset agents to stop logging
        self._waypoint_reached = {aid: False for aid in self.possible_agents}  # Reset waypoint reached flags
        self._waypoint_reached_this_step = {aid: False for aid in self.possible_agents}  # Reset this-step flags
        self._agent_done = {aid: False for aid in self.possible_agents}  # Reset agent done flags
        self._agent_wpreached = {aid: False for aid in self.possible_agents}  # Reset persistent waypoint flags
        
        # Reset real-time hallucination detection
        if self._enable_hallucination_detection and self._hallucination_detector:
            self._rt_trajectory = {
                "positions": [],
                "actions": [],
                "agents": {aid: {"headings": [], "speeds": []} for aid in self.possible_agents},
                "waypoint_status": {aid: {} for aid in self.possible_agents}
            }
        
        infos = {aid: {} for aid in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, np.ndarray]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        # Apply actions and store for logging
        normalized_actions = {}  # Store normalized actions for cost calculation
        for aid, a in actions.items():
            if aid not in self.agents:
                continue
                
            # Zero actions for agents who have already reached their waypoint
            if self._waypoint_reached.get(aid, False):
                # Agent has completed waypoint - apply zero action (maintain course)
                a = np.array([0.0, 0.0], dtype=np.float32)
                normalized_actions[aid] = a
                self._prev_actions_deg_kt[aid][:] = [0.0, 0.0]
                continue
                
            # Apply action delay if enabled
            raw = np.asarray(a, dtype=np.float32).reshape(-1)
            raw = np.clip(raw, -1.0, 1.0)  # Ensure normalized actions are in [-1, 1]
            
            if self.action_delay_steps > 0:
                # Insert into per-agent FIFO; pop the executed command
                q = self._action_queues[aid]
                q.append(raw.tolist())  # Convert to list for queue storage
                a = np.array(q.pop(0), dtype=np.float32)
            else:
                a = raw
                
            normalized_actions[aid] = a  # Store normalized actions for cost calculation
            
            dh = float(a[0]) * D_HEADING     # heading change in degrees
            dv = float(a[1]) * D_VELOCITY    # speed change in knots

            idx = bs.traf.id2idx(aid)
            if not (isinstance(idx, int) and idx >= 0):
                continue

            # --- Direct control (current-state-relative) ---
            try:
                # Use hdg (not trk) to match drift penalty calculation
                current_hdg = float(getattr(bs.traf, 'hdg', [0])[idx])
                # BlueSky cas is stored in m/s internally, convert to knots for calculations
                current_cas_ms = float(getattr(bs.traf, 'cas', [250])[idx])
                current_cas_kt = current_cas_ms * 1.94384  # m/s to knots
            except Exception:
                current_hdg = 0.0
                current_cas_kt = 250.0

            # Apply heading change with proper angle bounding
            new_hdg = current_hdg + dh
            # Bound to [-180, 180] then convert to [0, 360] for BlueSky
            while new_hdg > 180.0: new_hdg -= 360.0
            while new_hdg < -180.0: new_hdg += 360.0
            if new_hdg < 0: new_hdg += 360.0

            # Apply speed change with bounds (in knots)
            new_spd_kt = current_cas_kt + dv
            lo, hi = self._spd_bounds_kt
            new_spd_kt = max(lo, min(hi, new_spd_kt))  # clamp to training window

            # Issue BlueSky commands (SPD expects knots, not m/s)
            bs.stack.stack(f"HDG {aid} {new_hdg:.1f}")
            bs.stack.stack(f"SPD {aid} {new_spd_kt:.1f}")

            # for logging/shaping
            self._prev_actions_deg_kt[aid][:] = [dh, dv]

        # Advance simulation
        for _ in range(10):
            bs.sim.step()

        self._step_idx += 1

        # Get current state for all calculations
        obs = self._collect_observations()
        
        # Enhanced distance and conflict calculations
        min_sep_nm_global = float('inf')
        conflict_flags = {aid: 0 for aid in self.agents}
        collision_flags = {aid: 0 for aid in self.agents}
        # Track agent states before main loop
        reached_wp = {aid: 0 for aid in self.agents}
        first_time_reach = {aid: False for aid in self.agents}  # Track first-time reaches this step
        dist_wp_nm_by_agent: Dict[str, float] = {}
        pairwise_dist_nm: Dict[str, Dict[str, float]] = {aid: {} for aid in self.agents}
        conflict_pairs_count = 0
        
        # Reset this-step waypoint tracking
        self._waypoint_reached_this_step = {aid: False for aid in self.possible_agents}
        
        # Track if any pair is inside collision band this step
        any_collision_this_step = False
        
        # Calculate waypoint distances and pairwise separations
        for i, aid_i in enumerate(self.agents):
            idx_i = bs.traf.id2idx(aid_i)
            if not isinstance(idx_i, int) or idx_i < 0:
                dist_wp_nm_by_agent[aid_i] = 1000.0
                continue
                
            try:
                lat_i = float(bs.traf.lat[idx_i])
                lon_i = float(bs.traf.lon[idx_i])
            except (IndexError, TypeError):
                dist_wp_nm_by_agent[aid_i] = 1000.0
                continue
            
            # Waypoint distance and reaching logic
            wpt = self._agent_waypoints.get(aid_i, (0, 0))
            dist_wp_nm = haversine_nm(lat_i, lon_i, wpt[0], wpt[1])
            dist_wp_nm_by_agent[aid_i] = float(dist_wp_nm)
            
            # Check if agent reaches waypoint (within 1 NM) - use persistent tracking
            if not self._agent_wpreached[aid_i] and dist_wp_nm <= WAYPOINT_THRESHOLD_NM:
                self._agent_wpreached[aid_i] = True
                reached_wp[aid_i] = 1
                self._waypoint_reached_this_step[aid_i] = True  # Mark as reached this step
                first_time_reach[aid_i] = True  # Mark as first-time reach
                self._waypoint_reached[aid_i] = True
                self._waypoint_hits[aid_i] = 1  # Individual agent hit count (0 or 1)
                # NOTE: Don't mark as done yet - let this step be logged first
            elif self._agent_wpreached[aid_i]:
                # Agent has already reached waypoint
                reached_wp[aid_i] = 1
                self._waypoint_reached[aid_i] = True
                self._waypoint_hits[aid_i] = 1  # Keep individual count at 1
                
            # Debug waypoint status every 10 steps
            if self._step_idx % 10 == 0 and aid_i == self.agents[0]:  # Only for first agent to avoid spam
                active_agents = [aid for aid in self.agents if not self._agent_done.get(aid, False)]
                done_agents = [aid for aid in self.agents if self._agent_done.get(aid, False)]

            # Pairwise separation
            for j, aid_j in enumerate(self.agents):
                if j <= i:
                    continue
                idx_j = bs.traf.id2idx(aid_j)
                if not isinstance(idx_j, int) or idx_j < 0:
                    continue
                    
                try:
                    lat_j = float(bs.traf.lat[idx_j])
                    lon_j = float(bs.traf.lon[idx_j])
                except (IndexError, TypeError):
                    continue
                d_nm = haversine_nm(lat_i, lon_i, lat_j, lon_j)
                min_sep_nm_global = min(min_sep_nm_global, d_nm)
                
                # Fill symmetric matrix
                pairwise_dist_nm[aid_i][aid_j] = float(d_nm)
                pairwise_dist_nm.setdefault(aid_j, {})[aid_i] = float(d_nm)
                
                if d_nm < self._separation_threshold_nm:
                    conflict_flags[aid_i] = 1
                    conflict_flags[aid_j] = 1
                    conflict_pairs_count += 1
                if d_nm < self.collision_nm:  # configurable hard collision threshold
                    collision_flags[aid_i] = 1
                    collision_flags[aid_j] = 1
                    any_collision_this_step = True

        # Compute comprehensive rewards with all components
        rewards = {}
        reward_parts_by_agent = {}
        dt = 10.0  # 10 sim steps × 1s each
        
        for aid in self.agents:
            idx = bs.traf.id2idx(aid)
            if not isinstance(idx, int) or idx < 0:
                r = -1.0
                # Calculate action cost using normalized actions if available, otherwise use deg/kt deltas
                norm = normalized_actions.get(aid, np.zeros(2, np.float32))
                if np.any(norm):  # Use normalized if available
                    act_cost = self.action_cost_per_unit * (abs(norm[0]) + abs(norm[1]))
                else:  # Fallback to deg/kt deltas
                    act_cost = self.action_cost_per_unit * float(np.abs(self._prev_actions_deg_kt[aid]).sum())
                parts = {"los_penalty": 0.0, "act_cost": float(act_cost), "progress": 0.0, "backtrack": 0.0,
                        "drift": 0.0, "align": 0.0, "time": -1.0, "dwell": 0.0, 
                        "intrusion": 0.0, "reach": 0.0, "total": float(r + act_cost)}
            else:
                # Special handling for agents who have already reached their waypoint
                if self._waypoint_reached[aid]:
                    # Simplified reward for completed agents: just avoid conflicts and minimize cost
                    r_dwell = 0.0; r_intr = 0.0
                    if conflict_flags[aid]:
                        r_intr = self.r_intrusion
                        r_dwell = self.r_dwell_per_s * dt
                    
                    los_penalty = 0.0  # don't double-count conflicts
                    act_cost = self.action_cost_per_unit * float(np.abs(self._prev_actions_deg_kt[aid]).sum())
                    r_time = self.r_time_per_s * dt * 0.1  # Reduced time penalty for completed agents
                    
                    # Give reach bonus ONLY on the first step when waypoint is reached (not subsequent steps)
                    r_reach = self.r_reach_bonus if first_time_reach[aid] else 0.0
                    
                    r_total = los_penalty + r_intr + r_dwell + act_cost + r_time + r_reach
                    
                    parts = {
                        "los_penalty": float(los_penalty), "act_cost": float(act_cost), 
                        "progress": 0.0, "backtrack": 0.0, "drift": 0.0, "align": 0.0,
                        "time": float(r_time), "dwell": float(r_dwell), "intrusion": float(r_intr),
                        "reach": float(r_reach), "terminal": 0.0, "total": float(r_total),
                    }
                else:
                    # Normal reward calculation for agents still working toward waypoint
                    # Get waypoint data
                    wp_lat, wp_lon = self._agent_waypoints[aid]
                    d_wp_nm = haversine_nm(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), wp_lat, wp_lon)
                    prev_wp = self._prev_wp_dist_nm.get(aid, None)
                    
                    # 1) Progress vs backtrack rewards (km)
                    r_progress = 0.0; r_backtrack = 0.0
                    if prev_wp is not None:
                        delta_nm = prev_wp - d_wp_nm
                        if delta_nm > 0:
                            r_progress = (delta_nm * NM_TO_KM) * self.r_prog_per_km
                        elif delta_nm < 0:
                            r_backtrack = (abs(delta_nm) * NM_TO_KM) * self.r_back_per_km
                    self._prev_wp_dist_nm[aid] = d_wp_nm
                    
                    # 2) Heading alignment (per sec) - drop competing drift penalty
                    try:
                        qdr, _ = bs.tools.geo.kwikqdrdist(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), wp_lat, wp_lon)
                    except:
                        qdr = 0.0
                    hdg = float(getattr(bs.traf, 'hdg', [0])[idx] if hasattr(bs.traf, 'hdg') else 0)
                    drift = hdg - qdr
                    while drift > 180: drift -= 360
                    while drift < -180: drift += 360
                    r_drift = self.r_drift_per_s * dt * abs(drift) / DRIFT_NORM_DEN  # Enable drift penalty
                    
                    # 3) Time penalty (per sec)
                    r_time = self.r_time_per_s * dt
                    
                    # 4) Conflict dwell penalty and intrusion penalty
                    r_dwell = 0.0; r_intr = 0.0
                    if conflict_flags[aid]:
                        r_intr = self.r_intrusion
                        r_dwell = self.r_dwell_per_s * dt
                    
                    # 5) Action cost based on normalized actions
                    norm = normalized_actions.get(aid, np.zeros(2, np.float32))
                    act_cost = self.action_cost_per_unit * (abs(norm[0]) + abs(norm[1]))
                    
                    # 6) Disable los_penalty to avoid double-counting conflicts
                    los_penalty = 0.0   # don't double-count conflicts
                    
                    # 7) Reach bonus (give ONLY on first step when reaching waypoint, not subsequent steps)
                    r_reach = self.r_reach_bonus if first_time_reach[aid] else 0.0
                    
                    # 8) Terminal penalty for not reaching waypoint at episode end
                    r_terminal = 0.0
                    if self._step_idx >= self.max_episode_steps and not self._waypoint_reached[aid]:
                        r_terminal = self.terminal_not_reached_penalty
                    
                    # Total reward
                    r_total = (
                        los_penalty +  # disabled (set to 0) to avoid double-counting
                        r_intr + r_dwell +
                        r_progress + r_backtrack +
                        r_drift +  # drift penalty
                        act_cost +     # action cost based on normalized actions
                        r_time +
                        r_reach +
                        r_terminal     # terminal penalty for not reaching waypoint
                    )
                    
                    parts = {
                        "los_penalty": float(los_penalty),
                        "act_cost": float(act_cost),
                        "progress": float(r_progress),
                        "backtrack": float(r_backtrack),
                        "drift": float(r_drift),
                        "time": float(r_time),
                        "dwell": float(r_dwell),
                        "intrusion": float(r_intr),
                        "reach": float(r_reach),
                        "terminal": float(r_terminal),
                        "total": float(r_total),
                    }
            
            rewards[aid] = float(parts.get("total", r_total if 'r_total' in locals() else 0.0))
            reward_parts_by_agent[aid] = parts

        # ---- Team PBRS shaping ----
        phi_now = self._compute_team_phi()
        dphi = self.team_gamma * phi_now - (self._team_phi if self._team_phi is not None else phi_now)
        # EMA smoothing
        self._team_dphi_ema = (1.0 - self.team_ema_a) * self._team_dphi_ema + self.team_ema_a * dphi
        # cap and anneal
        dphi_shaped = max(-self.team_cap, min(self.team_cap, self._team_dphi_ema)) * self.team_anneal
        self._team_phi = phi_now

        # share weights per agent
        weights = {aid: 1.0 for aid in self.agents}
        if self.team_share == "neighbor":
            # weight by near neighbors (within threshold)
            for aid in self.agents:
                near = 0
                for aj in self.agents:
                    if aj == aid: continue
                    dij = pairwise_dist_nm.get(aid, {}).get(aj, np.inf)
                    if np.isfinite(dij) and (dij*1.852) <= self.team_nb_km:
                        near += 1
                weights[aid] = 1.0 + near  # base 1 + near neighbor count
        elif self.team_share == "responsibility":
            # weight by participation in violating/conflicting pairs
            for aid in self.agents:
                w = 1.0
                for aj in self.agents:
                    if aj == aid: continue
                    dij = pairwise_dist_nm.get(aid, {}).get(aj, np.inf)
                    if np.isfinite(dij) and dij < self._separation_threshold_nm:
                        w += 1.0
                weights[aid] = w
        # normalize weights so Σ_i W_i = N (keeps scale stable)
        Wsum = sum(weights.values()) or 1.0
        scale = len(self.agents) / Wsum
        for aid in weights:
            weights[aid] *= scale

        # add team term to each agent's reward and log it
        for aid in self.agents:
            r_team = self.team_w * weights[aid] * dphi_shaped
            rewards[aid] = float(rewards.get(aid, 0.0) + r_team)
            reward_parts_by_agent[aid]["team"] = float(r_team)
            reward_parts_by_agent[aid]["team_phi"] = float(phi_now)
            reward_parts_by_agent[aid]["team_dphi"] = float(dphi_shaped)

        # Note: Collision detection still used for rewards but not for episode termination
        # self._collide_streak tracking removed - no early termination on collisions
        
        # Terminations and truncations - only time limit and waypoint completion
        all_at_waypoint = len(self.agents) > 0 and all(reached_wp.values())
        time_limit_reached = self._step_idx >= self.max_episode_steps
        
        # Episode-level termination for RLlib compatibility
        # All agents terminate together when episode is complete
        episode_complete = all_at_waypoint or time_limit_reached
        
        terminations = {aid: episode_complete for aid in self.agents}
        truncations = {aid: False for aid in self.agents}

        # Enhanced infos with reward breakdown
        infos = {}
        # Calculate total cumulative waypoint hits (number of agents who have reached waypoints)
        total_waypoint_hits = sum(1 for aid in self.agents if self._waypoint_reached.get(aid, False))
        for aid in self.agents:
            # Calculate per-agent minimum separation
            agent_min_sep = 200.0  # default
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
                "sim_time_s": float(bs.sim.simt),  # Add simulation time for trajectory logging
                "total_waypoint_hits": total_waypoint_hits
            }

        # Update real-time trajectory for hallucination detection
        if self._enable_hallucination_detection and self._hallucination_detector:
            self._update_rt_trajectory(normalized_actions)
        
        # Enhanced trajectory logging
        if self.log_trajectories:
            self._log_step(conflict_flags, collision_flags, 
                          rewards, pairwise_dist_nm, dist_wp_nm_by_agent, 
                          conflict_pairs_count, reward_parts_by_agent, first_time_reach)

        # Mark agents as done AFTER logging their waypoint completion step
        for aid in self.agents:
            if self._agent_wpreached.get(aid, False) and not self._agent_done.get(aid, False):
                self._agent_done[aid] = True
                self._agents_to_stop_logging.add(aid)

        # End episode when termination conditions are met
        if episode_complete:
            if self.log_trajectories and self._traj_rows:
                self._save_trajectory_csv()
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def _relative_obs_for(self, aid: str) -> dict:
        """Return dict obs with relative features (no raw lat/lon)."""
        idx = bs.traf.id2idx(aid)
        if not isinstance(idx, int) or idx < 0:
            K = self.neighbor_topk
            null = np.zeros(1, np.float32)
            zK  = np.zeros(K, np.float32)
            return {
                "wp_dist_norm": null, "cos_to_wp": null, "sin_to_wp": null,
                "airspeed": null, "x_r": zK, "y_r": zK, "vx_r": zK, "vy_r": zK
            }

        # drift wrt waypoint course and waypoint distance
        wlat, wlon = self._agent_waypoints[aid]
        try:
            qdr, _ = bs.tools.geo.kwikqdrdist(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), wlat, wlon)  # deg
        except:
            qdr = 0.0
        
        # Calculate distance to waypoint and normalize
        wp_d_nm = haversine_nm(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]), wlat, wlon)
        # Normalize distance around the 1 NM capture radius -> [-1, 1]
        wp_dist_norm = np.array([np.clip((wp_d_nm - WAYPOINT_THRESHOLD_NM)/WAYPOINT_THRESHOLD_NM, -1.0, 1.0)], np.float32)
        
        try:
            hdg_array = getattr(bs.traf, 'hdg', None)
            if hdg_array is not None and len(hdg_array) > idx:
                hdg = float(hdg_array[idx])
            else:
                hdg = 0.0
        except (IndexError, TypeError, AttributeError):
            hdg = 0.0
            
        # Direction-to-WP vs current heading  
        theta = math.radians(qdr - hdg)
        cos_to_wp = np.array([math.cos(theta)], np.float32)
        sin_to_wp = np.array([math.sin(theta)], np.float32)

        # own airspeed (normalize to roughly ~[-1,1])
        try:
            tas_array = getattr(bs.traf, 'tas', None)
            if tas_array is not None and len(tas_array) > idx:
                tas_mps = float(tas_array[idx])
            else:
                tas_mps = 150.0
        except (IndexError, TypeError, AttributeError):
            tas_mps = 150.0
        airspeed = np.array([(tas_mps - 150.0) / 6.0], np.float32)

        # build neighbor arrays
        ids = [a for a in self.agents if a != aid]
        # sort by range
        ranges = []
        for aj in ids:
            jdx = bs.traf.id2idx(aj)
            if not isinstance(jdx, int) or jdx < 0: continue
            d = haversine_nm(float(bs.traf.lat[idx]), float(bs.traf.lon[idx]),
                           float(bs.traf.lat[jdx]), float(bs.traf.lon[jdx]))
            ranges.append((d, aj, jdx))
        ranges.sort(key=lambda t: t[0])
        ranges = ranges[:self.neighbor_topk]

        # own ground velocity approx (NM/s)
        tas_kt = tas_mps * 1.94384
        vx = kt_to_nms(tas_kt) * math.cos(math.radians(hdg))
        vy = kt_to_nms(tas_kt) * math.sin(math.radians(hdg))

        K = self.neighbor_topk
        x_r = np.zeros(K, np.float32); y_r = np.zeros(K, np.float32)
        vx_r= np.zeros(K, np.float32); vy_r= np.zeros(K, np.float32)

        # local x (east), y (north) in NM—use simple ENU around ownship
        lat0, lon0 = float(bs.traf.lat[idx]), float(bs.traf.lon[idx])
        for k,(d_nm, aj, jdx) in enumerate(ranges):
            # position in NM
            dx_nm = haversine_nm(lat0, lon0, lat0, float(bs.traf.lon[jdx])) * (1 if float(bs.traf.lon[jdx])>=lon0 else -1)
            dy_nm = haversine_nm(lat0, lon0, float(bs.traf.lat[jdx]), lon0) * (1 if float(bs.traf.lat[jdx])>=lat0 else -1)
            x_r[k] = dx_nm * NM_TO_KM * 1000.0 / 13000.0   # scaling
            y_r[k] = dy_nm * NM_TO_KM * 1000.0 / 13000.0

            # velocities (NM/s)
            try:
                hdg_array = getattr(bs.traf, 'hdg', None)
                tas_array = getattr(bs.traf, 'tas', None)
                if hdg_array is not None and len(hdg_array) > jdx:
                    hj = float(hdg_array[jdx])
                else:
                    hj = 0.0
                if tas_array is not None and len(tas_array) > jdx:
                    taj_mps = float(tas_array[jdx])
                else:
                    taj_mps = 150.0
            except (IndexError, TypeError, AttributeError):
                hj = 0.0
                taj_mps = 150.0
            vj = kt_to_nms(taj_mps * 1.94384)
            vxj = vj*math.cos(math.radians(hj)); vyj = vj*math.sin(math.radians(hj))
            vx_r[k] = (vxj - vx) / 32.0
            vy_r[k] = (vyj - vy) / 66.0

        return {
            "wp_dist_norm": wp_dist_norm,  # normalized distance to own waypoint
            "cos_to_wp": cos_to_wp,        # cosine of direction to waypoint
            "sin_to_wp": sin_to_wp,        # sine of direction to waypoint
            "airspeed":  airspeed,
            "x_r":       x_r, "y_r": y_r,
            "vx_r":      vx_r, "vy_r": vy_r,
        }

    def _update_rt_trajectory(self, actions: Dict[str, np.ndarray]):
        """Update real-time trajectory data for hallucination detection."""
        if not self._enable_hallucination_detection or not self._hallucination_detector:
            return
            
        # Capture current positions
        positions = {}
        for aid in self.agents:
            idx = bs.traf.id2idx(aid)
            if isinstance(idx, int) and idx >= 0:
                try:
                    lat = float(bs.traf.lat[idx])
                    lon = float(bs.traf.lon[idx])
                    positions[aid] = (lat, lon)
                    
                    # Capture headings and speeds
                    hdg = float(getattr(bs.traf, 'hdg', [0])[idx] if hasattr(bs.traf, 'hdg') else 0)
                    tas_ms = float(getattr(bs.traf, 'tas', [150])[idx] if hasattr(bs.traf, 'tas') else 150)
                    tas_kt = tas_ms * 1.94384  # Convert to knots
                    
                    self._rt_trajectory["agents"][aid]["headings"].append(hdg)
                    self._rt_trajectory["agents"][aid]["speeds"].append(tas_kt)
                    
                except (IndexError, TypeError, AttributeError):
                    positions[aid] = (0.0, 0.0)
                    self._rt_trajectory["agents"][aid]["headings"].append(0.0)
                    self._rt_trajectory["agents"][aid]["speeds"].append(250.0)
            else:
                positions[aid] = (0.0, 0.0)
                self._rt_trajectory["agents"][aid]["headings"].append(0.0)
                self._rt_trajectory["agents"][aid]["speeds"].append(250.0)
        
        self._rt_trajectory["positions"].append(positions)
        
        # Capture actions (convert from normalized to deg/kt deltas)
        action_step = {}
        for aid in self.agents:
            if aid in actions:
                # Convert normalized actions to actual deltas
                a = actions[aid]
                dh = float(a[0]) * D_HEADING  # degrees
                dv = float(a[1]) * D_VELOCITY  # knots
                action_step[aid] = [dh, dv]
            else:
                action_step[aid] = [0.0, 0.0]
        
        self._rt_trajectory["actions"].append(action_step)
        
        # Update waypoint status
        for aid in self.agents:
            step_idx = len(self._rt_trajectory["positions"]) - 1
            self._rt_trajectory["waypoint_status"][aid][step_idx] = self._agent_wpreached.get(aid, False)

    def _collect_observations(self) -> Dict[str, np.ndarray]:
        """Build observations based on obs_mode."""
        obs = {}
        
        try:
            for aid in self.agents:
                if aid is None:
                    continue
                
                if self.obs_mode == "relative":
                    obs[aid] = self._relative_obs_for(aid)
                else:
                    # Legacy absolute observation mode
                    idx = bs.traf.id2idx(aid)
                    if isinstance(idx, int) and idx >= 0:
                        try:
                            lat = float(getattr(bs.traf, 'lat', [0])[idx] if hasattr(bs.traf, 'lat') else 0)
                            lon = float(getattr(bs.traf, 'lon', [0])[idx] if hasattr(bs.traf, 'lon') else 0)
                            hdg = float(getattr(bs.traf, 'hdg', [0])[idx] if hasattr(bs.traf, 'hdg') else 0)
                            tas_ms = float(getattr(bs.traf, 'tas', [50])[idx] if hasattr(bs.traf, 'tas') else 50)
                            alt_m = float(getattr(bs.traf, 'alt', [3000])[idx] if hasattr(bs.traf, 'alt') else 3000)
                            
                            tas_kt = tas_ms * 1.94384  # m/s to knots
                            alt_ft = alt_m * 3.28084   # m to feet
                            
                            # Simple separation to nearest other aircraft
                            min_sep = 200.0  # default if no other aircraft
                            for other_aid in self.agents:
                                if other_aid != aid:
                                    other_idx = bs.traf.id2idx(other_aid)
                                    if isinstance(other_idx, int) and other_idx >= 0:
                                        other_lat = float(getattr(bs.traf, 'lat', [0])[other_idx] if hasattr(bs.traf, 'lat') else 0)
                                        other_lon = float(getattr(bs.traf, 'lon', [0])[other_idx] if hasattr(bs.traf, 'lon') else 0)
                                        # Simple Euclidean distance in NM (approximation)
                                        d_lat = (lat - other_lat) * 60.0
                                        d_lon = (lon - other_lon) * 60.0 * max(1e-6, math.cos(math.radians(lat)))
                                        sep = math.sqrt(d_lat*d_lat + d_lon*d_lon)
                                        min_sep = min(min_sep, sep)
                            
                            # Build observation vector
                            vec = np.array([
                                lat, lon, hdg, tas_kt, alt_ft, min_sep,
                                min(self._step_idx / self.max_episode_steps, 1.0)
                            ], dtype=np.float32)
                            
                            # Clip to declared Box bounds (only for Box spaces)
                            space = self._observation_spaces[aid]
                            if isinstance(space, spaces.Box):
                                vec = np.clip(vec, space.low, space.high).astype(np.float32)
                            obs[aid] = vec
                            
                        except Exception as e:
                            # Fallback if BlueSky data access fails
                            obs[aid] = np.array([0, 0, 0, 100, 10000, 100, 0], dtype=np.float32)
                    else:
                        # Aircraft not found in BlueSky
                        obs[aid] = np.array([0, 0, 0, 100, 10000, 100, 0], dtype=np.float32)
                
        except Exception as e:
            self._logger.error(f"Error in observation collection: {e}")
            for aid in self.agents:
                if aid is not None:
                    if self.obs_mode == "relative":
                        K = self.neighbor_topk
                        null = np.zeros(1, np.float32)
                        zK  = np.zeros(K, np.float32)
                        obs[aid] = {
                            "wp_dist_norm": null, "cos_to_wp": null, "sin_to_wp": null,
                            "airspeed": null, "x_r": zK, "y_r": zK, "vx_r": zK, "vy_r": zK
                        }
                    else:
                        obs[aid] = np.array([0, 0, 0, 100, 10000, 100, 0], dtype=np.float32)
                
        return obs

    def _log_step(self, conflict_flags: Dict[str, int],
                  collision_flags: Dict[str, int], rewards: Dict[str, float],
                  pairwise_dist_nm: Dict[str, Dict[str, float]], 
                  dist_wp_nm_by_agent: Dict[str, float], conflict_pairs_count: int,
                  reward_parts_by_agent: Dict[str, dict], first_time_reach: Dict[str, bool]):
        """Append step trace rows for all agents with enhanced data including real-time hallucination detection."""        
        # Compute real-time hallucination metrics if enabled
        rt_hallucination_data = {}
        if (self._enable_hallucination_detection and self._hallucination_detector and 
            len(self._rt_trajectory["positions"]) >= 2):  # Need at least 2 steps for detection
            try:
                # Run hallucination detection on current trajectory
                cm = self._hallucination_detector.compute(
                    self._rt_trajectory, 
                    sep_nm=self._separation_threshold_nm, 
                    return_series=True
                )
                series_data = cm.get("series", {})
                
                # Get current step index (0-based)
                current_step = len(self._rt_trajectory["positions"]) - 1
                if (current_step < len(series_data.get("gt_conflict", [])) and 
                    current_step < len(series_data.get("alert", []))):
                    rt_hallucination_data = {
                        "gt_conflict": series_data["gt_conflict"][current_step],
                        "predicted_alert": series_data["alert"][current_step],
                        "tp": series_data["tp"][current_step],
                        "fp": series_data["fp"][current_step],
                        "fn": series_data["fn"][current_step],
                        "tn": series_data["tn"][current_step]
                    }
            except Exception as e:
                # Fallback if detection fails
                self._logger.debug(f"Real-time hallucination detection failed: {e}")
                rt_hallucination_data = {
                    "gt_conflict": 0, "predicted_alert": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0
                }
        
        # Stable order for distance columns
        all_ids = list(self.possible_agents)
        
        for aid in self.agents:
            # Skip logging for agents that are done (reached waypoint in previous steps)
            if self._agent_done.get(aid, False):
                if self._step_idx % 20 == 0:  # Debug every 20 steps to avoid spam
                    pass  # Silent skip
                continue
            idx = bs.traf.id2idx(aid)
            
            # Basic state data
            if isinstance(idx, int) and idx >= 0:
                try:
                    lat_deg = float(bs.traf.lat[idx])
                    lon_deg = float(bs.traf.lon[idx])
                    alt_ft = float(bs.traf.alt[idx] * 3.28084)
                    hdg_deg = float(getattr(bs.traf, 'hdg', [0])[idx] if hasattr(bs.traf, 'hdg') else 0)
                    tas_kt = float(bs.traf.tas[idx] * 1.94384)
                    cas_kt = float(getattr(bs.traf, 'cas', [150])[idx] * 1.94384 if hasattr(bs.traf, 'cas') else 150)
                except (IndexError, TypeError, AttributeError):
                    lat_deg = lon_deg = alt_ft = hdg_deg = tas_kt = cas_kt = np.nan
            else:
                lat_deg = lon_deg = alt_ft = hdg_deg = tas_kt = cas_kt = np.nan
            
            # Calculate per-agent minimum separation
            agent_min_sep = 200.0  # default
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
            
            # Per-agent distances to EVERY other agent
            for other_id in all_ids:
                key = f"dist_to_{other_id}_nm"
                if other_id == aid:
                    row[key] = ""
                else:
                    row[key] = float(pairwise_dist_nm.get(aid, {}).get(other_id, np.nan))
            
            # Comprehensive reward breakdown
            parts = reward_parts_by_agent.get(aid, {})
            row.update({
                "reward_los_penalty": parts.get("los_penalty", 0.0),
                "reward_act_cost": parts.get("act_cost", 0.0),
                "reward_progress": parts.get("progress", 0.0),
                "reward_backtrack": parts.get("backtrack", 0.0),
                "reward_drift": parts.get("drift", 0.0),
                "reward_time": parts.get("time", 0.0),
                "reward_dwell": parts.get("dwell", 0.0),
                "reward_intrusion": parts.get("intrusion", 0.0),
                "reward_reach": parts.get("reach", 0.0),
                "reward_total": parts.get("total", 0.0),
                "reward_team": parts.get("team", 0.0),
                "team_phi": parts.get("team_phi", 0.0),
                "team_dphi": parts.get("team_dphi", 0.0),
            })
            
            # Add real-time hallucination detection results (same for all agents at this timestep)
            if rt_hallucination_data:
                row.update({
                    "gt_conflict": rt_hallucination_data.get("gt_conflict", 0),
                    "predicted_alert": rt_hallucination_data.get("predicted_alert", 0),
                    "tp": rt_hallucination_data.get("tp", 0),
                    "fp": rt_hallucination_data.get("fp", 0),
                    "fn": rt_hallucination_data.get("fn", 0),
                    "tn": rt_hallucination_data.get("tn", 0)
                })
            else:
                # Default values if detection is disabled or failed
                row.update({
                    "gt_conflict": 0, "predicted_alert": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0
                })
            
            self._traj_rows.append(row)
    
    def _save_trajectory_csv(self):
        """Save trajectory data to CSV file."""
        if not self._traj_rows:
            return
            
        # Use the results directory passed from trainer (or default)
        results_dir = getattr(self, 'results_dir', 'results')
        
        # Ensure absolute path
        if not os.path.isabs(results_dir):
            results_dir = os.path.join(os.path.dirname(__file__), "..", "..", results_dir)
        
        # Use episode_tag from config if available, otherwise use episode_id
        episode_tag = getattr(self, 'episode_tag', None)
        if episode_tag:
            traj_path = os.path.join(results_dir, f"traj_{episode_tag}.csv")
        else:
            traj_path = os.path.join(results_dir, f"traj_ep_{self._episode_id:04d}.csv")
        os.makedirs(results_dir, exist_ok=True)
        
        # Get all column names from first row
        if self._traj_rows:
            fieldnames = list(self._traj_rows[0].keys())
            
            with open(traj_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._traj_rows)
        
        # Clear for next episode
        self._traj_rows = []

    def _compute_team_phi(self) -> float:
        """Compute team potential Phi(s) based on minimum pairwise separation (0..1 in [0..10 NM])."""
        ids = self.agents
        if not ids or len(ids) < 2:
            return 1.0  # Maximum potential when no conflicts possible
        
        min_separation_nm = float('inf')
        
        for i in range(len(ids)):
            idx_i = bs.traf.id2idx(ids[i])
            if not isinstance(idx_i, int) or idx_i < 0:
                continue
                
            for j in range(i+1, len(ids)):
                idx_j = bs.traf.id2idx(ids[j])
                if not isinstance(idx_j, int) or idx_j < 0:
                    continue
                    
                try:
                    d_nm = haversine_nm(float(bs.traf.lat[idx_i]), float(bs.traf.lon[idx_i]),
                                       float(bs.traf.lat[idx_j]), float(bs.traf.lon[idx_j]))
                    min_separation_nm = min(min_separation_nm, d_nm)
                except (IndexError, TypeError):
                    continue
                    
        # Handle case where no valid pairs found
        if min_separation_nm == float('inf'):
            return 1.0
            
        # Normalize min separation to [0, 1] range for [0, 10 NM]
        phi_t = np.clip(min_separation_nm / 10.0, 0.0, 1.0)
        return float(phi_t)