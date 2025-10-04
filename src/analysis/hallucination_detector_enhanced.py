"""
Module Name: hallucination_detector_enhanced.py
Description: Real-time detection of conflict prediction errors (hallucinations) in MARL-based ATC systems.
Author: Som
Date: 2025-10-04

Compares ground truth conflict predictions (TCPA/DCPA) against policy action patterns to identify
false alerts and missed conflicts with intent-aware filtering and IoU-based window matching.
"""

import math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np


def _bearing_deg(lat1, lon1, lat2, lon2):
    """Calculate initial bearing from point 1 to point 2 in degrees."""
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.cos(math.radians(lon2 - lon1)))
    brg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return brg

def _angdiff(a, b):
    """Calculate the smallest angular difference between angles a and b."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d

def _threat_for_agent(pos_t, hdg_t, spd_t, aid, horizon_s, los_nm):
    """
    Find the most threatening intruder aircraft for a given agent.
    
    Args:
        pos_t: Dictionary of agent positions at current timestep
        hdg_t: Dictionary of agent headings at current timestep  
        spd_t: Dictionary of agent speeds at current timestep
        aid: Target agent ID to find threats for
        horizon_s: Time horizon for TCPA/DCPA calculations
        los_nm: Loss of separation threshold in nautical miles
    
    Returns:
        Tuple of (intruder_id, tcpa_seconds, dcpa_nautical_miles) for most threatening intruder
    """
    lat_i, lon_i = pos_t[aid]
    hi, si_kt = hdg_t[aid], spd_t[aid]
    ei, ni = heading_to_unit(hi)
    vi_e, vi_n = kt_to_nms(si_kt) * ei, kt_to_nms(si_kt) * ni
    best = (None, float('inf'), float('inf'))
    
    for aj, (lat_j, lon_j) in pos_t.items():
        if aj == aid: 
            continue
        hj, sj_kt = hdg_t[aj], spd_t[aj]
        ej, nj = heading_to_unit(hj)
        vj_e, vj_n = kt_to_nms(sj_kt) * ej, kt_to_nms(sj_kt) * nj
        tcpa, dcpa = compute_tcpa_dcpa_nm(lat_i, lon_i, vi_e, vi_n, lat_j, lon_j, vj_e, vj_n, horizon_s)
        
        # Use smallest DCPA, then smallest TCPA as tiebreaker
        if dcpa < best[2] or (abs(dcpa - best[2]) < 1e-6 and tcpa < best[1]):
            best = (aj, tcpa, dcpa)
    
    return best


def haversine_nm(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in nautical miles."""
    R_nm = 3440.065
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2.0) ** 2)
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return R_nm * c

def heading_to_unit(heading_deg: float) -> Tuple[float, float]:
    """Convert heading in degrees to unit vector (east, north)."""
    rad = math.radians(90.0 - heading_deg)
    return (math.cos(rad), math.sin(rad))

def kt_to_nms(kt: float) -> float:
    """Convert knots to nautical miles per second."""
    return kt / 3600.0


def compute_tcpa_dcpa_nm(lat_i, lon_i, vi_east_nms, vi_north_nms,
                         lat_j, lon_j, vj_east_nms, vj_north_nms,
                         horizon_s: float = 120.0) -> Tuple[float, float]:
    """
    Compute Time to Closest Point of Approach (TCPA) and Distance at CPA (DCPA).
    
    Args:
        lat_i, lon_i: Aircraft i position in degrees
        vi_east_nms, vi_north_nms: Aircraft i velocity in NM/s
        lat_j, lon_j: Aircraft j position in degrees  
        vj_east_nms, vj_north_nms: Aircraft j velocity in NM/s
        horizon_s: Maximum time horizon for TCPA calculation
        
    Returns:
        Tuple of (tcpa_seconds, dcpa_nautical_miles)
    """
    # Convert lat/lon differences to relative positions in NM
    dx_nm = haversine_nm(lat_i, lon_i, lat_i, lon_j) * (1 if lon_j >= lon_i else -1)
    dy_nm = haversine_nm(lat_i, lon_i, lat_j, lon_i) * (1 if lat_j >= lat_i else -1)

    # Relative position and velocity
    rx, ry = dx_nm, dy_nm
    vx, vy = (vj_east_nms - vi_east_nms), (vj_north_nms - vi_north_nms)
    vv = vx*vx + vy*vy
    
    # Handle negligible relative velocity
    if vv < 1e-12:
        return 0.0, math.hypot(rx, ry)
    
    # Calculate TCPA and constrain to [0, horizon_s]
    tcpa = -(rx*vx + ry*vy) / vv
    tcpa = max(0.0, min(horizon_s, tcpa))
    
    # Calculate DCPA at constrained TCPA
    dcpa = math.hypot(rx + vx*tcpa, ry + vy*tcpa)
    return tcpa, dcpa


class HallucinationDetector:
    """
    Real-time hallucination detector for MARL air traffic control systems.
    
    Detects false alerts (ghost conflicts) and missed conflicts by comparing ground truth
    TCPA/DCPA-based conflict predictions with policy action patterns. Uses sophisticated
    intent-aware filtering and threat-aware gating to minimize false positives.
    """

    def __init__(self, horizon_s: float = 120.0, action_thresh=(3.0, 5.0),
                 res_window_s: float = 60.0, action_period_s: float = 10.0,
                 los_threshold_nm: float = 5.0, lag_pre_steps: int = 1,
                 lag_post_steps: int = 1, debounce_n: int = 2, debounce_m: int = 3,
                 iou_threshold: float = 0.1):
        """
        Initialize the hallucination detector.
        
        Args:
            horizon_s: Time horizon for TCPA/DCPA calculations
            action_thresh: Tuple of (heading_deg, speed_kt) thresholds for alert detection
            res_window_s: Time window for resolution verification after conflicts
            action_period_s: Environment timestep duration in seconds
            los_threshold_nm: Loss of separation threshold in nautical miles
            lag_pre_steps: Alert expansion steps before detected action
            lag_post_steps: Alert expansion steps after detected action
            debounce_n: Minimum detections in window (e.g., 2 of 3)
            debounce_m: Debounce window size in timesteps
            iou_threshold: Intersection over Union threshold for event matching
        """
        self.horizon_s = float(horizon_s)
        self.theta_min_deg = float(action_thresh[0])
        self.v_min_kt = float(action_thresh[1])
        self.res_window_s = float(res_window_s)
        self.action_period_s = float(action_period_s)
        self.los_threshold_nm = float(los_threshold_nm)
        self.lag_pre_steps = int(lag_pre_steps)
        self.lag_post_steps = int(lag_post_steps)
        self.debounce_n = int(debounce_n)
        self.debounce_m = int(debounce_m)
        self.iou_threshold = float(iou_threshold)

    def _detect_los_events(self, pos_seq: List[Dict[str, Tuple[float, float]]]) -> Dict[str, Any]:
        """
        Detect Loss of Separation (LOS) events between aircraft pairs.
        
        Args:
            pos_seq: Sequence of position dictionaries by timestep
            
        Returns:
            Dictionary containing LOS event statistics and details
        """
        T = len(pos_seq)
        los_events = []
        min_separation = float('inf')
        total_los_duration = 0
        current_los_start = None
        
        for t in range(T):
            step = pos_seq[t]
            aids = list(step.keys())
            step_min_sep = float('inf')
            step_has_los = False
            
            for i in range(len(aids)):
                for j in range(i+1, len(aids)):
                    ai, aj = aids[i], aids[j]
                    lat_i, lon_i = step[ai]
                    lat_j, lon_j = step[aj]
                    
                    sep_nm = haversine_nm(lat_i, lon_i, lat_j, lon_j)
                    step_min_sep = min(step_min_sep, sep_nm)
                    
                    if sep_nm < self.los_threshold_nm:
                        step_has_los = True
                        if current_los_start is None:
                            current_los_start = t
                            los_events.append({
                                'start_time': t,
                                'aircraft_pair': (ai, aj),
                                'min_separation': sep_nm
                            })
                        else:
                            # Update minimum separation for current LOS event
                            if los_events and los_events[-1]['min_separation'] > sep_nm:
                                los_events[-1]['min_separation'] = sep_nm
            
            min_separation = min(min_separation, step_min_sep)
            
            # End LOS event if no LOS in this step
            if not step_has_los and current_los_start is not None:
                if los_events:
                    los_events[-1]['end_time'] = t - 1
                    los_events[-1]['duration'] = t - current_los_start
                    total_los_duration += los_events[-1]['duration']
                current_los_start = None
        
        # Close any ongoing LOS event
        if current_los_start is not None and los_events:
            los_events[-1]['end_time'] = T - 1
            los_events[-1]['duration'] = T - current_los_start
            total_los_duration += los_events[-1]['duration']
        
        return {
            'num_los_events': len(los_events),
            'total_los_duration': total_los_duration,
            'min_separation_nm': min_separation if min_separation < float('inf') else 0.0,
            'los_events': los_events
        }

    def _calculate_efficiency_metrics(self, trajectory: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate path efficiency and completion metrics.
        
        Args:
            trajectory: Episode trajectory data with positions, waypoints, and metadata
            
        Returns:
            Dictionary of efficiency metrics including path lengths, completion rates,
            and deviations from optimal routes
        """
        pos_seq = trajectory.get("positions", [])
        timestamps = trajectory.get("timestamps", [])
        waypoints = trajectory.get("waypoints", {})
        waypoint_status = trajectory.get("waypoint_status", {})
        scenario_meta = trajectory.get("scenario_metadata", {})
        
        if not pos_seq or not timestamps:
            return {
                'total_path_length_nm': 0.0,
                'avg_path_length_nm': 0.0,
                'flight_time_s': 0.0,
                'path_efficiency': 0.0,
                'waypoint_reached_ratio': 0.0,
                'total_extra_path_nm': 0.0,
                'avg_extra_path_nm': 0.0,
                'avg_extra_path_ratio': 0.0
            }
        
        # FIXED: Get all agents that appear in trajectory (not just first timestep)
        all_agents = set()
        for step in pos_seq:
            all_agents.update(step.keys())
        all_agents = list(all_agents)
        
        # Calculate per-agent path metrics
        agent_path_lengths, agent_extra_nm, agent_extra_ratio = {}, {}, {}
        waypoints_reached = 0
        total_agents = len(all_agents)

        for aid in all_agents:
            # Calculate actual flight path length
            pts = [step[aid] for step in pos_seq if aid in step]
            L = sum(haversine_nm(*pts[i-1], *pts[i]) for i in range(1, len(pts)))
            agent_path_lengths[aid] = L

            # FIXED: Check waypoint completion from trajectory data
            agent_reached_waypoint = False
            
            # Method 1: Check if we have CSV-style trajectory data with waypoint_reached column
            csv_waypoint_completion = trajectory.get("csv_waypoint_completion", {})
            if csv_waypoint_completion.get(aid) is not None:
                # Use CSV waypoint completion data (preferred for CSV-based trajectories)
                agent_reached_waypoint = csv_waypoint_completion[aid] > 0
            elif waypoint_status.get(aid):
                # Method 2: Use waypoint_status from trajectory metadata if available
                final_timestep = len(pos_seq) - 1
                agent_reached_waypoint = waypoint_status[aid].get(final_timestep, False)
            elif waypoints.get(aid) and pts:
                # Method 3: Fall back to distance-based detection (legacy method)
                wlat, wlon = waypoints[aid]["lat"], waypoints[aid]["lon"]
                final_lat, final_lon = pts[-1]
                final_dist = haversine_nm(final_lat, final_lon, wlat, wlon)
                agent_reached_waypoint = final_dist <= 1.0
            elif len(timestamps) > 50:
                # Method 4: Assume completion for reasonable flight duration (fallback)
                agent_reached_waypoint = True
            
            if agent_reached_waypoint:
                waypoints_reached += 1

            # Compare to direct path if waypoint is available
            if waypoints.get(aid) and pts:
                wlat, wlon = waypoints[aid]["lat"], waypoints[aid]["lon"]
                direct_nm = haversine_nm(pts[0][0], pts[0][1], wlat, wlon)
                extra_nm = max(0.0, L - direct_nm)
                agent_extra_nm[aid] = extra_nm
                agent_extra_ratio[aid] = (L / max(1e-6, direct_nm)) - 1.0
            else:
                agent_extra_nm[aid] = 0.0
                agent_extra_ratio[aid] = 0.0

        total_path_length = sum(agent_path_lengths.values())
        avg_path_length   = total_path_length / max(1, len(agent_path_lengths))
        total_extra_nm    = sum(agent_extra_nm.values())
        avg_extra_nm      = total_extra_nm / max(1, len(agent_extra_nm))
        avg_extra_ratio   = sum(agent_extra_ratio.values()) / max(1, len(agent_extra_ratio))

        flight_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        waypoint_reached_ratio = waypoints_reached / max(1, total_agents)
        
        # Path efficiency: use flight time vs expected time
        expected_time = 300.0  # seconds, based on typical scenario duration
        path_efficiency = min(1.0, expected_time / max(1.0, flight_time))
        
        return {
            'total_path_length_nm': float(total_path_length),
            'avg_path_length_nm': float(avg_path_length),
            'flight_time_s': float(flight_time),
            'path_efficiency': float(path_efficiency),
            'waypoint_reached_ratio': float(waypoint_reached_ratio),
            # NEW:
            'total_extra_path_nm': float(total_extra_nm),
            'avg_extra_path_nm': float(avg_extra_nm),
            'avg_extra_path_ratio': float(avg_extra_ratio)
        }

    def _alerts_from_actions(self, actions_seq, trajectory) -> Tuple[np.ndarray, List[Optional[Dict[str, Any]]]]:
        """
        Generate alert flags from agent actions using intent-aware and threat-aware filtering.
        
        This method filters out routine navigation actions and only triggers alerts when:
        1. Action magnitude exceeds thresholds (heading/speed changes)
        2. Action is not purely navigational (toward waypoint)
        3. A near-term threat exists (TCPA/DCPA within bounds)
        
        Args:
            actions_seq: Sequence of action dictionaries by timestep
            trajectory: Episode trajectory containing positions, waypoints, and agent data
            
        Returns:
            Tuple of (alert_flags_array, metadata_list) where alert_flags is a boolean
            array indicating alert status per timestep, and metadata contains threat
            information for each alert
        """
        T = len(actions_seq)
        alerts = np.zeros(T, dtype=bool)
        meta: List[Optional[Dict[str, Any]]] = [None] * T

        pos_seq   = trajectory.get("positions", [])
        agents    = trajectory.get("agents", {})
        wpts      = trajectory.get("waypoints", {})  # optional
        wp_status = trajectory.get("waypoint_status", {})

        # Detection thresholds and parameters
        theta_min = self.theta_min_deg  # Minimum heading change for alert (degrees)
        v_min = self.v_min_kt           # Minimum speed change for alert (knots)
        gate_nm = max(self.los_threshold_nm, 6.0)  # Threat proximity gate (NM)
        horizon_s = self.horizon_s      # TCPA horizon for threat detection
        toward_wp_slack_deg = 1.0       # Tolerance for navigation intent filtering

        for t in range(T):
            if t >= len(pos_seq): 
                break
            pos_t = pos_seq[t]

            # Build current state lookups for this timestep
            hdg_t = {aid: float(agents.get(aid, {}).get("headings", [0.0]*T)[t]) for aid in pos_t}
            spd_t = {aid: float(agents.get(aid, {}).get("speeds", [250.0]*T)[t]) for aid in pos_t}

            # Scan all agents for alert-worthy actions
            trigger = False
            cause = None

            for aid, arr in actions_seq[t].items():
                # Skip agents who have completed their waypoints
                if wp_status.get(aid, {}).get(t, False): 
                    continue

                # Extract action magnitudes
                dpsi = float(np.asarray(arr).reshape(-1)[0]) if len(np.asarray(arr).reshape(-1))>0 else 0.0
                dvkt = float(np.asarray(arr).reshape(-1)[1]) if len(np.asarray(arr).reshape(-1))>1 else 0.0
                
                # Filter 1: Action magnitude threshold
                if abs(dpsi) < theta_min and abs(dvkt) < v_min:
                    continue

                # Filter 2: Intent-aware filtering - ignore navigation turns toward waypoint
                is_nav_turn = False
                if wpts and aid in wpts and aid in pos_t:
                    brg_wp = _bearing_deg(pos_t[aid][0], pos_t[aid][1], wpts[aid]["lat"], wpts[aid]["lon"])
                    hdg_now = hdg_t.get(aid, 0.0)
                    drift_now = abs(_angdiff(brg_wp, hdg_now))
                    drift_after = abs(_angdiff(brg_wp, (hdg_now + dpsi) % 360.0))
                    
                    # Action reduces waypoint drift - likely navigation
                    if drift_after + toward_wp_slack_deg < drift_now:
                        is_nav_turn = True
                        
                if is_nav_turn:
                    continue

                # Filter 3: Threat-aware gating - require proximate threat
                intr, tcpa, dcpa = _threat_for_agent(pos_t, hdg_t, spd_t, aid, horizon_s, self.los_threshold_nm)
                has_threat = (intr is not None) and (0.0 <= tcpa <= horizon_s) and (dcpa <= gate_nm)
                
                if not has_threat:
                    continue

                # All filters passed - trigger alert
                trigger = True
                cause = {
                    "agent": aid, "intruder": intr, "tcpa_s": float(tcpa), 
                    "dcpa_nm": float(dcpa), "dpsi_deg": float(dpsi), "dv_kt": float(dvkt)
                }
                break  # First qualifying agent triggers alert for this timestep

            alerts[t] = trigger
            meta[t] = cause

        # Apply temporal filtering: debouncing and lag expansion
        debounced = np.zeros(T, dtype=bool)
        for t in range(T):
            # N-of-M debouncing: require N detections in M-step window
            window = alerts[max(0, t - self.debounce_m + 1): t + 1]
            if np.sum(window) >= self.debounce_n:
                debounced[t] = True

        # Expand alerts with pre/post lag to capture alert periods
        final_alerts = np.zeros(T, dtype=bool)
        for t in range(T):
            if debounced[t]:
                start = max(0, t - self.lag_pre_steps)
                end = min(T, t + self.lag_post_steps + 1)
                final_alerts[start:end] = True

        return final_alerts, meta

    def _ground_truth_series(self, traj: Dict[str, Any], sep_nm: float, waypoint_status: Optional[Dict] = None) -> Tuple[np.ndarray, float, int]:
        """
        Generate ground truth conflict flags based on TCPA/DCPA calculations.
        
        Args:
            traj: Episode trajectory data
            sep_nm: Separation threshold in nautical miles
            waypoint_status: Optional waypoint completion status by agent and timestep
            
        Returns:
            Tuple of (conflict_flags_array, minimum_cpa, conflict_timestep_count)
        """
        pos_seq = traj.get("positions", [])
        agents = traj.get("agents", {})
        waypoint_status = waypoint_status or {}
        T = len(pos_seq)
        g = np.zeros(T, dtype=bool)
        min_cpa = float('inf')
        
        # Debug: count total exclusions
        total_exclusions = 0
        
        for t in range(T):
            step = pos_seq[t]
            keys = list(step.keys())
            
            # Filter out agents who have reached waypoints
            active_keys = [aid for aid in keys if not waypoint_status.get(aid, {}).get(t, False)]
            excluded_keys = [aid for aid in keys if waypoint_status.get(aid, {}).get(t, False)]
            
            if excluded_keys:
                total_exclusions += len(excluded_keys)
            
            for i in range(len(active_keys)):
                for j in range(i+1, len(active_keys)):
                    ai, aj = active_keys[i], active_keys[j]
                    lat_i, lon_i = step[ai]
                    lat_j, lon_j = step[aj]
                    # Approximate velocities from logged headings/speeds
                    hi = float(agents.get(ai, {}).get("headings", [0.0]*T)[t])
                    hj = float(agents.get(aj, {}).get("headings", [0.0]*T)[t])
                    si = float(agents.get(ai, {}).get("speeds", [250.0]*T)[t])
                    sj = float(agents.get(aj, {}).get("speeds", [250.0]*T)[t])
                    ue, un = heading_to_unit(hi)
                    vi_e, vi_n = kt_to_nms(si) * ue, kt_to_nms(si) * un
                    ue, un = heading_to_unit(hj)
                    vj_e, vj_n = kt_to_nms(sj) * ue, kt_to_nms(sj) * un

                    tcpa, dcpa = compute_tcpa_dcpa_nm(lat_i, lon_i, vi_e, vi_n,
                                                      lat_j, lon_j, vj_e, vj_n, self.horizon_s)
                    min_cpa = min(min_cpa, dcpa)
                    if dcpa < sep_nm:
                        g[t] = True
        
        num_conflict_steps = int(g.sum())
        
        # Debug output (disabled during training)
        # if total_exclusions > 0:
        #     print(f"DEBUG: Excluded {total_exclusions} agent-timesteps due to waypoint completion")
        
        return g, min_cpa, num_conflict_steps

    def _merge_runs(self, b: np.ndarray, expand_steps: int = 0):
        """Merge consecutive True values in boolean array into runs, optionally expanding each run."""
        runs = []
        T = len(b)
        i = 0
        while i < T:
            if b[i]:
                j = i + 1
                while j < T and b[j]:
                    j += 1
                # Expand run by expand_steps in both directions
                start = max(0, i - expand_steps)
                end = min(T, j + expand_steps)
                runs.append((start, end))
                i = j
            else:
                i += 1
        return runs

    def _compute_iou(self, run1: Tuple[int, int], run2: Tuple[int, int]) -> float:
        """Compute Intersection over Union (IoU) for two time intervals."""
        start1, end1 = run1
        start2, end2 = run2
        
        # Intersection
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        intersection_length = max(0, intersection_end - intersection_start)
        
        # Union
        union_start = min(start1, start2)
        union_end = max(end1, end2)
        union_length = union_end - union_start
        
        if union_length == 0:
            return 0.0
        
        return float(intersection_length) / float(union_length)

    def _match_windows(self, G: List[Tuple[int, int]], A: List[Tuple[int, int]], 
                       iou_threshold: float = 0.1) -> Tuple[List[Tuple[int, int]], set]:
        """Match ground truth windows G with alert windows A using IoU threshold.
        
        Returns:
            matches: List of (G_idx, A_idx) pairs
            used_A: Set of A indices that were matched
        """
        matches = []
        used_A = set()
        
        # Greedy matching: for each G window, find best A window by IoU
        for g_idx, g_window in enumerate(G):
            best_iou = 0.0
            best_a_idx = -1
            
            for a_idx, a_window in enumerate(A):
                if a_idx in used_A:
                    continue
                    
                iou = self._compute_iou(g_window, a_window)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_a_idx = a_idx
            
            if best_a_idx >= 0:
                matches.append((g_idx, best_a_idx))
                used_A.add(best_a_idx)
        
        return matches, used_A

    def _min_hmd_series(self, pos_seq: List[Dict[str, Tuple[float, float]]]) -> np.ndarray:
        """Compute per-step minimum horizontal miss distance series."""
        T = len(pos_seq)
        per_step_min_sep = np.zeros(T, dtype=float)
        
        for t in range(T):
            step = pos_seq[t]
            aids = list(step.keys())
            min_sep = float("inf")
            
            for i in range(len(aids)):
                for j in range(i+1, len(aids)):
                    lat_i, lon_i = step[aids[i]]
                    lat_j, lon_j = step[aids[j]]
                    d = haversine_nm(lat_i, lon_i, lat_j, lon_j)
                    min_sep = min(min_sep, d)
            
            per_step_min_sep[t] = 0.0 if min_sep == float("inf") else float(min_sep)
        
        return per_step_min_sep

    def _resolution_cm(self, trajectory: Dict[str, Any], G: List[Tuple[int, int]], 
                       matches: List[Tuple[int, int]], d_sep_nm: float, window_s: float) -> Dict[str, int]:
        """Compute resolution success/failure for matched conflict windows."""
        pos_seq = trajectory.get("positions", [])
        T = len(pos_seq)
        window_steps = int(round(window_s / max(1e-6, self.action_period_s)))
        
        tp_res = 0
        fn_res = 0
        
        for g_idx, a_idx in matches:
            g_start, g_end = G[g_idx]
            # FIXED: Check resolution in window after conflict end
            check_start = g_end  
            check_end = min(T, g_end + window_steps)
            
            # Handle edge case where conflict extends to trajectory end
            if check_start >= T:
                # Conflict extends to end - check if it was resolved in final steps
                final_window_start = max(0, T - window_steps)
                final_window = pos_seq[final_window_start:T]
                if final_window:
                    min_hmd = self._min_hmd_window(final_window)
                    if min_hmd > d_sep_nm:
                        tp_res += 1
                    else:
                        fn_res += 1
            elif check_start < T and check_end > check_start:
                # Normal case - check resolution window after conflict
                resolution_window = pos_seq[check_start:check_end]
                if resolution_window:
                    min_hmd = self._min_hmd_window(resolution_window)
                    if min_hmd > d_sep_nm:
                        tp_res += 1
                    else:
                        fn_res += 1
                else:
                    # Empty resolution window - treat as unresolved
                    fn_res += 1
            else:
                # Invalid window - treat as unresolved  
                fn_res += 1
        
        return {"tp_res": tp_res, "fn_res": fn_res}

    def _min_hmd_window(self, window_steps: List[Dict[str, Tuple[float, float]]]) -> float:
        """Minimum horizontal miss distance within a window of position dicts."""
        m = float('inf')
        for step in window_steps:
            aids = list(step.keys())
            for i in range(len(aids)):
                for j in range(i+1, len(aids)):
                    ai, aj = aids[i], aids[j]
                    lat_i, lon_i = step[ai]
                    lat_j, lon_j = step[aj]
                    d = haversine_nm(lat_i, lon_i, lat_j, lon_j)
                    if d < m:
                        m = d
        return m if m < float('inf') else 0.0

    def compute(self, trajectory: Dict[str, Any], sep_nm: float = 5.0, return_series: bool = False) -> Dict[str, Any]:
        pos_seq = trajectory.get("positions", [])
        actions_seq = trajectory.get("actions", [])
        
        # Safeguard against empty trajectories
        if not trajectory.get("positions"):
            return {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                    "ghost_conflict": 0.0, "missed_conflict": 0.0,
                    "resolution_fail_rate": 0.0, "oscillation_rate": 0.0,
                    "num_los_events": 0, "total_los_duration": 0,
                    "min_separation_nm": 0.0, "total_path_length_nm": 0.0,
                    "avg_path_length_nm": 0.0, "flight_time_s": 0.0,
                    "path_efficiency": 0.0, "waypoint_reached_ratio": 0.0,
                    "unwanted_interventions": 0, "excess_alert_time_s": 0.0,
                    "avg_lead_time_s": 0.0, "escape_improve_hits": 0}
        
        T = min(len(pos_seq), len(actions_seq))
        if T == 0:
            return {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                    "ghost_conflict": 0.0, "missed_conflict": 0.0,
                    "resolution_fail_rate": 0.0, "oscillation_rate": 0.0,
                    "num_los_events": 0, "total_los_duration": 0,
                    "min_separation_nm": 0.0, "total_path_length_nm": 0.0,
                    "avg_path_length_nm": 0.0, "flight_time_s": 0.0,
                    "path_efficiency": 0.0, "waypoint_reached_ratio": 0.0,
                    "unwanted_interventions": 0, "excess_alert_time_s": 0.0,
                    "avg_lead_time_s": 0.0, "escape_improve_hits": 0}

        # Get waypoint status from trajectory metadata if available
        waypoint_status = trajectory.get("waypoint_status", {})
        
        # FIXED: Extract waypoint completion from CSV data if available
        csv_waypoint_data = trajectory.get("csv_waypoint_data", None)
        if csv_waypoint_data:
            # Override efficiency calculation with CSV-based waypoint data
            trajectory = trajectory.copy()  # Don't modify original
            trajectory["csv_waypoint_completion"] = csv_waypoint_data
        
        g, min_cpa, num_conflict_steps = self._ground_truth_series(trajectory, sep_nm, waypoint_status)
        a, alert_meta = self._alerts_from_actions(actions_seq, trajectory)

        # Optional immediate "escape" check - CPA improvement analysis
        escape_hits = 0
        for t, cause in enumerate(alert_meta):
            if not cause: continue
            aid, intr = cause["agent"], cause["intruder"]
            if not intr: continue
            # get CPA before/after one step (10s) using headings/speeds at t and t+1
            def cpa_at(step):
                if step >= len(trajectory["positions"]): return float('inf')
                pos = trajectory["positions"][step]
                hdg = {a: float(trajectory["agents"][a]["headings"][step]) for a in pos}
                spd = {a: float(trajectory["agents"][a]["speeds"][step])   for a in pos}
                if aid not in pos or intr not in pos: return float('inf')
                lat_i, lon_i = pos[aid]; lat_j, lon_j = pos[intr]
                ei, ni = heading_to_unit(hdg[aid]); vi_e, vi_n = kt_to_nms(spd[aid]) * ei, kt_to_nms(spd[aid]) * ni
                ej, nj = heading_to_unit(hdg[intr]); vj_e, vj_n = kt_to_nms(spd[intr]) * ej, kt_to_nms(spd[intr]) * nj
                _, dc = compute_tcpa_dcpa_nm(lat_i, lon_i, vi_e, vi_n, lat_j, lon_j, vj_e, vj_n, self.horizon_s)
                return dc
            dc0 = cpa_at(t)
            dc1 = cpa_at(min(t+1, len(trajectory["positions"])-1))
            if (dc1 - dc0) > 0.3:  # ≥0.3 NM improvement next step
                escape_hits += 1

        # Event windows and IoU matching
        expand_gt_steps = int(30.0 / self.action_period_s)  # ±30s for ground truth
        expand_alert_steps = int(10.0 / self.action_period_s)  # ±10s for alerts
        
        G = self._merge_runs(g, expand_steps=expand_gt_steps)
        A = self._merge_runs(a, expand_steps=expand_alert_steps)
        
        matches, used_A = self._match_windows(G, A, iou_threshold=self.iou_threshold)

        # Event-based confusion matrix
        TP = len(matches)
        FN = len(G) - TP
        FP = len(A) - len(used_A)
        TN_steps = int(np.sum(~g & ~a))
        TN_pct = float(TN_steps) / max(1, T)

        # Calculate traditional rates for backward compatibility
        ghost = FP / max(1, FP + TN_steps) if (FP + TN_steps) > 0 else 0.0
        missed = FN / max(1, FN + TP) if (FN + TP) > 0 else 0.0

        # FIXED: Intervention counts based on action threshold exceedances (not alert events)
        # Count timesteps where any agent's action exceeds thresholds
        num_interventions_total = 0
        intervention_steps = []
        
        for t in range(T):
            has_intervention = False
            for aid, action in actions_seq[t].items():
                action_arr = np.asarray(action).reshape(-1)
                hdg_delta = abs(action_arr[0]) if len(action_arr) > 0 else 0.0
                spd_delta = abs(action_arr[1]) if len(action_arr) > 1 else 0.0
                
                if hdg_delta >= self.theta_min_deg or spd_delta >= self.v_min_kt:
                    has_intervention = True
                    break
            
            if has_intervention:
                num_interventions_total += 1
                intervention_steps.append(t)
        
        # Count interventions that match TP/TN (correct interventions)
        # Count interventions that are FP/FN (incorrect interventions)
        num_interventions_matched = 0
        num_interventions_false = 0
        
        for t in intervention_steps:
            if g[t] and a[t]:  # TP - intervention during actual conflict
                num_interventions_matched += 1
            elif not g[t] and not a[t]:  # TN - no intervention when no conflict (shouldn't happen since we're in intervention_steps)
                num_interventions_matched += 1
            else:  # FP or FN - incorrect intervention
                num_interventions_false += 1

        precision = TP / max(1, TP + FP)
        recall    = TP / max(1, TP + FN)
        f1_score  = 2*precision*recall / max(1e-9, (precision + recall))

        alert_duty_cycle   = float(np.mean(a)) if len(a) else 0.0
        total_alert_time_s = float(np.sum(a)) * self.action_period_s

        # (optional) threat-at-alert quality
        dcpa_at_alert, tcpa_at_alert = [], []
        for t, meta in enumerate(alert_meta):
            if a[t] and meta:
                dcpa_at_alert.append(float(meta["dcpa_nm"]))
                tcpa_at_alert.append(float(meta["tcpa_s"]))
        avg_alert_dcpa_nm = float(np.mean(dcpa_at_alert)) if dcpa_at_alert else 0.0
        avg_alert_tcpa_s  = float(np.mean(tcpa_at_alert))  if tcpa_at_alert  else 0.0

        # Resolution efficiency using matched windows
        res_metrics = self._resolution_cm(trajectory, G, matches, d_sep_nm=sep_nm, window_s=self.res_window_s)
        res_fail_rate = res_metrics["fn_res"] / max(1, res_metrics["tp_res"] + res_metrics["fn_res"])

        # Unwanted interventions: alerts that never matched any conflict
        unwanted_interventions = len(A) - len(used_A)
        excess_alert_time = 0.0
        for a_idx, (start, end) in enumerate(A):
            if a_idx not in used_A:
                excess_alert_time += (end - start) * self.action_period_s

        # Lead time analysis for matched windows
        lead_times = []
        for g_idx, a_idx in matches:
            g_start, _ = G[g_idx]
            a_start, _ = A[a_idx]
            lead_time_s = (a_start - g_start) * self.action_period_s
            lead_times.append(lead_time_s)
        
        avg_lead_time = float(np.mean(lead_times)) if lead_times else 0.0

        # FIXED: Action oscillation based on max action value exceedances (not sign flips)
        # Count timesteps where any agent's action exceeds maximum values
        max_hdg_deg = 18.0  # Maximum reasonable heading change (degrees)
        max_spd_kt = 10.0   # Maximum reasonable speed change (knots)
        
        oscillations = 0
        for t in range(T):
            has_extreme_action = False
            for aid, arr in actions_seq[t].items():
                arr = np.asarray(arr).reshape(-1)
                hdg_delta = abs(arr[0]) if len(arr) > 0 else 0.0
                spd_delta = abs(arr[1]) if len(arr) > 1 else 0.0
                
                # Check if either heading or speed change exceeds maximum values
                if hdg_delta > max_hdg_deg or spd_delta > max_spd_kt:
                    has_extreme_action = True
                    break
            
            if has_extreme_action:
                oscillations += 1
                
        oscillation_rate = oscillations / max(1, T)

        # Enhanced metrics
        los_metrics = self._detect_los_events(pos_seq)
        efficiency_metrics = self._calculate_efficiency_metrics(trajectory)

        # Combine all metrics
        result = {
            "tp": TP, "fp": FP, "fn": FN, "tn": TN_steps,
            "tn_pct": TN_pct,
            "ghost_conflict": float(ghost),
            "missed_conflict": float(missed),
            "resolution_fail_rate": float(res_fail_rate),
            "oscillation_rate": float(oscillation_rate),
            "min_CPA_nm": float(min_cpa),
            "num_conflict_steps": int(num_conflict_steps),
            "unwanted_interventions": int(unwanted_interventions),
            "excess_alert_time_s": float(excess_alert_time),
            "avg_lead_time_s": float(avg_lead_time),
            "escape_improve_hits": int(escape_hits),
            # NEW: Intervention and detection quality metrics
            "num_interventions": int(num_interventions_total),
            "num_interventions_matched": int(num_interventions_matched),
            "num_interventions_false": int(num_interventions_false),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "alert_duty_cycle": float(alert_duty_cycle),
            "total_alert_time_s": float(total_alert_time_s),
            "avg_alert_dcpa_nm": float(avg_alert_dcpa_nm),
            "avg_alert_tcpa_s": float(avg_alert_tcpa_s),
        }
        
        # Add LOS metrics
        result.update(los_metrics)
        
        # Add efficiency metrics
        result.update(efficiency_metrics)
        
        # Optionally return per-step series (for visualization)
        if return_series:
            # Per-step masks for visualization
            TP_mask = a & g
            FP_mask = a & ~g
            FN_mask = ~a & g
            TN_mask = ~a & ~g
            
            # Per-step min separation for overlays
            per_step_min_sep = self._min_hmd_series(pos_seq)

            result["series"] = {
                "gt_conflict": g.astype(int).tolist(),
                "alert": a.astype(int).tolist(),
                "tp": TP_mask.astype(int).tolist(),
                "fp": FP_mask.astype(int).tolist(),
                "fn": FN_mask.astype(int).tolist(),
                "tn": TN_mask.astype(int).tolist(),
                "min_separation_nm": per_step_min_sep.tolist(),
            }
        
        return result