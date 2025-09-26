"""
Enhanced hallucination detector computing confusion matrices from TCPA/DCPA and action magnitudes.
Includes LOS event tracking and efficiency metrics.

- Ground truth: conflict if any pair's DCPA < 5 NM within 300 s (per timestep flag).
- Prediction proxy: "alert" when any agent issues a large action (|dpsi|>3 deg OR |dv|>5 kt).
- Resolution: after an alert, did min HMD exceed 5 NM within 60 s?
- LOS Events: Track when aircraft separation drops below threshold
- Efficiency: Calculate path lengths, flight times, and completion metrics

Outputs a pandas-friendly dict per episode.
"""

import math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np


def haversine_nm(lat1, lon1, lat2, lon2):
    R_nm = 3440.065
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2.0) ** 2)
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return R_nm * c

def heading_to_unit(heading_deg: float) -> Tuple[float, float]:
    rad = math.radians(90.0 - heading_deg)
    return (math.cos(rad), math.sin(rad))

def kt_to_nms(kt: float) -> float:
    return kt / 3600.0


def compute_tcpa_dcpa_nm(lat_i, lon_i, vi_east_nms, vi_north_nms,
                         lat_j, lon_j, vj_east_nms, vj_north_nms,
                         horizon_s: float = 300.0) -> Tuple[float, float]:
    dx_nm = haversine_nm(lat_i, lon_i, lat_i, lon_j) * (1 if lon_j >= lon_i else -1)
    dy_nm = haversine_nm(lat_i, lon_i, lat_j, lon_i) * (1 if lat_j >= lat_i else -1)

    rx, ry = dx_nm, dy_nm
    vx, vy = (vj_east_nms - vi_east_nms), (vj_north_nms - vi_north_nms)
    vv = vx*vx + vy*vy
    if vv < 1e-12:
        return 0.0, math.hypot(rx, ry)
    tcpa = -(rx*vx + ry*vy) / vv
    tcpa = max(0.0, min(horizon_s, tcpa))
    dcpa = math.hypot(rx + vx*tcpa, ry + vy*tcpa)
    return tcpa, dcpa


class HallucinationDetector:
    def __init__(self, horizon_s: float = 180.0, action_thresh=(3.0, 5.0),
                 res_window_s: float = 90.0, action_period_s: float = 10.0,
                 los_threshold_nm: float = 5.0, lag_pre_steps: int = 1,
                 lag_post_steps: int = 1, debounce_n: int = 2, debounce_m: int = 3,
                 iou_threshold: float = 0.1):
        self.horizon_s = float(horizon_s)  # Reduced from 300 to 180 to reduce "always in future" flags
        self.theta_min_deg = float(action_thresh[0])
        self.v_min_kt = float(action_thresh[1])
        self.res_window_s = float(res_window_s)  # Increased to 90s for better resolution verification
        self.action_period_s = float(action_period_s)  # Should match environment timestep
        self.los_threshold_nm = float(los_threshold_nm)
        self.lag_pre_steps = int(lag_pre_steps)
        self.lag_post_steps = int(lag_post_steps)
        self.debounce_n = int(debounce_n)  # 2-of-3 timesteps
        self.debounce_m = int(debounce_m)
        self.iou_threshold = float(iou_threshold)

    def _detect_los_events(self, pos_seq: List[Dict[str, Tuple[float, float]]]) -> Dict[str, Any]:
        """Detect Loss of Separation (LOS) events between aircraft pairs."""
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
        """Calculate efficiency metrics: path length, time to completion, etc."""
        pos_seq = trajectory.get("positions", [])
        timestamps = trajectory.get("timestamps", [])
        scenario_meta = trajectory.get("scenario_metadata", {})
        
        if not pos_seq or not timestamps:
            return {
                'total_path_length_nm': 0.0,
                'avg_path_length_nm': 0.0,
                'flight_time_s': 0.0,
                'path_efficiency': 0.0,
                'waypoint_reached_ratio': 0.0
            }
        
        # Calculate path lengths for each aircraft
        agent_path_lengths = {}
        waypoints_reached = 0
        total_agents = 0
        
        for aid in pos_seq[0].keys():
            total_agents += 1
            path_length = 0.0
            agent_positions = []
            
            # Extract positions for this agent
            for step in pos_seq:
                if aid in step:
                    agent_positions.append(step[aid])
            
            # Calculate cumulative path length
            for i in range(1, len(agent_positions)):
                lat1, lon1 = agent_positions[i-1]
                lat2, lon2 = agent_positions[i]
                segment_length = haversine_nm(lat1, lon1, lat2, lon2)
                path_length += segment_length
            
            agent_path_lengths[aid] = path_length
            
            # Check if agent reached waypoint (simple heuristic - if final position is close to waypoint)
            if len(agent_positions) > 0:
                final_lat, final_lon = agent_positions[-1]
                # Note: Would need waypoint coordinates from scenario to calculate exact efficiency
                # For now, assume completion if flight lasted reasonable time
                if len(timestamps) > 50:  # Arbitrary threshold for "completion"
                    waypoints_reached += 1
        
        total_path_length = sum(agent_path_lengths.values())
        avg_path_length = total_path_length / max(1, total_agents)
        flight_time = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        waypoint_reached_ratio = waypoints_reached / max(1, total_agents)
        
        # Path efficiency could be calculated as direct distance / actual path length
        # For now, use a simple metric based on flight time vs expected time
        expected_time = 300.0  # seconds, based on typical scenario duration
        path_efficiency = min(1.0, expected_time / max(1.0, flight_time))
        
        return {
            'total_path_length_nm': float(total_path_length),
            'avg_path_length_nm': float(avg_path_length),
            'flight_time_s': float(flight_time),
            'path_efficiency': float(path_efficiency),
            'waypoint_reached_ratio': float(waypoint_reached_ratio)
        }

    def _alerts_from_actions(self, actions_seq: List[Dict[str, np.ndarray]], waypoint_status: Optional[Dict] = None) -> np.ndarray:
        T = len(actions_seq)
        raw_alerts = np.zeros(T, dtype=bool)
        waypoint_status = waypoint_status or {}
        
        # First pass: detect raw action threshold crossings
        for t in range(T):
            acts = actions_seq[t]
            for aid, arr in acts.items():
                # Skip agents who have reached their waypoint
                if waypoint_status.get(aid, {}).get(t, False):
                    continue
                    
                arr = np.asarray(arr).reshape(-1)
                dpsi = float(arr[0]) if arr.size > 0 else 0.0
                dvkt = float(arr[1]) if arr.size > 1 else 0.0
                if abs(dpsi) >= self.theta_min_deg or abs(dvkt) >= self.v_min_kt:
                    raw_alerts[t] = True
                    break
        
        # Apply debounce: 2-of-3 timesteps within sliding window
        debounced = np.zeros(T, dtype=bool)
        for t in range(T):
            window_start = max(0, t - self.debounce_m + 1)
            window_end = min(T, t + 1)
            window_alerts = raw_alerts[window_start:window_end]
            if np.sum(window_alerts) >= self.debounce_n:
                debounced[t] = True
        
        # Apply pre/post lag expansion
        final_alerts = np.zeros(T, dtype=bool)
        for t in range(T):
            if debounced[t]:
                start = max(0, t - self.lag_pre_steps)
                end = min(T, t + self.lag_post_steps + 1)
                final_alerts[start:end] = True
        
        return final_alerts

    def _ground_truth_series(self, traj: Dict[str, Any], sep_nm: float, waypoint_status: Optional[Dict] = None) -> Tuple[np.ndarray, float, int]:
        """Return g[t] ∈ {0,1}, min CPA, and count of conflict timesteps, excluding agents who reached waypoints."""
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
            # Check resolution in window after conflict end
            check_start = g_end
            check_end = min(T, g_end + window_steps)
            
            if check_start < T:
                min_hmd = self._min_hmd_window(pos_seq[check_start:check_end])
                if min_hmd > d_sep_nm:
                    tp_res += 1
                else:
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
                    "avg_lead_time_s": 0.0}
        
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
                    "avg_lead_time_s": 0.0}

        # Get waypoint status from trajectory metadata if available
        waypoint_status = trajectory.get("waypoint_status", {})
        
        g, min_cpa, num_conflict_steps = self._ground_truth_series(trajectory, sep_nm, waypoint_status)
        a = self._alerts_from_actions(actions_seq, waypoint_status)

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

        # Action oscillation: fraction of timesteps where heading command flips sign (per episode, max over agents)
        flips = 0
        prev_sign = None
        for t in range(T):
            # Use average sign across agents at step t
            sgns = []
            for aid, arr in actions_seq[t].items():
                arr = np.asarray(arr).reshape(-1)
                sgns.append(np.sign(arr[0]) if arr.size > 0 else 0.0)
            s = np.sign(np.mean(sgns)) if sgns else 0.0
            if prev_sign is not None and s * prev_sign < 0:
                flips += 1
            if s != 0:
                prev_sign = s
        oscillation_rate = flips / max(1, T-1)

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