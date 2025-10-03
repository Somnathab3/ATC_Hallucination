# src/testing/baseline_vs_shift_matrix.py
import os, re, json, argparse, pathlib, time
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ray
from ray.rllib.algorithms.algorithm import Algorithm

# --- Reuse your stable env registration (path-healing) ---
# NOTE: this function exists in your repo already.
from src.testing.targeted_shift_tester import make_env   # :contentReference[oaicite:3]{index=3}
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# Try both import locations for your detector
try:
    from src.analysis.hallucination_detector_enhanced import HallucinationDetector
except Exception:
    from hallucination_detector_enhanced import HallucinationDetector  # fallback

# Your custom env (wrapped by make_env internally)
from src.environment.marl_collision_env_minimal import MARLCollisionEnv

# Import trajectory visualization modules
try:
    from src.analysis.trajectory_comparison_plot import create_trajectory_comparison_plot, create_shift_analysis_dashboard
except ImportError:
    print("Warning: trajectory_comparison_plot module not available")
    create_trajectory_comparison_plot = None
    create_shift_analysis_dashboard = None

try:
    from src.analysis.trajectory_comparison_map import create_comparison_map, generate_shift_comparison_maps
except ImportError:
    print("Warning: trajectory_comparison_map module not available")
    create_comparison_map = None
    generate_shift_comparison_maps = None

ENV_NAME = "marl_collision_env_v0"
DEFAULT_SCENARIOS = ["head_on","parallel","t_formation","converging","canonical_crossing"]

# ------------------ utils ------------------
def abspath(p): return str(pathlib.Path(p).expanduser().resolve())

def discover_scenarios(scen_dir: str) -> Dict[str,str]:
    d = {}
    for p in pathlib.Path(scen_dir).glob("*.json"):
        d[p.stem] = abspath(p)
    return d

def parse_baseline_scenario_from_ckpt(ckpt_path: str) -> Optional[str]:
    """
    Extract training scenario from folder name like:
    models/PPO_canonical_crossing_20250924_225408
    """
    base = os.path.basename(ckpt_path.rstrip(r"\/"))
    m = re.search(r"PPO_([a-z_]+)_\d{8}", base, re.IGNORECASE)
    return (m.group(1) if m else None)

def list_checkpoints_from_dir(models_dir: str) -> Dict[str,str]:
    out = {}
    for p in pathlib.Path(models_dir).glob("PPO_*"):
        if p.is_dir():
            out[p.name] = abspath(str(p))
    return out

def register_env_once():
    try:
        register_env(ENV_NAME, lambda cfg: make_env(cfg))   # :contentReference[oaicite:4]{index=4}
    except Exception:
        # ignore re-register attempts
        pass

def init_ray(use_gpu=False):
    """Initialize Ray with GPU support for accelerated testing."""
    if not ray.is_initialized():
        init_kwargs = {
            "ignore_reinit_error": True, 
            "log_to_driver": False,
            "configure_logging": False,
        }
        
        if use_gpu:
            print("🚀 Initializing Ray with GPU support for testing")
            # Let Ray auto-detect GPUs, don't force local mode
        else:
            print("🔧 Initializing Ray in CPU mode")
            init_kwargs["local_mode"] = True
            
        ray.init(**init_kwargs)

def load_algo(ckpt: str, use_gpu: bool = False) -> Tuple[Algorithm, str]:
    """
    Load the Algorithm exactly as trained; pick 'shared_policy' if present.
    """
    init_ray(use_gpu=use_gpu)
    register_env_once()
    algo = Algorithm.from_checkpoint(abspath(ckpt))  # :contentReference[oaicite:5]{index=5}
    
    # Note: GPU configuration is handled during training, not inference
    # Ray will automatically use available GPUs for workers if available
    if use_gpu:
        print(f"🚀 Using GPU-accelerated Ray workers for faster evaluation")
    
    # prefer your shared policy id from training
    policy_id = "shared_policy"
    try:
        algo.get_policy(policy_id)
    except Exception:
        # fall back to the first available policy
        worker = getattr(algo, "workers", None)
        local = worker.local_worker() if worker and hasattr(worker, "local_worker") else None
        policy_id = list(local.policy_map.keys())[0] if local else "default_policy"
    return algo, policy_id

def build_env(scenario_json: str, results_dir: str, seed: int = 123):
    cfg = {
        "scenario_path": abspath(scenario_json),
        "results_dir": abspath(results_dir),
        "enable_hallucination_detection": True,
        "log_trajectories": True,
        "seed": seed,
        # IMPORTANT: keep the SAME obs/reward settings as training to avoid space mismatches.
        "neighbor_topk": 3,
        "collision_nm": 3.0,
    }
    # make_env already returns ParallelPettingZooEnv(MARLCollisionEnv(cfg)) with path healing
    return make_env(cfg)

def compute_once(algo: Algorithm, policy_id: str, env: ParallelPettingZooEnv):
    obs, _ = env.reset()
    done = False
    while not done:
        actions = {}
        for aid in env.agents:
            if aid in obs:
                a = algo.compute_single_action(obs[aid], explore=True, policy_id=policy_id)  # :contentReference[oaicite:6]{index=6}
                actions[aid] = np.asarray(a, dtype=np.float32)
            else:
                actions[aid] = np.array([0.0, 0.0], dtype=np.float32)
        next_obs, rewards, term, trunc, infos = env.step(actions)
        obs = next_obs
        done = (term and all(term.values())) or (trunc and all(trunc.values()))
    env.close()

def latest_traj_csv(run_dir: pathlib.Path) -> Optional[str]:
    cands = list(run_dir.glob("traj_*.csv"))
    return str(max(cands, key=lambda p: p.stat().st_mtime)) if cands else None

def csv_to_trajectory(csv_path: str) -> Dict:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["step_idx","agent_id"])
    agents = sorted(df["agent_id"].astype(str).unique().tolist())
    steps = sorted(df["step_idx"].unique().tolist())

    pos, acts, ts = [], [], []
    headings = {a: [] for a in agents}
    speeds   = {a: [] for a in agents}

    # column fallbacks
    hdg_col = "hdg_deg" if "hdg_deg" in df.columns else ("hdg" if "hdg" in df.columns else None)
    spd_col = "tas_kt"  if "tas_kt" in df.columns else ("tas" if "tas" in df.columns else None)

    for t in steps:
        sdf = df[df.step_idx == t]
        ts.append(float(sdf["sim_time_s"].iloc[0]) if "sim_time_s" in sdf.columns else float(t)*10.0)
        p_t, a_t = {}, {}
        for _, r in sdf.iterrows():
            aid = str(r["agent_id"])
            p_t[aid] = (float(r["lat_deg"]), float(r["lon_deg"]))
            if hdg_col: headings[aid].append(float(r[hdg_col]))
            if spd_col: speeds[aid].append(float(r[spd_col]) if "tas_kt" in df else float(r[spd_col])*1.94384)
            # logged physical deltas (deg/kt) → keep as-is
            if "action_hdg_delta_deg" in r and "action_spd_delta_kt" in r:
                a_t[aid] = [float(r["action_hdg_delta_deg"]), float(r["action_spd_delta_kt"])]
            else:
                a_t[aid] = [0.0, 0.0]
        pos.append(p_t)
        acts.append(a_t)

    return {
        "positions": pos,
        "actions": acts,
        "timestamps": ts,
        "agents": {aid: {"headings": headings[aid], "speeds": speeds[aid]} for aid in agents},
        "scenario_metadata": {"traj_csv": abspath(csv_path), "num_agents": len(agents), "num_steps": len(steps)}
    }

def metrics_from_csv(csv_path: str, sep_nm: float = 5.0) -> Dict[str, float]:
    traj = csv_to_trajectory(csv_path)
    hd = HallucinationDetector(action_thresh=(3.0, 5.0), horizon_s=120.0, res_window_s=60.0, action_period_s=10.0)
    m = hd.compute(traj, sep_nm=sep_nm, return_series=False)
    
    # SAFETY: fix key name (was "min_cpa_nm")
    minsep = m.get("min_CPA_nm", m.get("min_cpa_nm", 0.0))
    
    # If detector didn't yet provide P/R/F1 (older file), compute fallbacks
    tp, fp, fn = m.get("tp", 0), m.get("fp", 0), m.get("fn", 0)
    prec = m.get("precision", tp / max(1, tp + fp))
    rec  = m.get("recall",    tp / max(1, tp + fn))
    f1   = m.get("f1_score",  2*prec*rec / max(1e-9, (prec + rec)))
    
    return {
        # Safety
        "min_separation_nm": minsep,
        "num_los_events": m.get("num_los_events", 0),
        "total_los_duration": m.get("total_los_duration", 0.0),
        # Hallucination
        "precision": prec, 
        "recall": rec, 
        "f1_score": f1,
        "alert_duty_cycle": m.get("alert_duty_cycle", 0.0),
        # Performance
        "path_efficiency": m.get("path_efficiency", 0.0),
        "flight_time_s": m.get("flight_time_s", 0.0),
        "waypoint_reached_ratio": m.get("waypoint_reached_ratio", 0.0),
        # (optionally pull the new extras for shift deltas)
        "total_extra_path_nm": m.get("total_extra_path_nm", 0.0)
    }


def extract_comprehensive_metrics_from_csv(csv_path: str, sep_nm: float = 5.0) -> Dict[str, float]:
    """
    Extract comprehensive metrics from trajectory CSV using the same approach as targeted_shift_tester.
    This mirrors the exact metric extraction process to ensure consistency.
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return _get_default_metrics()
        
        # Collapse per-step data for episode-level metrics (like targeted shift tester)
        step_collapsed = df.groupby('step_idx').agg({
            'min_separation_nm': 'min',
            'conflict_flag': 'max' if 'conflict_flag' in df.columns else lambda x: 0,
            'predicted_alert': 'max' if 'predicted_alert' in df.columns else lambda x: 0,
            'tp': 'sum' if 'tp' in df.columns else lambda x: 0,
            'fp': 'sum' if 'fp' in df.columns else lambda x: 0,
            'fn': 'sum' if 'fn' in df.columns else lambda x: 0,
            'tn': 'sum' if 'tn' in df.columns else lambda x: 0,
        }).reset_index()
        
        # Basic confusion matrix metrics
        tp_sum = int(step_collapsed['tp'].sum()) if 'tp' in step_collapsed.columns else 0
        fp_sum = int(step_collapsed['fp'].sum()) if 'fp' in step_collapsed.columns else 0
        fn_sum = int(step_collapsed['fn'].sum()) if 'fn' in step_collapsed.columns else 0
        tn_sum = int(step_collapsed['tn'].sum()) if 'tn' in step_collapsed.columns else 0
        total_steps = len(step_collapsed)
        
        # Precision, recall, F1
        precision = tp_sum / max(1, tp_sum + fp_sum)
        recall = tp_sum / max(1, tp_sum + fn_sum)
        f1_score = 2 * precision * recall / max(1e-9, precision + recall)
        
        # Flight time calculation
        if 'timestamp' in df.columns:
            flight_time_s = float(df['timestamp'].max() - df['timestamp'].min())
        else:
            flight_time_s = float(len(df) * 10.0)  # Assume 10s per step
        
        # Loss of Separation events
        num_los_events = 0
        total_los_duration = 0.0
        if 'min_separation_nm' in df.columns:
            los_mask = df['min_separation_nm'] < sep_nm
            if los_mask.any():
                # Count LOS transitions
                los_transitions = los_mask.astype(int).diff().fillna(0)
                num_los_events = int((los_transitions == 1).sum())
                total_los_duration = float(los_mask.sum() * 10.0)  # 10s per step
        
        # Path metrics calculation
        try:
            # Calculate path lengths per agent
            agent_path_lengths = {}
            agent_extra_nm = {}
            agent_extra_ratio = {}
            
            for agent_id in df['agent_id'].unique():
                agent_df = df[df['agent_id'] == agent_id].sort_values('step_idx')
                if len(agent_df) >= 2:
                    # Calculate path length using haversine distance
                    path_length = 0.0
                    for i in range(1, len(agent_df)):
                        lat1, lon1 = agent_df.iloc[i-1]['lat_deg'], agent_df.iloc[i-1]['lon_deg']
                        lat2, lon2 = agent_df.iloc[i]['lat_deg'], agent_df.iloc[i]['lon_deg']
                        dist = haversine_nm(lat1, lon1, lat2, lon2)
                        path_length += dist
                    
                    agent_path_lengths[agent_id] = path_length
                    
                    # Calculate direct distance to waypoint (if available)
                    if 'wp_lat' in agent_df.columns and 'wp_lon' in agent_df.columns:
                        start_lat, start_lon = agent_df.iloc[0]['lat_deg'], agent_df.iloc[0]['lon_deg']
                        wp_lat, wp_lon = agent_df.iloc[0]['wp_lat'], agent_df.iloc[0]['wp_lon']
                        direct_dist = haversine_nm(start_lat, start_lon, wp_lat, wp_lon)
                        extra_nm = max(0, path_length - direct_dist)
                        agent_extra_nm[agent_id] = extra_nm
                        agent_extra_ratio[agent_id] = extra_nm / max(1.0, direct_dist)
                    else:
                        agent_extra_nm[agent_id] = 0.0
                        agent_extra_ratio[agent_id] = 0.0
                else:
                    agent_path_lengths[agent_id] = 0.0
                    agent_extra_nm[agent_id] = 0.0
                    agent_extra_ratio[agent_id] = 0.0
            
            total_path_length_nm = sum(agent_path_lengths.values())
            total_extra_path_nm = sum(agent_extra_nm.values())
            avg_extra_path_nm = total_extra_path_nm / max(1, len(agent_extra_nm))
            avg_extra_path_ratio = sum(agent_extra_ratio.values()) / max(1, len(agent_extra_ratio))
            
            # Path efficiency: based on time vs expected time
            expected_time = 300.0  # seconds
            path_efficiency = min(1.0, expected_time / max(1.0, flight_time_s))
            
        except Exception:
            total_path_length_nm = 0.0
            total_extra_path_nm = 0.0
            avg_extra_path_nm = 0.0
            avg_extra_path_ratio = 0.0
            path_efficiency = 1.0
        
        # Waypoint reached ratio
        try:
            if 'waypoint_reached' in df.columns:
                waypoint_reached_ratio = float(df['waypoint_reached'].max())
            else:
                waypoint_reached_ratio = 0.0
        except Exception:
            waypoint_reached_ratio = 0.0
        
        # Alert timing and burden metrics
        try:
            if 'predicted_alert' in df.columns:
                alert_steps = df['predicted_alert'].sum()
                alert_duty_cycle = float(alert_steps / max(1, len(df)))
                alerts_per_min = float(alert_steps * 6.0)  # 6 = 60s/10s per step
                total_alert_time_s = float(alert_steps * 10.0)
            else:
                alert_duty_cycle = 0.0
                alerts_per_min = 0.0
                total_alert_time_s = 0.0
            
            # Lead time calculation (simplified)
            avg_lead_time_s = 0.0  # Would need more complex calculation
            
        except Exception:
            alert_duty_cycle = 0.0
            alerts_per_min = 0.0
            total_alert_time_s = 0.0
            avg_lead_time_s = 0.0
        
        # Intervention metrics (simplified for now)
        num_interventions = tp_sum + fp_sum  # Total alerts as interventions
        num_interventions_matched = tp_sum   # True positives as matched
        num_interventions_false = fp_sum     # False positives as false interventions
        
        # NEW: Total reward calculation
        try:
            if 'reward' in df.columns:
                reward_total = float(df['reward'].sum())
            else:
                reward_total = 0.0
        except Exception:
            reward_total = 0.0
        
        # Return the same structure as targeted_shift_tester
        return {
            "tp": tp_sum,
            "fp": fp_sum,
            "fn": fn_sum,
            "tn": tn_sum,
            "min_separation_nm": float(step_collapsed['min_separation_nm'].min()) if not step_collapsed.empty else 200.0,
            "num_conflict_steps": int(step_collapsed['conflict_flag'].sum()) if 'conflict_flag' in step_collapsed.columns else 0,
            "flight_time_s": flight_time_s,
            "num_los_events": num_los_events,
            "total_los_duration": total_los_duration,
            "total_path_length_nm": total_path_length_nm,
            "path_efficiency": path_efficiency,
            "waypoint_reached_ratio": waypoint_reached_ratio,
            "total_extra_path_nm": total_extra_path_nm,
            "avg_extra_path_nm": avg_extra_path_nm,
            "avg_extra_path_ratio": avg_extra_path_ratio,
            "num_interventions": num_interventions,
            "num_interventions_matched": num_interventions_matched,
            "num_interventions_false": num_interventions_false,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "alert_duty_cycle": alert_duty_cycle,
            "alerts_per_min": alerts_per_min,
            "total_alert_time_s": total_alert_time_s,
            "avg_lead_time_s": avg_lead_time_s,
            "ghost_conflict": fp_sum / max(1, total_steps) if total_steps > 0 else 0.0,
            "missed_conflict": fn_sum / max(1, tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0,
            "resolution_fail_rate": 0.0,  # Would need more complex calculation
            "oscillation_rate": 0.0,  # Would need action sequence analysis
            "reward_total": reward_total,  # NEW: Total reward from trajectory
        }
        
    except Exception as e:
        print(f"Warning: Failed to extract comprehensive metrics from {csv_path}: {e}")
        return _get_default_metrics()


def _get_default_metrics() -> Dict[str, float]:
    """Return default metrics structure matching targeted_shift_tester."""
    return {
        "tp": 0, "fp": 0, "fn": 0, "tn": 0,
        "min_separation_nm": 200.0, "num_conflict_steps": 0,
        "flight_time_s": 0.0,
        "num_los_events": 0,
        "total_los_duration": 0.0,
        "total_path_length_nm": 0.0,
        "path_efficiency": 1.0,
        "waypoint_reached_ratio": 0.0,
        "total_extra_path_nm": 0.0,
        "avg_extra_path_nm": 0.0,
        "avg_extra_path_ratio": 0.0,
        "num_interventions": 0,
        "num_interventions_matched": 0,
        "num_interventions_false": 0,
        "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
        "alert_duty_cycle": 0.0, "alerts_per_min": 0.0, "total_alert_time_s": 0.0,
        "avg_lead_time_s": 0.0,
        "ghost_conflict": 0.0, "missed_conflict": 1.0,
        "resolution_fail_rate": 0.0, "oscillation_rate": 0.0,
        "reward_total": 0.0,  # NEW: Total reward from trajectory
    }


def haversine_nm(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points in nautical miles."""
    R_nm = 3440.065  # Earth radius in nautical miles
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat/2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon/2.0) ** 2)
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return R_nm * c

def run_model_on_scenario(algo, policy_id, scenario_json, out_dir, episodes=1, seed=123, show_progress=True) -> pd.DataFrame:
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    
    if show_progress and episodes > 1:
        print(f"    Running {episodes} episodes...", end="", flush=True)
    
    for ep in range(episodes):
        if show_progress and episodes > 3:
            if ep % max(1, episodes // 5) == 0:  # Show progress every 20%
                print(f" {ep+1}/{episodes}", end="", flush=True)
        
        run_dir = out_dir / f"ep_{ep+1:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        env = build_env(scenario_json, str(run_dir), seed=seed+ep)
        compute_once(algo, policy_id, env)
        csv_path = latest_traj_csv(run_dir)
        if csv_path:
            # Use comprehensive metrics extraction (same as targeted_shift_tester)
            m = extract_comprehensive_metrics_from_csv(csv_path)
            # Create a new dict to avoid type issues
            row_data = dict(m)  # copy metrics
            row_data["traj_csv"] = csv_path
            row_data["episode"] = ep+1
            rows.append(row_data)
    return pd.DataFrame(rows)

def pct_delta(val, base):
    if base is None or np.isnan(base) or base == 0: return np.nan
    return 100.0 * (val - base) / base

def plot_overlay(csv_path: str, out_png: str, title: str):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(6,6))
    for aid, sdf in df.groupby("agent_id"):
        plt.plot(sdf["lon_deg"].values, sdf["lat_deg"].values, label=str(aid))
        hits = sdf[sdf.get("waypoint_reached", 0) == 1]
        if not hits.empty: plt.scatter(hits["lon_deg"], hits["lat_deg"], marker="x")
    plt.xlabel("Longitude [deg]"); plt.ylabel("Latitude [deg]"); plt.title(title); plt.legend(fontsize=8); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()

def plot_minsep(csv_path: str, out_png: str, title: str):
    df = pd.read_csv(csv_path)
    g = df.groupby("step_idx")["min_separation_nm"].min().reset_index() if "min_separation_nm" in df else None
    if g is None or g.empty: return
    plt.figure(figsize=(7,3))
    plt.plot(g["step_idx"].values, g["min_separation_nm"].values, linewidth=1.5)
    plt.axhline(5.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Step"); plt.ylabel("Min sep [NM]"); plt.title(title); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()

def generate_scenario_centric_visualizations(results_data: Dict, scen_map: Dict[str, str], outdir: pathlib.Path):
    """
    Generate scenario-centric visualizations showing how all models perform on each scenario.
    
    Args:
        results_data: Dict with structure {scenario: {model: {'type': 'baseline'/'shift', 'csv_path': str}}}
        scen_map: Dict mapping scenario names to scenario JSON paths
        outdir: Output directory for visualizations
    """
    print(f"\n🎨 Generating Scenario-Centric Visualizations")
    print("=" * 60)
    
    # Create master scenario visualization directory
    scenario_viz_dir = outdir / "scenario_centric_visualizations"
    scenario_viz_dir.mkdir(parents=True, exist_ok=True)
    
    for scenario, model_data in results_data.items():
        print(f"\n📊 Creating visualizations for scenario: {scenario}")
        
        # Create scenario-specific directory
        scn_dir = scenario_viz_dir / f"scenario_{scenario}_analysis"
        scn_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate baseline and shift models
        baseline_csvs = {}
        shift_csvs = {}
        
        for model, info in model_data.items():
            if info['type'] == 'baseline':
                baseline_csvs[model] = info['csv_path']
                print(f"  🏠 Baseline: {model} (trained on {scenario})")
            else:
                shift_csvs[model] = info['csv_path']
                print(f"  🔄 Shift: {model} (trained elsewhere)")
        
        # Generate combined visualization if we have both baseline and shift data
        if baseline_csvs and shift_csvs and create_trajectory_comparison_plot is not None:
            # Create combined trajectory plot
            all_trajectories = {}
            all_trajectories.update({f"{model}_baseline": csv for model, csv in baseline_csvs.items()})
            all_trajectories.update({f"{model}_shift": csv for model, csv in shift_csvs.items()})
            
            combined_plot_file = scn_dir / f"scenario_{scenario}_all_models_comparison.html"
            combined_title = f"Scenario Analysis: {scenario.title()} - All Models (Baseline vs Shift)"
            
            print(f"    📈 Creating combined plot: {combined_plot_file.name}")
            
            success = create_trajectory_comparison_plot(
                baseline_csv=list(baseline_csvs.values())[0],  # Use first baseline as reference
                shift_csvs=all_trajectories,
                out_html=str(combined_plot_file),
                title=combined_title,
                scenario_path=scen_map.get(scenario)
            )
            
            if success:
                print(f"      ✅ Generated: {combined_plot_file.name}")
            
            # Generate individual baseline vs shift comparisons
            for baseline_model, baseline_csv in baseline_csvs.items():
                for shift_model, shift_csv in shift_csvs.items():
                    comparison_file = scn_dir / f"scenario_{scenario}_{baseline_model}_vs_{shift_model}.html"
                    comparison_title = f"Scenario {scenario.title()}: {baseline_model} (Baseline) vs {shift_model} (Shift)"
                    
                    success = create_trajectory_comparison_plot(
                        baseline_csv=baseline_csv,
                        shift_csvs={f"{shift_model}_shift": shift_csv},
                        out_html=str(comparison_file),
                        title=comparison_title,
                        scenario_path=scen_map.get(scenario)
                    )
                    
                    if success:
                        print(f"      ✅ Generated: {comparison_file.name}")
        
        # Generate maps if available
        if baseline_csvs and shift_csvs and create_comparison_map is not None:
            print(f"    🗺️  Creating trajectory maps...")
            
            for baseline_model, baseline_csv in baseline_csvs.items():
                for shift_model, shift_csv in shift_csvs.items():
                    map_file = scn_dir / f"scenario_{scenario}_{baseline_model}_vs_{shift_model}_map.html"
                    map_title = f"Scenario {scenario.title()}: {baseline_model} vs {shift_model}"
                    
                    success = create_comparison_map(
                        baseline_csv=baseline_csv,
                        shifted_csv=shift_csv,
                        out_html=str(map_file),
                        title=map_title
                    )
                    
                    if success:
                        print(f"      ✅ Generated: {map_file.name}")
        
        # Create scenario navigation index
        create_scenario_navigation_index(scn_dir, scenario, list(baseline_csvs.keys()), list(shift_csvs.keys()))
    
    # Create master navigation index
    create_master_navigation_index(scenario_viz_dir, list(results_data.keys()))
    
    return str(scenario_viz_dir)

# Model-specific enhanced visualizations function removed
# Only scenario-centric visualizations are now generated

def create_scenario_navigation_index(scn_dir: pathlib.Path, scenario: str, baseline_models: List[str], shift_models: List[str]):
    """
    Create an HTML navigation index for a specific scenario analysis.
    """
    index_file = scn_dir / f"scenario_{scenario}_index.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Scenario Analysis: {scenario.title()}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }}
            h1 {{ margin: 0; font-size: 2em; }}
            h2 {{ color: #333; border-bottom: 3px solid #667eea; padding-bottom: 10px; margin-top: 30px; }}
            .model-summary {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .model-card {{
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border-left: 5px solid #28a745;
            }}
            .model-card.shift {{ border-left-color: #ffc107; }}
            .viz-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
            .viz-card {{
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.2s;
            }}
            .viz-card:hover {{ transform: translateY(-5px); }}
            .viz-card h3 {{ margin-top: 0; color: #333; }}
            .viz-card a {{
                display: inline-block;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 25px;
                margin: 5px;
                font-weight: bold;
                transition: transform 0.2s;
            }}
            .viz-card a:hover {{ transform: scale(1.05); }}
            .description {{ color: #666; margin: 10px 0; line-height: 1.6; }}
            .icon {{ font-size: 1.2em; margin-right: 8px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛩️ Scenario Analysis Dashboard</h1>
            <h2 style="margin: 10px 0 0 0; border: none;">Scenario: {scenario.title()}</h2>
        </div>
        
        <div class="description">
            <p><strong>Scenario-centric analysis</strong> showing how different models perform on the <strong>{scenario}</strong> scenario.</p>
            <p>Compare baseline performance (models trained on this scenario) vs shift performance (models trained on other scenarios).</p>
        </div>
        
        <h2><span class="icon">🤖</span>Model Performance Overview</h2>
        <div class="model-summary">
            <div class="model-card">
                <h3><span class="icon">🏠</span>Baseline Models</h3>
                <p><strong>Trained on {scenario}:</strong></p>
                <p>{', '.join(baseline_models) if baseline_models else 'None'}</p>
                <p><em>Expected to perform optimally on this scenario</em></p>
            </div>
            <div class="model-card shift">
                <h3><span class="icon">🔄</span>Shift Models</h3>
                <p><strong>Trained on other scenarios:</strong></p>
                <p>{', '.join(shift_models) if shift_models else 'None'}</p>
                <p><em>Tests generalization capability</em></p>
            </div>
        </div>
        
        <h2><span class="icon">📊</span>Combined Analysis</h2>
        <div class="viz-grid">
            <div class="viz-card">
                <h3>🌟 All Models Comparison</h3>
                <div class="description">Interactive plot showing all baseline and shift performances on {scenario} scenario</div>
                <a href="./scenario_{scenario}_all_models_comparison.html" target="_blank">🔍 View Combined Analysis</a>
            </div>
        </div>
        
        <h2><span class="icon">📈</span>Individual Comparisons</h2>
        <div class="viz-grid">
    """
    
    # Add individual comparison cards
    for baseline_model in baseline_models:
        for shift_model in shift_models:
            html_content += f"""
                <div class="viz-card">
                    <h3>🆚 {baseline_model} vs {shift_model}</h3>
                    <div class="description">Baseline vs Shift comparison on {scenario} scenario</div>
                    <a href="./scenario_{scenario}_{baseline_model}_vs_{shift_model}.html" target="_blank">📊 View Plot</a>
                    <a href="./scenario_{scenario}_{baseline_model}_vs_{shift_model}_map.html" target="_blank">🗺️ View Map</a>
                </div>
            """
    
    html_content += f"""
        </div>
        
        <h2><span class="icon">🎯</span>Key Insights</h2>
        <div class="description" style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <ul style="list-style-type: none; padding: 0;">
                <li><span class="icon">🏆</span><strong>Baseline Performance:</strong> How models trained on {scenario} perform (expected to be optimal)</li>
                <li><span class="icon">🔄</span><strong>Shift Performance:</strong> How models trained on other scenarios perform (tests generalization)</li>
                <li><span class="icon">📉</span><strong>Domain Specificity:</strong> Performance drops in shifts indicate scenario-specific learning</li>
                <li><span class="icon">💪</span><strong>Robustness:</strong> Smaller performance drops indicate better generalization</li>
            </ul>
        </div>
        
        <div style="text-align: center; margin-top: 40px; padding: 20px; background: white; border-radius: 10px;">
            <p><a href="../master_scenario_analysis_index.html" style="background: #28a745; color: white; padding: 15px 30px; text-decoration: none; border-radius: 25px; font-weight: bold;">🏠 Back to Master Index</a></p>
        </div>
        
        <hr style="margin: 40px 0; border: none; height: 1px; background: #ddd;">
        <p style="text-align: center; color: #666; font-size: 0.9em;">
            Generated by scenario-centric baseline vs shift matrix analysis
        </p>
    </body>
    </html>
    """
    
    try:
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"      ✅ Generated scenario index: {index_file.name}")
    except Exception as e:
        print(f"      ❌ Failed to generate scenario index: {e}")

def create_master_navigation_index(viz_dir: pathlib.Path, scenarios: List[str]):
    """
    Create a master navigation index for all scenario analyses.
    """
    master_index = viz_dir / "master_scenario_analysis_index.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ATC Scenario Analysis - Master Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); min-height: 100vh; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 20px; text-align: center; margin-bottom: 30px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); }}
            h1 {{ margin: 0; font-size: 3em; font-weight: 300; }}
            .subtitle {{ font-size: 1.2em; margin-top: 10px; opacity: 0.9; }}
            .scenario-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px; margin: 30px 0; }}
            .scenario-card {{
                background: white;
                border-radius: 15px;
                padding: 30px;
                text-align: center;
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                transition: all 0.3s ease;
                border-top: 5px solid #667eea;
            }}
            .scenario-card:hover {{ transform: translateY(-10px); box-shadow: 0 15px 30px rgba(0,0,0,0.15); }}
            .scenario-card h3 {{ margin-top: 0; color: #333; font-size: 1.5em; }}
            .scenario-card a {{
                display: inline-block;
                background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                color: white;
                padding: 15px 30px;
                text-decoration: none;
                border-radius: 30px;
                margin: 15px;
                font-weight: bold;
                font-size: 1.1em;
                transition: all 0.3s ease;
                box-shadow: 0 4px 8px rgba(40, 167, 69, 0.3);
            }}
            .scenario-card a:hover {{ transform: scale(1.05); box-shadow: 0 6px 12px rgba(40, 167, 69, 0.4); }}
            .description {{ color: #666; margin: 20px 0; line-height: 1.8; font-size: 1.1em; }}
            .feature-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 40px 0; }}
            .feature-card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .icon {{ font-size: 2em; margin-bottom: 10px; }}
            .stats {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛩️ ATC Scenario Analysis</h1>
            <div class="subtitle">Master Dashboard - Scenario-Centric Baseline vs Shift Analysis</div>
            <div class="stats">
                <strong>📊 {len(scenarios)} Scenarios Analyzed</strong>
            </div>
        </div>
        
        <div class="description">
            <p><strong>Revolutionary scenario-centric approach:</strong> Instead of asking "How does model X perform on different scenarios?", 
            we now ask "How do all models perform on scenario X?"</p>
            <p>This provides deeper insights into scenario-specific learning patterns and cross-domain generalization capabilities.</p>
        </div>
        
        <div class="scenario-grid">
    """
    
    # Add scenario cards with enhanced styling
    scenario_icons = {
        'head_on': '✈️➡️⬅️✈️',
        'parallel': '✈️➡️➡️✈️',
        't_formation': '✈️⬆️⬇️✈️',
        'converging': '✈️↗️↙️✈️',
        'canonical_crossing': '✈️↗️↘️✈️'
    }
    
    for scenario in scenarios:
        icon = scenario_icons.get(scenario, '✈️🎯')
        html_content += f"""
            <div class="scenario-card">
                <div class="icon">{icon}</div>
                <h3>📊 {scenario.title().replace('_', ' ')} Scenario</h3>
                <div class="description">Comprehensive analysis of all model performances on the {scenario.replace('_', ' ')} scenario</div>
                <a href="./scenario_{scenario}_analysis/scenario_{scenario}_index.html" target="_blank">🔍 Explore Analysis</a>
            </div>
        """
    
    html_content += f"""
        </div>
        
        <div class="feature-grid">
            <div class="feature-card">
                <div class="icon">🔄</div>
                <h3>Before vs After</h3>
                <p><strong>Before:</strong> Model-centric - "How does model X perform on different scenarios?"</p>
                <p><strong>After:</strong> Scenario-centric - "How do all models perform on scenario X?"</p>
            </div>
            <div class="feature-card">
                <div class="icon">🎯</div>
                <h3>Key Benefits</h3>
                <p>• Direct baseline vs shift comparison per scenario</p>
                <p>• Understand domain-specific learning patterns</p>
                <p>• Identify robust vs specialized models</p>
            </div>
            <div class="feature-card">
                <div class="icon">📈</div>
                <h3>Analysis Features</h3>
                <p>• Interactive trajectory visualizations</p>
                <p>• Geographic map overlays</p>
                <p>• Performance metrics comparison</p>
            </div>
            <div class="feature-card">
                <div class="icon">🔍</div>
                <h3>Research Insights</h3>
                <p>• Scenario difficulty assessment</p>
                <p>• Model generalization capability</p>
                <p>• Safety risk identification</p>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 50px; padding: 30px; background: white; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h2 style="color: #333; margin-bottom: 20px;">🚀 Get Started</h2>
            <p style="font-size: 1.1em; color: #666; margin-bottom: 20px;">Select any scenario above to dive into the detailed analysis and discover how different models perform!</p>
        </div>
        
        <hr style="margin: 50px 0; border: none; height: 2px; background: linear-gradient(90deg, transparent, #667eea, transparent);">
        <p style="text-align: center; color: #666; font-size: 0.9em;">
            Generated by enhanced scenario-centric baseline vs shift matrix analysis | Improved visualization structure
        </p>
    </body>
    </html>
    """
    
    try:
        with open(master_index, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\n🎉 Generated master navigation index: {master_index.name}")
        print(f"📂 Open this file to navigate all scenario analyses: {master_index}")
    except Exception as e:
        print(f"❌ Failed to generate master index: {e}")

# Model-specific visualization index function removed
# Only scenario-centric navigation is now provided

def main():
    ap = argparse.ArgumentParser(description="Extensive baseline vs shift analysis with GPU acceleration")
    ap.add_argument("--models-index", type=str, default=None,
                    help="JSON: {'models': {'alias': 'path', ...}, 'baselines': {'alias_or_scenario': 'scenario_name'}}")
    ap.add_argument("--models-dir", type=str, default="models", help="Scan this folder for PPO_* checkpoints if no index.")
    ap.add_argument("--scenarios-dir", type=str, default="scenarios")
    ap.add_argument("--episodes", type=int, default=5, help="Number of episodes per scenario (default: 5 for more robust results)")
    ap.add_argument("--outdir", type=str, default="results_baseline_vs_shift")
    ap.add_argument("--use-gpu", action="store_true", help="Force GPU usage for testing acceleration")
    ap.add_argument("--extensive", action="store_true", help="Run extensive testing with more episodes and scenarios")
    args = ap.parse_args()
    
    # GPU Detection and Configuration
    import torch
    gpu_available = torch.cuda.is_available()
    use_gpu = args.use_gpu and gpu_available
    
    if args.use_gpu and not gpu_available:
        print("⚠️  GPU requested but not available. Falling back to CPU.")
    
    print(f"🔍 GPU Detection:")
    print(f"  CUDA available: {gpu_available}")
    print(f"  Using GPU: {use_gpu}")
    if gpu_available:
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Adjust episodes based on extensive flag
    if args.extensive:
        args.episodes = max(args.episodes, 10)
        print(f"🔬 EXTENSIVE MODE: Testing with {args.episodes} episodes per scenario")
    else:
        print(f"📊 STANDARD MODE: Testing with {args.episodes} episodes per scenario")

    outdir = pathlib.Path(args.outdir).resolve(); outdir.mkdir(parents=True, exist_ok=True)
    scen_map = discover_scenarios(args.scenarios_dir)
    if not scen_map: raise FileNotFoundError(f"No scenarios in {args.scenarios_dir}")

    # Load model list
    if args.models_index:
        with open(args.models_index,"r",encoding="utf-8") as f:
            mi = json.load(f)
        models = mi.get("models", {})
        explicit_baselines = mi.get("baselines", {})
    else:
        models = list_checkpoints_from_dir(args.models_dir)
        explicit_baselines = {}

    all_rows = []
    # Collect scenario-centric data for new visualization approach
    scenario_results = {}  # {scenario: {model: {'type': 'baseline'/'shift', 'csv_path': str}}}
    
    for alias, ckpt in models.items():
        # Figure baseline scenario name
        base_scn = explicit_baselines.get(alias) \
                    or explicit_baselines.get(parse_baseline_scenario_from_ckpt(ckpt) or "", None) \
                    or parse_baseline_scenario_from_ckpt(ckpt)

        if not base_scn or base_scn not in scen_map:
            # choose by substring fallback
            base_scn = next((s for s in scen_map if s in alias.lower()), None) or next(iter(scen_map.keys()))
        assert base_scn in scen_map, f"Could not resolve baseline scenario for {alias}"

        print(f"\n=== Model: {alias} ===")
        print(f"Checkpoint: {ckpt}")
        print(f"Baseline scenario: {base_scn}")

        algo, pid = load_algo(ckpt, use_gpu=use_gpu)  # uses registered env + path-healing; restores exactly as trained

        # --- Baseline run ---
        base_dir = outdir / f"{alias}__on__{base_scn}__baseline"
        base_df = run_model_on_scenario(algo, pid, scen_map[base_scn], str(base_dir), episodes=args.episodes)
        if base_df.empty:
            print(f"[WARN] No data for baseline {alias} on {base_scn}")
            continue
        base_df.to_csv(base_dir/"episode_metrics.csv", index=False)
        base_mean = {k: float(base_df[k].mean()) for k in ["f1_score","path_efficiency","min_separation_nm"] if k in base_df}

        # quick visuals (first episode)
        bc = base_df.iloc[0]["traj_csv"]
        plot_overlay(bc, str(base_dir/"overlay.png"), f"{alias} on {base_scn} – ep1")
        plot_minsep(bc,  str(base_dir/"minsep.png"),  f"{alias} on {base_scn} – min sep")
        
        # Collect for scenario-centric visualization
        if base_scn not in scenario_results:
            scenario_results[base_scn] = {}
        scenario_results[base_scn][alias] = {'type': 'baseline', 'csv_path': bc}

        # --- Shifted runs on all other scenarios ---
        shift_csvs = {}  # Collect shift trajectory CSVs for enhanced visualization
        for scen, scen_json in scen_map.items():
            if scen == base_scn: continue
            run_dir = outdir / f"{alias}__on__{scen}"
            df = run_model_on_scenario(algo, pid, scen_json, str(run_dir), episodes=args.episodes)
            if df.empty: 
                print(f"[WARN] No data for {alias} on {scen}")
                continue
            df["model_alias"] = alias
            df["scenario"] = scen
            df.to_csv(run_dir/"episode_metrics.csv", index=False)

            # visuals
            c = df.iloc[0]["traj_csv"]
            plot_overlay(c, str(run_dir/"overlay.png"), f"{alias} on {scen} – ep1")
            plot_minsep(c,  str(run_dir/"minsep.png"),  f"{alias} on {scen} – min sep")

            # Store CSV path for enhanced visualization
            shift_csvs[scen] = c
            
            # Collect for scenario-centric visualization
            if scen not in scenario_results:
                scenario_results[scen] = {}
            scenario_results[scen][alias] = {'type': 'shift', 'csv_path': c}

            # aggregate + deltas vs baseline
            row = {
                "model_alias": alias, "baseline_scenario": base_scn, "test_scenario": scen,
                # episode means
                "f1_score": float(df["f1_score"].mean()),
                "path_efficiency": float(df["path_efficiency"].mean()),
                "min_separation_nm": float(df["min_separation_nm"].mean()),
                # deltas (↑ better)
                "f1_vs_baseline_pct": pct_delta(float(df["f1_score"].mean()), base_mean.get("f1_score")),
                "path_eff_vs_baseline_pct": pct_delta(float(df["path_efficiency"].mean()), base_mean.get("path_efficiency")),
                "minsep_vs_baseline_pct": pct_delta(float(df["min_separation_nm"].mean()), base_mean.get("min_separation_nm")),
            }
            all_rows.append(row)

        # Model-specific visualizations removed - only scenario-centric visualizations are generated

        # stop algo between models to free resources
        try: algo.stop()
        except Exception: pass

    if not all_rows:
        print("No results.")
        return

    # Create comprehensive summary with detailed episode-level data (like targeted_shift_tester)
    detailed_rows = []
    
    # Process each model-scenario combination to extract detailed metrics
    for alias, ckpt in models.items():
        base_scn = explicit_baselines.get(alias) or parse_baseline_scenario_from_ckpt(ckpt)
        if not base_scn or base_scn not in scen_map:
            base_scn = next((s for s in scen_map if s in alias.lower()), None) or next(iter(scen_map.keys()))
        
        # Process baseline performance
        base_dir = outdir / f"{alias}__on__{base_scn}__baseline"
        if base_dir.exists():
            base_csv_path = base_dir / "episode_metrics.csv"
            if base_csv_path.exists():
                base_df = pd.read_csv(base_csv_path)
                for _, episode_row in base_df.iterrows():
                    if 'traj_csv' in episode_row and pd.notna(episode_row['traj_csv']):
                        # Extract detailed metrics from trajectory CSV
                        detailed_metrics = extract_comprehensive_metrics_from_csv(episode_row['traj_csv'])
                        
                        # Create summary row (similar to targeted_shift_tester structure)
                        summary_row = {
                            # Key identification columns
                            "model_alias": alias,
                            "baseline_scenario": base_scn,
                            "test_scenario": base_scn,  # Same as baseline for baseline tests
                            "model_type": "baseline",  # NEW: baseline vs shift indicator
                            "episode_id": int(episode_row.get('episode', 1)),
                            "seed": 42,  # Default seed
                            # Add all comprehensive metrics
                            **detailed_metrics
                        }
                        detailed_rows.append(summary_row)
        
        # Process shift performance
        for scen, scen_json in scen_map.items():
            if scen == base_scn:
                continue
                
            run_dir = outdir / f"{alias}__on__{scen}"
            if run_dir.exists():
                run_csv_path = run_dir / "episode_metrics.csv"
                if run_csv_path.exists():
                    run_df = pd.read_csv(run_csv_path)
                    for _, episode_row in run_df.iterrows():
                        if 'traj_csv' in episode_row and pd.notna(episode_row['traj_csv']):
                            # Extract detailed metrics from trajectory CSV
                            detailed_metrics = extract_comprehensive_metrics_from_csv(episode_row['traj_csv'])
                            
                            # Create summary row (similar to targeted_shift_tester structure)
                            summary_row = {
                                # Key identification columns
                                "model_alias": alias,
                                "baseline_scenario": base_scn,
                                "test_scenario": scen,
                                "model_type": "shift",  # NEW: baseline vs shift indicator
                                "episode_id": int(episode_row.get('episode', 1)),
                                "seed": 42,  # Default seed
                                # Add all comprehensive metrics
                                **detailed_metrics
                            }
                            detailed_rows.append(summary_row)
    
    # Save comprehensive detailed summary (like targeted_shift_tester)
    if detailed_rows:
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_summary_path = outdir / "baseline_vs_shift_detailed_summary.csv"
        detailed_df.to_csv(detailed_summary_path, index=False)
        print(f"\n📊 Generated detailed summary with {len(detailed_rows)} episode records")
        print(f"   Saved to: {detailed_summary_path}")
    else:
        print(f"\n⚠️  No detailed episode data found for comprehensive analysis")
    
    # Keep the original aggregated summary
    res = pd.DataFrame(all_rows)
    res.to_csv(outdir/"baseline_vs_shift_summary.csv", index=False)

    # simple grouped bars (one fig)
    for met in ["f1_score","path_efficiency","min_separation_nm"]:
        pv = res.pivot(index="test_scenario", columns="model_alias", values=met)
        plt.figure(figsize=(8, 4))
        pv.plot(kind="bar", ax=plt.gca(), alpha=0.85)
        plt.title(f"Avg {met} by model on shifted scenarios")
        plt.grid(True, axis="y", alpha=.3)
        plt.tight_layout()
        plt.savefig(outdir/f"summary_{met}.png", dpi=140)
        plt.close()

    # Performance summary
    total_episodes_tested = len(all_rows) * args.episodes
    unique_models = len(set(row['model_alias'] for row in all_rows))
    unique_scenarios = len(set(row['test_scenario'] for row in all_rows))
    
    # Generate scenario-centric visualizations
    if scenario_results:
        scenario_viz_path = generate_scenario_centric_visualizations(scenario_results, scen_map, outdir)
        print(f"\n🎨 Scenario-Centric Visualizations Generated!")
        print(f"📁 Navigate to: {scenario_viz_path}/master_scenario_analysis_index.html")
        print(f"🔄 New approach: See how all models perform on each scenario!")
    
    print(f"\n🎯 ANALYSIS COMPLETE! Results → {outdir}")
    print(f"📊 Performance Summary:")
    print(f"   • Models tested: {unique_models}")
    print(f"   • Scenarios tested: {unique_scenarios}")
    print(f"   • Total episodes: {total_episodes_tested}")
    print(f"   • GPU accelerated: {'✅ Yes' if use_gpu else '❌ No'}")
    if args.extensive:
        print(f"   • Extensive mode: ✅ Enabled")
    
    print(f"\n📁 Generated Files:")
    print("   • baseline_vs_shift_summary.csv (Aggregated statistical results)")
    print("   • baseline_vs_shift_detailed_summary.csv (Episode-level detailed metrics - same structure as targeted_shift_tester)")
    print("   • summary_*.png (Matplotlib visualizations)")
    print("   • � Scenario-centric visualizations in scenario_centric_visualizations/:")
    print("     ○ master_scenario_analysis_index.html (Main dashboard)")
    print("     ○ scenario_*_analysis/ (Per-scenario detailed analysis)")
    print("     ○ Interactive plots and maps for each scenario")
    
    print(f"\n🌐 Quick Access:")
    print(f"   🔥 Main Dashboard: scenario_centric_visualizations/master_scenario_analysis_index.html")
    
    if use_gpu:
        print(f"\n⚡ GPU acceleration was used for faster inference and testing.")
    else:
        print(f"\n💡 Tip: Use --use-gpu flag for faster testing with GPU acceleration.")
        
    if not args.extensive:
        print(f"💡 Tip: Use --extensive flag for more comprehensive testing (10+ episodes per scenario).")

if __name__ == "__main__":
    main()
