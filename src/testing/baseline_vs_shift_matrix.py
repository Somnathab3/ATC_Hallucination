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
    
    # Configure GPU usage for loaded algorithm if available
    if use_gpu:
        try:
            # Update algorithm config for GPU inference
            if hasattr(algo, 'config') and algo.config is not None:
                algo.config.num_gpus = 1
                print(f"✅ Algorithm configured for GPU inference")
            else:
                print(f"⚠️  Algorithm config not accessible for GPU configuration")
        except Exception as e:
            print(f"⚠️  Could not configure GPU for algorithm: {e}")
    
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
                a = algo.compute_single_action(obs[aid], explore=False, policy_id=policy_id)  # :contentReference[oaicite:6]{index=6}
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
            m = metrics_from_csv(csv_path)
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

def generate_enhanced_visualizations(alias: str, base_scn: str, base_csv: str, shift_csvs: Dict[str, str], 
                                   scen_map: Dict[str, str], viz_dir: pathlib.Path):
    """
    Generate interactive trajectory comparison plots and maps for baseline vs shifted scenarios.
    
    Args:
        alias: Model alias name
        base_scn: Baseline scenario name
        base_csv: Path to baseline trajectory CSV
        shift_csvs: Dict mapping scenario names to their trajectory CSV paths
        scen_map: Dict mapping scenario names to scenario JSON paths
        viz_dir: Output directory for visualizations
    """
    try:
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate interactive trajectory comparison plots using Plotly
        if create_trajectory_comparison_plot is not None:
            print(f"    Generating interactive trajectory plots...")
            
            # Create individual comparison plot for each shifted scenario
            for shift_scn, shift_csv in shift_csvs.items():
                plot_file = viz_dir / f"trajectory_comparison_{alias}_{base_scn}_vs_{shift_scn}.html"
                title = f"Trajectory Analysis: {alias} - {base_scn.title()} vs {shift_scn.title()}"
                
                success = create_trajectory_comparison_plot(
                    baseline_csv=base_csv,
                    shift_csvs={shift_scn: shift_csv},
                    out_html=str(plot_file),
                    title=title,
                    scenario_path=scen_map.get(shift_scn)
                )
                
                if success:
                    print(f"      ✅ Generated: trajectory_comparison_{alias}_{base_scn}_vs_{shift_scn}.html")
                else:
                    print(f"      ❌ Failed to generate trajectory plot for {shift_scn}")
            
            # Create combined plot showing all shifts for this model
            combined_plot_file = viz_dir / f"trajectory_comparison_{alias}_{base_scn}_vs_all.html"
            combined_title = f"Trajectory Analysis: {alias} - {base_scn.title()} vs All Scenarios"
            
            success = create_trajectory_comparison_plot(
                baseline_csv=base_csv,
                shift_csvs=shift_csvs,
                out_html=str(combined_plot_file),
                title=combined_title,
                scenario_path=scen_map.get(base_scn)
            )
            
            if success:
                print(f"      ✅ Generated combined trajectory plot: trajectory_comparison_{alias}_{base_scn}_vs_all.html")
        
        # Generate interactive maps using Folium
        if create_comparison_map is not None:
            print(f"    Generating interactive trajectory maps...")
            
            for shift_scn, shift_csv in shift_csvs.items():
                map_file = viz_dir / f"trajectory_map_{alias}_{base_scn}_vs_{shift_scn}.html"
                map_title = f"Trajectory Map: {alias} - {base_scn.title()} vs {shift_scn.title()}"
                
                success = create_comparison_map(
                    baseline_csv=base_csv,
                    shifted_csv=shift_csv,
                    out_html=str(map_file),
                    title=map_title
                )
                
                if success:
                    print(f"      ✅ Generated: trajectory_map_{alias}_{base_scn}_vs_{shift_scn}.html")
                else:
                    print(f"      ❌ Failed to generate trajectory map for {shift_scn}")
        
        # Create navigation index for all visualizations
        create_visualization_index(viz_dir, alias, base_scn, list(shift_csvs.keys()))
        
    except Exception as e:
        print(f"    ❌ Error generating enhanced visualizations: {e}")

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

def create_visualization_index(viz_dir: pathlib.Path, alias: str, base_scn: str, shift_scenarios: List[str]):
    """
    Create an HTML index file for easy navigation of all visualizations for this model.
    """
    index_file = viz_dir / f"visualization_index_{alias}_{base_scn}.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Visualization Index: {alias} - {base_scn.title()}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; text-align: center; }}
            h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
            .viz-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .viz-card {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background-color: #f9f9f9;
                text-align: center;
            }}
            .viz-card h3 {{ margin-top: 0; color: #333; }}
            .viz-card a {{
                display: inline-block;
                background-color: #007bff;
                color: white;
                padding: 8px 16px;
                text-decoration: none;
                border-radius: 4px;
                margin: 5px;
                font-size: 0.9em;
            }}
            .viz-card a:hover {{ background-color: #0056b3; }}
            .description {{ color: #666; margin: 10px 0; line-height: 1.4; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1>🛩️ Visualization Dashboard</h1>
        <h2>Model: {alias} | Baseline: {base_scn.title()}</h2>
        
        <div class="description">
            <p><strong>Interactive visualizations</strong> comparing baseline performance against shifted scenarios.</p>
            <p>Each visualization shows trajectory deviations, safety events, and performance metrics.</p>
        </div>
        
        <h2>📊 Interactive Trajectory Plots (Plotly)</h2>
        <div class="viz-grid">
    """
    
    # Add trajectory plot cards
    for shift_scn in shift_scenarios:
        html_content += f"""
            <div class="viz-card">
                <h3>{shift_scn.title()}</h3>
                <div class="description">Normalized trajectory comparison with statistical analysis</div>
                <a href="./trajectory_comparison_{alias}_{base_scn}_vs_{shift_scn}.html" target="_blank">View Plot</a>
            </div>
        """
    
    # Add combined plot card
    html_content += f"""
            <div class="viz-card">
                <h3>All Scenarios Combined</h3>
                <div class="description">Comprehensive view of all scenario shifts in one plot</div>
                <a href="./trajectory_comparison_{alias}_{base_scn}_vs_all.html" target="_blank">View Combined Plot</a>
            </div>
        </div>
        
        <h2>🗺️ Interactive Trajectory Maps (Folium)</h2>
        <div class="viz-grid">
    """
    
    # Add trajectory map cards
    for shift_scn in shift_scenarios:
        html_content += f"""
            <div class="viz-card">
                <h3>{shift_scn.title()}</h3>
                <div class="description">Geographic trajectory overlay with conflict detection markers</div>
                <a href="./trajectory_map_{alias}_{base_scn}_vs_{shift_scn}.html" target="_blank">View Map</a>
            </div>
        """
    
    html_content += f"""
        </div>
        
        <h2>📈 Key Features</h2>
        <div class="description">
            <ul>
                <li><strong>Trajectory Plots:</strong> Normalized position deviations with interactive hover details</li>
                <li><strong>Trajectory Maps:</strong> Geographic overlays with conflict markers and safety events</li>
                <li><strong>Performance Metrics:</strong> F1 score, path efficiency, and separation analysis</li>
                <li><strong>Safety Analysis:</strong> LoS events, collision risks, and hallucination detection results</li>
            </ul>
        </div>
        
        <hr>
        <p style="text-align: center; color: #666; font-size: 0.9em;">
            Generated by baseline_vs_shift_matrix.py with enhanced visualization support
        </p>
    </body>
    </html>
    """
    
    try:
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"      ✅ Generated visualization index: visualization_index_{alias}_{base_scn}.html")
    except Exception as e:
        print(f"      ❌ Failed to generate visualization index: {e}")

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

        # --- Generate Enhanced Visualizations ---
        if shift_csvs:  # Only if we have shift data
            print(f"  Generating enhanced visualizations for {alias}...")
            viz_dir = outdir / f"{alias}__visualizations"
            generate_enhanced_visualizations(alias, base_scn, bc, shift_csvs, scen_map, viz_dir)

        # stop algo between models to free resources
        try: algo.stop()
        except Exception: pass

    if not all_rows:
        print("No results.")
        return

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
    print("   • baseline_vs_shift_summary.csv (Statistical results)")
    print("   • summary_*.png (Matplotlib visualizations)")
    print("   • Model-centric visualizations in <model>__visualizations/ directories:")
    print("     ○ trajectory_comparison_*.html (Plotly interactive plots)")
    print("     ○ trajectory_map_*.html (Folium interactive maps)")
    print("     ○ visualization_index_*.html (Navigation dashboards)")
    print("   • 🆕 Scenario-centric visualizations in scenario_centric_visualizations/:")
    print("     ○ master_scenario_analysis_index.html (Main dashboard)")
    print("     ○ scenario_*_analysis/ (Per-scenario detailed analysis)")
    print("     ○ Interactive plots and maps for each scenario")
    
    print(f"\n🌐 Quick Access:")
    print(f"   🔥 NEW: scenario_centric_visualizations/master_scenario_analysis_index.html")
    print(f"   📊 Traditional: visualization_index_*.html files for model-centric views")
    
    if use_gpu:
        print(f"\n⚡ GPU acceleration was used for faster inference and testing.")
    else:
        print(f"\n💡 Tip: Use --use-gpu flag for faster testing with GPU acceleration.")
        
    if not args.extensive:
        print(f"💡 Tip: Use --extensive flag for more comprehensive testing (10+ episodes per scenario).")

if __name__ == "__main__":
    main()
