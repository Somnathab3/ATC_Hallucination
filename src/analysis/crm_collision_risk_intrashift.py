#!/usr/bin/env python3
"""
Module Name: crm_collision_risk_intrashift.py
Description: CRM analysis specifically for INTRASHIFT data structure.
Author: Som
Date: 2025-10-10

Computes lateral collision risk rate (N_ay) for intrashift perturbation testing.
Compares baseline (unperturbed) vs shift (perturbed) episodes within each trained model.

Data Structure Expected:
    intrashift_root/
        PPO_chase_2x2_TIMESTAMP/
            shifts/
                baseline/
                    summary_baseline_ep0.csv
                shift_type_1/
                    summary_shift_ep0.csv
                    summary_shift_ep1.csv
                ...

Usage:
    python crm_collision_risk_intrashift.py --data_dir "results/Intra_shift_results/Run" --output "crm_intrashift"
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math
import json

warnings.filterwarnings('ignore')

# Import constants and functions from main CRM module
sys.path.insert(0, str(Path(__file__).parent))
from crm_collision_risk import (
    S_Y_NM, S_X_NM, PZ_SAME_FL, LAMBDA_X, LAMBDA_Y, LAMBDA_Z,
    TLS_MIN, TLS_MAX, DEFAULT_SPEED_KT, SPEED_EPSILON,
    SCENARIO_GEOMETRY, haversine_nm, bearing_deg, angle_diff
)

# Set plotting style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})


class CRMIntrishiftAnalyzer:
    """CRM analyzer for intrashift perturbation data."""
    
    def __init__(self, data_dir: str, Sx_nm: float = S_X_NM, use_tier_b: bool = True):
        """
        Initialize intrashift CRM analyzer.
        
        Args:
            data_dir: Root directory containing model subdirectories
            Sx_nm: Longitudinal window for occupancy calculation
            use_tier_b: Whether to use trajectory-based P_y calculation (Tier B)
        """
        self.data_dir = Path(data_dir)
        self.Sx_nm = Sx_nm
        self.use_tier_b = use_tier_b
        self.episodes_data = []
        self.crm_results = []
        self.baseline_stats = {}
        
        print("🔄 Initializing CRM Intrashift Analyzer")
        print(f"   Data directory: {data_dir}")
        print(f"   Longitudinal window (S_x): {Sx_nm} NM")
        print(f"   Tier B (trajectory-based P_y): {use_tier_b}")
        
    def load_intrashift_data(self):
        """Load episode data from intrashift directory structure."""
        print("\n📂 Loading intrashift episode data...")
        
        if not self.data_dir.exists():
            raise ValueError(f"❌ Directory not found: {self.data_dir}")
        
        # Find all model directories (PPO_*)
        model_dirs = sorted([d for d in self.data_dir.iterdir() 
                            if d.is_dir() and d.name.startswith('PPO_')])
        
        if not model_dirs:
            raise ValueError(f"❌ No PPO model directories found in {self.data_dir}")
        
        print(f"   Found {len(model_dirs)} model directories")
        
        for model_dir in model_dirs:
            self._load_model_episodes(model_dir)
        
        if not self.episodes_data:
            raise ValueError("❌ No episode data loaded!")
        
        print(f"\n✅ Loaded {len(self.episodes_data)} episodes")
        print(f"   Models: {len(set(ep['model_name'] for ep in self.episodes_data))}")
        print(f"   Scenarios: {len(set(ep['scenario'] for ep in self.episodes_data))}")
        
    def _load_model_episodes(self, model_dir: Path):
        """Load all episodes for a single model."""
        # Extract model info from directory name
        # Format: PPO_scenario_TIMESTAMP
        parts = model_dir.name.split('_')
        if len(parts) >= 2:
            # Handle multi-part scenario names (e.g., chase_2x2)
            timestamp = parts[-1]
            scenario = '_'.join(parts[1:-1])
        else:
            scenario = "unknown"
            timestamp = "unknown"
        
        model_name = f"PPO_{scenario}_{timestamp}"
        
        shifts_dir = model_dir / "shifts"
        if not shifts_dir.exists():
            print(f"   ⚠️  No shifts directory in {model_dir.name}")
            return
        
        # Load baseline episode(s)
        baseline_dir = shifts_dir / "baseline"
        if baseline_dir.exists():
            baseline_episodes = self._load_shift_episodes(
                baseline_dir, model_name, scenario, "baseline", "baseline"
            )
            self.episodes_data.extend(baseline_episodes)
            print(f"   ✅ {model_name}: {len(baseline_episodes)} baseline episodes")
        else:
            print(f"   ⚠️  No baseline directory in {model_name}")
            return
        
        # Load all shift episodes
        shift_dirs = sorted([d for d in shifts_dir.iterdir() 
                            if d.is_dir() and d.name != "baseline"])
        
        total_shift_episodes = 0
        for shift_dir in shift_dirs:
            shift_name = shift_dir.name
            # Parse shift type from name (e.g., "hdg_micro_A1_+5deg" -> "heading")
            if shift_name.startswith('hdg_'):
                shift_type = 'heading'
            elif shift_name.startswith('pos_closer'):
                shift_type = 'position_closer'
            elif shift_name.startswith('pos_lateral'):
                shift_type = 'position_lateral'
            elif shift_name.startswith('speed_'):
                shift_type = 'speed'
            elif shift_name.startswith('wind_'):
                shift_type = 'wind'
            elif shift_name.startswith('waypoint_'):
                shift_type = 'waypoint'
            elif shift_name.startswith('aircraft_'):
                shift_type = 'aircraft_type'
            else:
                shift_type = 'unknown'
            
            shift_episodes = self._load_shift_episodes(
                shift_dir, model_name, scenario, shift_type, shift_name
            )
            self.episodes_data.extend(shift_episodes)
            total_shift_episodes += len(shift_episodes)
        
        if total_shift_episodes > 0:
            print(f"   ✅ {model_name}: {total_shift_episodes} shift episodes from {len(shift_dirs)} shifts")
    
    def _load_shift_episodes(self, shift_dir: Path, model_name: str, 
                            scenario: str, shift_type: str, shift_name: str) -> List[Dict]:
        """Load all episode CSVs from a shift directory."""
        episodes = []
        
        # Find all summary CSV files
        csv_files = list(shift_dir.glob("summary_*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Check required columns
                required_cols = ['min_separation_nm', 'flight_time_s', 'total_path_length_nm']
                if not all(col in df.columns for col in required_cols):
                    print(f"      ⚠️  Missing columns in {csv_file.name}")
                    continue
                
                # Extract episode data (assume one row per CSV)
                if len(df) == 0:
                    continue
                
                row = df.iloc[0]
                
                episode = {
                    'model_name': model_name,
                    'scenario': scenario,
                    'shift_type': shift_type,
                    'shift_name': shift_name,
                    'is_baseline': (shift_type == 'baseline'),
                    'episode_file': csv_file.name,
                    'shift_dir': shift_dir,  # Store directory for trajectory lookup
                    'min_separation_nm': float(row['min_separation_nm']),
                    'flight_time_s': float(row['flight_time_s']),
                    'total_path_length_nm': float(row['total_path_length_nm']),
                    'num_los_events': int(row.get('num_los_events', 0)),
                }
                
                episodes.append(episode)
                
            except Exception as e:
                print(f"      ⚠️  Error loading {csv_file.name}: {e}")
                continue
        
        return episodes
    
    def calculate_baseline_stats(self):
        """Calculate baseline CRM statistics for each model."""
        print("\n📊 Computing baseline CRM statistics...")
        
        baseline_episodes = [ep for ep in self.episodes_data if ep['is_baseline']]
        
        for model_name in set(ep['model_name'] for ep in baseline_episodes):
            model_baseline = [ep for ep in baseline_episodes if ep['model_name'] == model_name]
            
            if not model_baseline:
                continue
            
            scenario = model_baseline[0]['scenario']
            
            # Calculate P_y with Tier B if enabled
            p_y_method = "Tier A (episode min_sep)"
            p_y = self.calculate_p_y_tier_a(model_baseline)
            
            if self.use_tier_b:
                p_y_tier_b = self.calculate_p_y_tier_b(model_baseline)
                if p_y_tier_b is not None:
                    p_y = p_y_tier_b
                    p_y_method = "Tier B (trajectory pair-time)"
            
            # Estimate speed parameters
            speed_params = self._estimate_speed_parameters(model_baseline, scenario)
            
            # Calculate N_ay
            n_ay = self._calculate_n_ay(p_y, speed_params)
            
            self.baseline_stats[model_name] = {
                'scenario': scenario,
                'n_episodes': len(model_baseline),
                'p_y_5nm': p_y,
                'p_y_method': p_y_method,
                'n_ay_per_fh': n_ay,
                'mean_min_sep': np.mean([ep['min_separation_nm'] for ep in model_baseline]),
                'mean_flight_time': np.mean([ep['flight_time_s'] for ep in model_baseline]),
                'speed_params': speed_params
            }
            
            tls_flag = "PASS" if n_ay <= TLS_MAX else "FAIL"
            print(f"   {model_name}: N_ay={n_ay:.4e}, P_y={p_y:.4f} ({p_y_method}), TLS={tls_flag}")
    
    def calculate_p_y_tier_a(self, episodes: List[Dict]) -> float:
        """
        Tier A: Episode-level P_y using minimum separation.
        
        P_y(5) = fraction of episodes with min_separation < 5 NM
        
        Args:
            episodes: List of episode dictionaries
            
        Returns:
            P_y value
        """
        if not episodes:
            return 0.0
        
        los_count = sum(1 for ep in episodes if ep['min_separation_nm'] < S_Y_NM)
        return los_count / len(episodes)
    
    def calculate_p_y_tier_b(self, episodes: List[Dict]) -> Optional[float]:
        """
        Tier B: Trajectory-based P_y using pair-timestep lateral separation.
        
        P_y(5) = fraction of (pair, timestep) samples with lateral separation < 5 NM
        
        This is more accurate than Tier A because:
        - Uses all timesteps, not just minimum
        - Captures lateral separation specifically
        - Gives stable results even with few episodes
        
        Args:
            episodes: List of episode dictionaries with 'shift_dir' paths
            
        Returns:
            P_y value or None if trajectories unavailable
        """
        all_overlap_counts = []
        all_total_counts = []
        
        for episode in episodes:
            shift_dir = episode.get('shift_dir')
            if shift_dir is None:
                continue
            
            # Find trajectory CSV in shift directory
            traj_files = list(shift_dir.glob("traj_*.csv"))
            if not traj_files:
                continue
            
            traj_file = traj_files[0]
            
            try:
                traj_df = pd.read_csv(traj_file)
                
                # Check required columns
                if not all(col in traj_df.columns for col in ['timestep', 'agent', 'lat', 'lon']):
                    continue
                
                # Calculate pair-timestep lateral overlap
                overlap_count, total_count = self._calculate_lateral_overlap_from_trajectory(traj_df)
                
                if total_count > 0:
                    all_overlap_counts.append(overlap_count)
                    all_total_counts.append(total_count)
                    
            except Exception as e:
                print(f"      ⚠️  Error processing {traj_file.name}: {e}")
                continue
        
        if not all_total_counts:
            return None
        
        # Aggregate across all episodes
        total_overlap = sum(all_overlap_counts)
        total_pairs_timesteps = sum(all_total_counts)
        
        if total_pairs_timesteps == 0:
            return None
        
        p_y = total_overlap / total_pairs_timesteps
        return p_y
    
    def _calculate_lateral_overlap_from_trajectory(self, traj_df: pd.DataFrame) -> Tuple[int, int]:
        """
        Calculate lateral overlap counts from trajectory data.
        
        For each timestep t and pair (i,j):
        1. Compute horizontal separation distance
        2. If heading available, calculate lateral (cross-track) component
        3. Check if lateral separation < 5 NM
        
        Args:
            traj_df: Trajectory DataFrame with columns: timestep, agent, lat, lon, [heading, speed]
            
        Returns:
            Tuple of (overlap_count, total_pairs_timesteps)
        """
        timesteps = sorted(traj_df['timestep'].unique())
        
        overlap_count = 0
        total_pairs_timesteps = 0
        
        for t in timesteps:
            t_data = traj_df[traj_df['timestep'] == t]
            
            if len(t_data) < 2:
                continue
            
            agents = t_data['agent'].unique()
            
            # Check all pairs at this timestep
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    aid_i = agents[i]
                    aid_j = agents[j]
                    
                    row_i = t_data[t_data['agent'] == aid_i].iloc[0]
                    row_j = t_data[t_data['agent'] == aid_j].iloc[0]
                    
                    lat_i, lon_i = row_i['lat'], row_i['lon']
                    lat_j, lon_j = row_j['lat'], row_j['lon']
                    
                    # Calculate horizontal distance
                    hmd = haversine_nm(lat_i, lon_i, lat_j, lon_j)
                    
                    # Calculate lateral component if heading available
                    if 'heading' in row_i and pd.notna(row_i['heading']):
                        # Bearing from i to j
                        brg_ij = bearing_deg(lat_i, lon_i, lat_j, lon_j)
                        hdg_i = row_i['heading']
                        
                        # Angle between i's track and pair line
                        track_angle = angle_diff(brg_ij, hdg_i)
                        
                        # Lateral component (cross-track separation)
                        lateral_sep = hmd * abs(math.sin(math.radians(track_angle)))
                    else:
                        # No heading data, use total HMD as approximation
                        lateral_sep = hmd
                    
                    total_pairs_timesteps += 1
                    
                    if lateral_sep < S_Y_NM:
                        overlap_count += 1
        
        return overlap_count, total_pairs_timesteps
    
    def compute_crm_for_shifts(self):
        """Compute CRM for all shift episodes and compare to baseline."""
        print("\n🧮 Computing CRM for shift perturbations...")
        
        # Group shift episodes by model and shift_name
        shift_episodes = [ep for ep in self.episodes_data if not ep['is_baseline']]
        
        groups = {}
        for ep in shift_episodes:
            key = (ep['model_name'], ep['shift_name'])
            if key not in groups:
                groups[key] = []
            groups[key].append(ep)
        
        print(f"   Processing {len(groups)} shift configurations...")
        
        for (model_name, shift_name), episodes in groups.items():
            self._compute_crm_for_group(model_name, shift_name, episodes)
        
        print(f"\n✅ Computed CRM for {len(self.crm_results)} shift configurations")
    
    def _compute_crm_for_group(self, model_name: str, shift_name: str, episodes: List[Dict]):
        """Compute CRM metrics for a group of shift episodes."""
        if not episodes:
            return
        
        scenario = episodes[0]['scenario']
        shift_type = episodes[0]['shift_type']
        
        # Calculate P_y with Tier B if enabled
        p_y_method = "Tier A (episode min_sep)"
        p_y = self.calculate_p_y_tier_a(episodes)
        
        if self.use_tier_b:
            p_y_tier_b = self.calculate_p_y_tier_b(episodes)
            if p_y_tier_b is not None:
                p_y = p_y_tier_b
                p_y_method = "Tier B (trajectory pair-time)"
                print(f"      Using Tier B: P_y={p_y:.4f} from trajectory analysis")
        
        # Estimate speed parameters
        speed_params = self._estimate_speed_parameters(episodes, scenario)
        
        # Calculate N_ay
        n_ay = self._calculate_n_ay(p_y, speed_params)
        
        # Compare to baseline
        baseline_key = model_name
        n_ay_delta_pct = None
        tls_flag = "PASS" if n_ay <= TLS_MAX else "FAIL"
        
        if baseline_key in self.baseline_stats:
            baseline_n_ay = self.baseline_stats[baseline_key]['n_ay_per_fh']
            if baseline_n_ay > 0:
                n_ay_delta_pct = (n_ay - baseline_n_ay) / baseline_n_ay * 100
        
        result = {
            'model_name': model_name,
            'scenario': scenario,
            'shift_type': shift_type,
            'shift_name': shift_name,
            'n_episodes': len(episodes),
            'p_y_5nm': p_y,
            'p_y_method': p_y_method,
            'n_ay_per_fh': n_ay,
            'tls_flag': tls_flag,
            'n_ay_delta_vs_baseline_pct': n_ay_delta_pct,
            'mean_min_separation_nm': np.mean([ep['min_separation_nm'] for ep in episodes]),
            'mean_flight_time_s': np.mean([ep['flight_time_s'] for ep in episodes]),
            'v_avg_kt': speed_params['v_avg_kt'],
            'flow_same_per_h': speed_params['flow_same'],
            'flow_opp_per_h': speed_params['flow_opp'],
            'Sx_nm': speed_params['Sx_nm']
        }
        
        self.crm_results.append(result)
    
    def _estimate_speed_parameters(self, episodes: List[Dict], scenario_name: str) -> Dict[str, float]:
        """Estimate speed and flow parameters from episodes."""
        # Calculate average speed
        valid_episodes = [ep for ep in episodes 
                         if ep['flight_time_s'] > 0 and ep['total_path_length_nm'] > 0]
        
        if valid_episodes:
            speeds = [ep['total_path_length_nm'] / (ep['flight_time_s'] / 3600.0) 
                     for ep in valid_episodes]
            v_avg_kt = np.mean(speeds)
        else:
            v_avg_kt = DEFAULT_SPEED_KT
        
        # Get scenario geometry
        scenario_geom = SCENARIO_GEOMETRY.get(scenario_name, SCENARIO_GEOMETRY['generic'])
        
        delta_v_kt = max(v_avg_kt * scenario_geom['delta_v_factor'], SPEED_EPSILON)
        ydot_kt = v_avg_kt * scenario_geom['ydot_factor']
        zdot_kt = 0.0
        
        return {
            'flow_same': scenario_geom['flow_same'],
            'flow_opp': scenario_geom['flow_opp'],
            'delta_v_kt': delta_v_kt,
            'ydot_kt': ydot_kt,
            'zdot_kt': zdot_kt,
            'v_avg_kt': v_avg_kt,
            'Sx_nm': self.Sx_nm
        }
    
    def _calculate_n_ay(self, p_y: float, speed_params: Dict[str, float]) -> float:
        """Calculate lateral collision risk rate using ICAO/Eurocontrol CRM formula."""
        v_avg_kt = speed_params['v_avg_kt']
        Sx_nm = speed_params['Sx_nm']
        
        # Calculate occupancies
        E_same = speed_params['flow_same'] * (2.0 * Sx_nm) / max(SPEED_EPSILON, v_avg_kt)
        E_opp = speed_params['flow_opp'] * (2.0 * Sx_nm) / max(SPEED_EPSILON, v_avg_kt)
        
        delta_v_kt = speed_params['delta_v_kt']
        ydot_kt = speed_params['ydot_kt']
        zdot_kt = speed_params['zdot_kt']
        
        # CRM brackets
        bracket_same = (
            delta_v_kt / (2.0 * LAMBDA_X) +
            ydot_kt / (2.0 * LAMBDA_Y) +
            zdot_kt / (2.0 * LAMBDA_Z)
        )
        
        bracket_opp = (
            (2.0 * v_avg_kt) / (2.0 * LAMBDA_X) +
            ydot_kt / (2.0 * LAMBDA_Y) +
            zdot_kt / (2.0 * LAMBDA_Z)
        )
        
        # ICAO formula
        n_ay = p_y * PZ_SAME_FL * (LAMBDA_X / Sx_nm) * (
            E_same * bracket_same + E_opp * bracket_opp
        )
        
        return n_ay
    
    def export_results(self, output_dir: str):
        """Export CRM results to CSV and visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Exporting results to {output_dir}...")
        
        # Export shift results CSV
        df_shifts = pd.DataFrame(self.crm_results)
        df_shifts = df_shifts.sort_values(['model_name', 'shift_type', 'shift_name'])
        
        csv_path = output_path / "crm_intrashift_results.csv"
        df_shifts.to_csv(csv_path, index=False, float_format='%.10e')
        print(f"   ✅ Saved: {csv_path.name}")
        
        # Export baseline results CSV
        baseline_rows = []
        for model_name, stats in self.baseline_stats.items():
            baseline_rows.append({
                'model_name': model_name,
                'scenario': stats['scenario'],
                'n_episodes': stats['n_episodes'],
                'p_y_5nm': stats['p_y_5nm'],
                'p_y_method': stats.get('p_y_method', 'Tier A'),
                'n_ay_per_fh': stats['n_ay_per_fh'],
                'tls_flag': "PASS" if stats['n_ay_per_fh'] <= TLS_MAX else "FAIL",
                'mean_min_separation_nm': stats['mean_min_sep'],
                'mean_flight_time_s': stats['mean_flight_time']
            })
        
        df_baseline = pd.DataFrame(baseline_rows)
        baseline_csv_path = output_path / "crm_intrashift_baseline.csv"
        df_baseline.to_csv(baseline_csv_path, index=False, float_format='%.10e')
        print(f"   ✅ Saved: {baseline_csv_path.name}")
        
        # Generate visualizations
        self._generate_visualizations(df_shifts, df_baseline, output_path)
        
        # Generate summary report
        self._generate_summary_report(df_shifts, df_baseline, output_path)
        
        print("\n✅ CRM intrashift analysis complete!")
    
    def _generate_visualizations(self, df_shifts: pd.DataFrame, 
                                 df_baseline: pd.DataFrame, output_path: Path):
        """Generate visualization plots."""
        print("\n📊 Generating visualizations...")
        
        # 1. Baseline vs shift comparison by shift type
        self._plot_shift_type_comparison(df_shifts, df_baseline, output_path)
        
        # 2. TLS violations heatmap
        self._plot_tls_heatmap(df_shifts, output_path)
        
        # 3. Delta N_ay distribution
        self._plot_delta_distribution(df_shifts, output_path)
    
    def _plot_shift_type_comparison(self, df_shifts: pd.DataFrame, 
                                    df_baseline: pd.DataFrame, output_path: Path):
        """Plot N_ay comparison: baseline vs shift types."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Group by model and shift type
        models = sorted(df_shifts['model_name'].unique())
        shift_types = sorted(df_shifts['shift_type'].unique())
        
        x = np.arange(len(models))
        width = 0.8 / (len(shift_types) + 1)
        
        # Plot baseline
        baseline_vals = [df_baseline[df_baseline['model_name'] == m]['n_ay_per_fh'].values[0] 
                        if m in df_baseline['model_name'].values else 0 
                        for m in models]
        ax.bar(x - width * len(shift_types) / 2, baseline_vals, width, 
               label='Baseline', color='#2ecc71', alpha=0.9)
        
        # Plot each shift type
        colors = plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(shift_types)))
        for i, shift_type in enumerate(shift_types):
            shift_vals = []
            for model in models:
                vals = df_shifts[(df_shifts['model_name'] == model) & 
                                (df_shifts['shift_type'] == shift_type)]['n_ay_per_fh']
                shift_vals.append(vals.mean() if len(vals) > 0 else 0)
            
            ax.bar(x - width * len(shift_types) / 2 + width * (i + 1), shift_vals, width,
                  label=shift_type.replace('_', ' ').title(), color=colors[i], alpha=0.8)
        
        # TLS line
        ax.axhline(y=TLS_MAX, color='red', linestyle='--', linewidth=2, 
                  label=f'TLS Max ({TLS_MAX:.2e})')
        
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('N_ay (per flight hour)', fontweight='bold')
        ax.set_title('CRM Collision Risk: Baseline vs Shift Types', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('PPO_', '').replace('_2025', '') for m in models], 
                          rotation=45, ha='right')
        ax.legend(loc='upper left', ncol=2, fontsize=8)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        save_path = output_path / "crm_intrashift_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
    
    def _plot_tls_heatmap(self, df_shifts: pd.DataFrame, output_path: Path):
        """Plot TLS violations heatmap."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create pivot: rows=shift_names, cols=models, values=TLS flag
        pivot_data = []
        
        shift_names = sorted(df_shifts['shift_name'].unique())
        models = sorted(df_shifts['model_name'].unique())
        
        for shift_name in shift_names:
            row = {'shift': shift_name}
            for model in models:
                subset = df_shifts[(df_shifts['shift_name'] == shift_name) & 
                                  (df_shifts['model_name'] == model)]
                if len(subset) > 0:
                    n_ay = subset['n_ay_per_fh'].values[0]
                    row[model] = 1 if n_ay > TLS_MAX else 0  # 1 = FAIL, 0 = PASS
                else:
                    row[model] = np.nan
            pivot_data.append(row)
        
        pivot_df = pd.DataFrame(pivot_data).set_index('shift')
        
        sns.heatmap(pivot_df, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'TLS Violation (1=FAIL, 0=PASS)'}, ax=ax,
                   vmin=0, vmax=1, linewidths=0.5)
        
        ax.set_title('TLS Compliance Heatmap (Intrashift)', fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Shift Configuration', fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
        
        plt.tight_layout()
        save_path = output_path / "crm_intrashift_tls_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
    
    def _plot_delta_distribution(self, df_shifts: pd.DataFrame, output_path: Path):
        """Plot distribution of N_ay delta vs baseline."""
        valid_deltas = df_shifts[df_shifts['n_ay_delta_vs_baseline_pct'].notna()]
        
        if len(valid_deltas) == 0:
            print("   ⚠️  No delta values available for distribution plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        deltas = valid_deltas['n_ay_delta_vs_baseline_pct']
        ax1.hist(deltas, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Baseline')
        ax1.axvline(x=deltas.median(), color='orange', linestyle=':', linewidth=2, 
                   label=f'Median ({deltas.median():.1f}%)')
        ax1.set_xlabel('Δ N_ay vs Baseline (%)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Distribution of N_ay Change Under Perturbations', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot by shift type
        shift_types = sorted(valid_deltas['shift_type'].unique())
        data_by_type = [valid_deltas[valid_deltas['shift_type'] == st]['n_ay_delta_vs_baseline_pct'] 
                       for st in shift_types]
        
        bp = ax2.boxplot(data_by_type, labels=[st.replace('_', '\n') for st in shift_types],
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Baseline')
        ax2.set_xlabel('Shift Type', fontweight='bold')
        ax2.set_ylabel('Δ N_ay vs Baseline (%)', fontweight='bold')
        ax2.set_title('N_ay Change by Perturbation Type', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = output_path / "crm_intrashift_delta_distribution.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
    
    def _generate_summary_report(self, df_shifts: pd.DataFrame, 
                                 df_baseline: pd.DataFrame, output_path: Path):
        """Generate markdown summary report."""
        md_path = output_path / "crm_intrashift_summary.md"
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# CRM Intrashift Analysis Summary\n\n")
            f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Baseline Performance\n\n")
            f.write("| Model | Scenario | N_ay (per FH) | TLS | P_y(5 NM) | Method | Min Sep (NM) |\n")
            f.write("|-------|----------|---------------|-----|-----------|--------|-------------|\n")
            for _, row in df_baseline.iterrows():
                f.write(f"| {row['model_name'].replace('PPO_', '')} | {row['scenario']} | "
                       f"{row['n_ay_per_fh']:.4e} | **{row['tls_flag']}** | "
                       f"{row['p_y_5nm']:.4f} | {row.get('p_y_method', 'Tier A')} | {row['mean_min_separation_nm']:.2f} |\n")
            f.write("\n")
            
            f.write("## Shift Perturbation Impact\n\n")
            
            # Group by shift type
            for shift_type in sorted(df_shifts['shift_type'].unique()):
                type_data = df_shifts[df_shifts['shift_type'] == shift_type]
                
                f.write(f"### {shift_type.replace('_', ' ').title()}\n\n")
                f.write(f"- **Configurations tested:** {len(type_data)}\n")
                
                valid_deltas = type_data[type_data['n_ay_delta_vs_baseline_pct'].notna()]
                if len(valid_deltas) > 0:
                    f.write(f"- **Mean Δ N_ay:** {valid_deltas['n_ay_delta_vs_baseline_pct'].mean():.2f}%\n")
                    f.write(f"- **Max increase:** {valid_deltas['n_ay_delta_vs_baseline_pct'].max():.2f}%\n")
                    f.write(f"- **Max decrease:** {valid_deltas['n_ay_delta_vs_baseline_pct'].min():.2f}%\n")
                
                tls_violations = (type_data['tls_flag'] == 'FAIL').sum()
                f.write(f"- **TLS violations:** {tls_violations} / {len(type_data)} "
                       f"({tls_violations/len(type_data)*100:.1f}%)\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Worst performers
            worst = df_shifts.nlargest(10, 'n_ay_per_fh')
            f.write("**Highest Risk Configurations:**\n\n")
            for i, (_, row) in enumerate(worst.iterrows(), 1):
                f.write(f"{i}. {row['shift_name']} ({row['model_name'].replace('PPO_', '')}): "
                       f"{row['n_ay_per_fh']:.4e} - **{row['tls_flag']}**")
                if pd.notna(row['n_ay_delta_vs_baseline_pct']):
                    f.write(f" (Δ{row['n_ay_delta_vs_baseline_pct']:+.1f}%)")
                f.write("\n")
            f.write("\n")
            
            f.write(f"## TLS Compliance: {TLS_MIN:.2e} to {TLS_MAX:.2e} per FH\n\n")
            total_configs = len(df_shifts)
            passed = (df_shifts['tls_flag'] == 'PASS').sum()
            f.write(f"- **Overall pass rate:** {passed}/{total_configs} ({passed/total_configs*100:.1f}%)\n")
            
        print(f"   ✅ Saved: {md_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='CRM analysis for intrashift perturbation data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python crm_collision_risk_intrashift.py --data_dir "results/Intra_shift_results/Run" --output "crm_intrashift"
        """
    )
    
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        required=True,
        help='Root directory containing PPO model subdirectories with shifts/'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='crm_intrashift_analysis',
        help='Output directory for results. Default: crm_intrashift_analysis'
    )
    
    parser.add_argument(
        '--Sx-nm',
        type=float,
        default=S_X_NM,
        help=f'Longitudinal window for occupancy (NM). Default: {S_X_NM}'
    )
    
    parser.add_argument(
        '--no-tier-b',
        action='store_true',
        help='Disable Tier B trajectory-based P_y calculation (faster but less accurate)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = CRMIntrishiftAnalyzer(
            data_dir=args.data_dir,
            Sx_nm=args.Sx_nm,
            use_tier_b=not args.no_tier_b
        )
        
        # Load data
        analyzer.load_intrashift_data()
        
        # Compute baseline CRM
        analyzer.calculate_baseline_stats()
        
        # Compute CRM for shifts
        analyzer.compute_crm_for_shifts()
        
        # Export results
        analyzer.export_results(args.output)
        
        print("\n" + "="*70)
        print("✅ CRM Intrashift Analysis Complete!")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during CRM analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
