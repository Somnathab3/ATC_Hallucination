#!/usr/bin/env python3
"""
Module Name: crm_collision_risk.py
Description: Lateral Collision Risk Model (CRM) analysis for MARL-based ATC systems.
Author: Som
Date: 2025-10-08

Computes lateral collision risk rate (N_ay) for 5 NM separation using the standard
CRM formula. Compares baseline vs shift vs generic model performance against TLS
(Target Level of Safety) thresholds.

CRM Formula (lateral):
    N_ay = P_y(S_y) * P_z(0) * λ_x * [
        E_y(same) * (|ΔV|/(2λ_x) + |ẏ|/(2λ_y) + |ż|/(2λ_z)) +
        E_y(opp)  * (2|ΔV|/(2λ_x) + |ẏ|/(2λ_y) + |ż|/(2λ_z))
    ]

Where:
    - P_y(S_y): Probability of lateral overlap within S_y NM
    - P_z(0): Probability of same flight level (1.0 for 2D scenarios)
    - λ_x, λ_y, λ_z: Aircraft geometric dimensions (length, wingspan, height)
    - E_y(same/opp): Expected occupancy for same/opposite direction
    - |ΔV|: Relative along-track speed
    - |ẏ|: Lateral rate of closure
    - |ż|: Vertical rate (0 for same FL)

Two-tier P_y estimation:
    - Tier A (fallback): Episode-level min_separation_nm < 5 NM fraction
    - Tier B (preferred): Exact lateral overlap from trajectory pair-time analysis

Usage:
    python crm_collision_risk.py --data_dir "out2" --output "crm_analysis"
    python crm_collision_risk.py --intershift "out2" --intrashift "results" --output "crm_full"
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
from collections import defaultdict
from scipy import stats
import math

warnings.filterwarnings('ignore')

# Set publication-ready style
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

# ============================================================================
# CRM CONSTANTS (A320-like geometry)
# ============================================================================

# Separation standard
S_Y_NM = 5.0  # Lateral separation in nautical miles

# Longitudinal window for occupancy calculation (ICAO standard)
# Typical value: 60-120 NM depending on sector density
S_X_NM = 100.0  # Longitudinal window in nautical miles

# Probability of same flight level (1.0 when conditioning on 2D scenarios)
PZ_SAME_FL = 1.0

# Aircraft geometric dimensions in nautical miles (1 NM = 1852 m)
# A320 reference: length=37.6m, wingspan=35.8m, height=11.8m
LAMBDA_X = 37.6 / 1852.0  # length ≈ 0.0203 NM
LAMBDA_Y = 35.8 / 1852.0  # wingspan ≈ 0.0193 NM
LAMBDA_Z = 11.8 / 1852.0  # height ≈ 0.0064 NM

# Target Level of Safety (TLS) band
TLS_MIN = 2.0e-9  # Fatal accidents per flight hour
TLS_MAX = 5.0e-9

# Default speed for episodes with zero path (kt = NM/h)
DEFAULT_SPEED_KT = 250.0

# Default traffic flow rates (aircraft per hour on adjacent route)
# Conservative estimates for en-route sectors
DEFAULT_FLOW_SAME = 10.0  # Same direction traffic (ac/h)
DEFAULT_FLOW_OPP = 5.0    # Opposite direction traffic (ac/h)

# Epsilon for near-zero relative speeds
SPEED_EPSILON = 1e-3


# ============================================================================
# SCENARIO GEOMETRY DEFAULTS (Flow-based)
# ============================================================================

SCENARIO_GEOMETRY = {
    "head_on": {
        "flow_same": 0.0,      # No same-direction traffic (ac/h)
        "flow_opp": 10.0,      # Opposite direction traffic (ac/h)
        "delta_v_factor": 0.0, # |Δv| for same-dir pairs (not applicable)
        "ydot_factor": 0.0,    # Lateral closure rate
        "angle_deg": 180.0,
        "description": "Head-on encounter"
    },
    "parallel": {
        "flow_same": 10.0,     # Same-direction traffic (ac/h)
        "flow_opp": 0.0,       # No opposite traffic
        "delta_v_factor": 0.1, # Small speed difference
        "ydot_factor": 0.0,
        "angle_deg": 0.0,
        "description": "Parallel tracks"
    },
    "canonical_crossing": {
        "flow_same": 5.0,      # Mixed traffic
        "flow_opp": 5.0,
        "delta_v_factor": 1.0,
        "ydot_factor": 1.0,    # θ ≈ 90°, |ẏ| ≈ V_avg
        "angle_deg": 90.0,
        "description": "Perpendicular crossing"
    },
    "converging": {
        "flow_same": 5.0,
        "flow_opp": 5.0,
        "delta_v_factor": 1.0,
        "ydot_factor": 0.866,  # θ ≈ 60°, |ẏ| ≈ V_avg * sin(60°)
        "angle_deg": 60.0,
        "description": "Converging encounter"
    },
    "t_formation": {
        "flow_same": 10.0,
        "flow_opp": 0.0,
        "delta_v_factor": 0.5,
        "ydot_factor": 0.5,
        "angle_deg": 45.0,
        "description": "T-formation encounter"
    }
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance in nautical miles."""
    R_nm = 3440.065
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return R_nm * c


def heading_to_unit(heading_deg: float) -> Tuple[float, float]:
    """Convert heading in degrees to unit vector (east, north)."""
    rad = math.radians(90.0 - heading_deg)
    return (math.cos(rad), math.sin(rad))


def kt_to_nmh(kt: float) -> float:
    """Keep speed in knots (nautical miles per hour) - ICAO standard."""
    return kt  # No conversion needed - kt is already NM/h


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate initial bearing from point 1 to point 2 in degrees."""
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = (math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) -
         math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.cos(math.radians(lon2 - lon1)))
    brg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return brg


def angle_diff(a: float, b: float) -> float:
    """Calculate smallest angular difference between angles a and b."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


# ============================================================================
# CRM ANALYZER CLASS
# ============================================================================

class CRMCollisionRiskAnalyzer:
    """
    Collision Risk Model analyzer for MARL ATC systems.
    
    Computes lateral collision risk rate (N_ay) using episode metrics and
    optional raw trajectory data. Compares against TLS thresholds and baseline.
    """

    def __init__(self, data_dirs: Dict[str, str], use_tier_b: bool = True,
                 Sx_nm: float = S_X_NM, flow_same_override: Optional[float] = None,
                 flow_opp_override: Optional[float] = None):
        """
        Initialize CRM analyzer.
        
        Args:
            data_dirs: Dictionary with keys 'intershift', 'intrashift' pointing to data directories
            use_tier_b: Whether to use exact trajectory-based P_y calculation (Tier B)
            Sx_nm: Longitudinal window for occupancy calculation
            flow_same_override: Override same-direction flow rate (ac/h), None uses scenario defaults
            flow_opp_override: Override opposite-direction flow rate (ac/h), None uses scenario defaults
        """
        self.data_dirs = data_dirs
        self.use_tier_b = use_tier_b
        self.Sx_nm = Sx_nm
        self.flow_same_override = flow_same_override
        self.flow_opp_override = flow_opp_override
        self.df_detailed = None
        self.df_summary = None
        self.crm_results = []
        self.baseline_stats = {}
        
        print("🔄 Initializing CRM Collision Risk Analyzer")
        print(f"   Data directories: {list(data_dirs.keys())}")
        print(f"   Tier B (trajectory-based P_y): {use_tier_b}")
        print(f"   Longitudinal window (S_x): {Sx_nm} NM")
        if flow_same_override is not None or flow_opp_override is not None:
            print(f"   Flow rate overrides: same={flow_same_override}, opp={flow_opp_override}")
        
    def load_data(self):
        """Load baseline vs shift summary data from all sources."""
        print("\n📂 Loading summary data...")
        
        dfs_detailed = []
        dfs_summary = []
        
        for source_name, data_dir in self.data_dirs.items():
            data_path = Path(data_dir)
            
            if not data_path.exists():
                print(f"   ⚠️  Directory not found: {data_dir}")
                continue
            
            # Load detailed summary
            detailed_csv = data_path / "baseline_vs_shift_detailed_summary.csv"
            if detailed_csv.exists():
                df = pd.read_csv(detailed_csv)
                df['source'] = source_name
                dfs_detailed.append(df)
                print(f"   ✅ Loaded {len(df)} episodes from {source_name} (detailed)")
            
            # Load summary
            summary_csv = data_path / "baseline_vs_shift_summary.csv"
            if summary_csv.exists():
                df = pd.read_csv(summary_csv)
                df['source'] = source_name
                dfs_summary.append(df)
                print(f"   ✅ Loaded {len(df)} entries from {source_name} (summary)")
        
        if not dfs_detailed:
            raise ValueError("❌ No detailed summary data found!")
        
        self.df_detailed = pd.concat(dfs_detailed, ignore_index=True)
        if dfs_summary:
            self.df_summary = pd.concat(dfs_summary, ignore_index=True)
        
        # Calculate baseline statistics
        self._calculate_baseline_stats()
        
        print(f"\n✅ Total loaded: {len(self.df_detailed)} episodes")
        print(f"   Models: {self.df_detailed['model_alias'].nunique()}")
        print(f"   Scenarios: {self.df_detailed['test_scenario'].nunique()}")
        
    def _calculate_baseline_stats(self):
        """Calculate baseline statistics for comparison."""
        baseline_df = self.df_detailed[
            (self.df_detailed['test_scenario'] == self.df_detailed['baseline_scenario']) |
            (self.df_detailed['model_type'] == 'baseline')
        ]
        
        for (model_alias, scenario), group in baseline_df.groupby(['model_alias', 'baseline_scenario']):
            key = (model_alias, scenario)
            self.baseline_stats[key] = {
                'mean_min_sep': group['min_separation_nm'].mean(),
                'p_y_5nm': (group['min_separation_nm'] < S_Y_NM).mean(),
                'mean_flight_time': group['flight_time_s'].mean(),
                'mean_speed': (group['total_path_length_nm'] / (group['flight_time_s'] / 3600.0 + 1e-6)).mean(),
                'n_episodes': len(group)
            }
    
    def calculate_p_y_tier_a(self, episodes_df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Tier A: Conservative P_y(5) using episode-level minimum separation.
        
        P_y(5) ≈ Pr(X < 5) where X = min_t min_pairs HMD(t) per episode
        
        Args:
            episodes_df: DataFrame with min_separation_nm column
            
        Returns:
            Tuple of (P_y_mean, P_y_ci_lower, P_y_ci_upper)
        """
        if len(episodes_df) == 0:
            return 0.0, 0.0, 0.0
        
        # Episode-level binary indicator: was min_separation < 5 NM?
        los_indicators = (episodes_df['min_separation_nm'] < S_Y_NM).astype(float)
        
        # Mean and bootstrap CI
        p_y_mean = los_indicators.mean()
        
        # Bootstrap 90% CI
        if len(los_indicators) >= 10:
            bootstrap_samples = []
            n_bootstrap = 1000
            for _ in range(n_bootstrap):
                sample = np.random.choice(los_indicators, size=len(los_indicators), replace=True)
                bootstrap_samples.append(sample.mean())
            
            ci_lower = np.percentile(bootstrap_samples, 5)
            ci_upper = np.percentile(bootstrap_samples, 95)
        else:
            ci_lower, ci_upper = p_y_mean, p_y_mean
        
        return p_y_mean, ci_lower, ci_upper
    
    def calculate_p_y_tier_b(self, episodes_df: pd.DataFrame, 
                             data_dir: Path) -> Optional[Tuple[float, float, float]]:
        """
        Tier B: Exact lateral overlap from trajectory pair-time analysis.
        
        For each timestep t and pair (i,j):
        1. Compute relative bearing between aircraft
        2. Transform positions to track-aligned frame (along-track, cross-track)
        3. Check if |y_lateral| < 5 NM
        4. P_y(5) = fraction of (pair, timestep) samples with lateral overlap
        
        Args:
            episodes_df: DataFrame with episode metadata
            data_dir: Path to directory containing episode subdirectories with traj_ep_*.csv
            
        Returns:
            Tuple of (P_y_mean, P_y_ci_lower, P_y_ci_upper) or None if trajectories unavailable
        """
        episode_p_y_values = []
        
        for _, row in episodes_df.iterrows():
            # Find episode directory
            model_dir_name = f"{row['model_alias']}__on__{row['test_scenario']}"
            if row.get('model_type') == 'baseline':
                model_dir_name += "__baseline"
            elif row.get('model_type') == 'generic':
                model_dir_name += "__generic_shift"
            
            episode_dir = data_dir / model_dir_name / f"ep_{int(row['episode_id']):03d}"
            
            if not episode_dir.exists():
                continue
            
            # Find trajectory CSV
            traj_files = list(episode_dir.glob("traj_ep_*.csv"))
            if not traj_files:
                continue
            
            traj_file = traj_files[0]
            
            try:
                traj_df = pd.read_csv(traj_file)
                
                # Check required columns
                if not all(col in traj_df.columns for col in ['timestep', 'agent', 'lat', 'lon']):
                    continue
                
                # Calculate lateral overlap for this episode
                p_y_episode = self._calculate_lateral_overlap_from_trajectory(traj_df)
                if p_y_episode is not None:
                    episode_p_y_values.append(p_y_episode)
                    
            except Exception as e:
                print(f"      ⚠️  Error processing {traj_file.name}: {e}")
                continue
        
        if not episode_p_y_values:
            return None
        
        # Aggregate across episodes
        p_y_mean = np.mean(episode_p_y_values)
        
        # Bootstrap CI
        if len(episode_p_y_values) >= 10:
            bootstrap_samples = []
            for _ in range(1000):
                sample = np.random.choice(episode_p_y_values, size=len(episode_p_y_values), replace=True)
                bootstrap_samples.append(np.mean(sample))
            
            ci_lower = np.percentile(bootstrap_samples, 5)
            ci_upper = np.percentile(bootstrap_samples, 95)
        else:
            ci_lower, ci_upper = p_y_mean, p_y_mean
        
        return p_y_mean, ci_lower, ci_upper
    
    def _calculate_lateral_overlap_from_trajectory(self, traj_df: pd.DataFrame) -> Optional[float]:
        """
        Calculate lateral overlap fraction from trajectory data.
        
        For each timestep, for each pair, compute cross-track separation and
        check if |y_lateral| < 5 NM.
        
        Args:
            traj_df: Trajectory DataFrame with columns: timestep, agent, lat, lon, heading, speed
            
        Returns:
            Fraction of (pair, timestep) samples with lateral overlap, or None if data insufficient
        """
        # Group by timestep
        timesteps = sorted(traj_df['timestep'].unique())
        
        overlap_count = 0
        total_pairs_timesteps = 0
        
        for t in timesteps:
            t_data = traj_df[traj_df['timestep'] == t]
            
            if len(t_data) < 2:
                continue
            
            agents = t_data['agent'].unique()
            
            # Check all pairs
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
                    
                    # If we have heading data, calculate lateral component
                    if 'heading' in row_i and 'heading' in row_j and pd.notna(row_i['heading']):
                        # Bearing from i to j
                        brg_ij = bearing_deg(lat_i, lon_i, lat_j, lon_j)
                        
                        # Heading of aircraft i
                        hdg_i = row_i['heading']
                        
                        # Angle between i's track and pair line
                        track_angle = angle_diff(brg_ij, hdg_i)
                        
                        # Lateral component (cross-track)
                        lateral_sep = hmd * abs(math.sin(math.radians(track_angle)))
                    else:
                        # No heading data, use total HMD as approximation
                        lateral_sep = hmd
                    
                    total_pairs_timesteps += 1
                    
                    if lateral_sep < S_Y_NM:
                        overlap_count += 1
        
        if total_pairs_timesteps == 0:
            return None
        
        return overlap_count / total_pairs_timesteps
    
    def estimate_speed_parameters(self, episodes_df: pd.DataFrame, 
                                  scenario_name: str, 
                                  Sx_nm: float = S_X_NM) -> Dict[str, float]:
        """
        Estimate speed and flow parameters from episodes or scenario defaults.
        
        Args:
            episodes_df: DataFrame with episode metrics
            scenario_name: Test scenario name for geometry lookup
            Sx_nm: Longitudinal window for occupancy calculation
            
        Returns:
            Dictionary with keys: flow_same, flow_opp, delta_v_kt, ydot_kt, zdot_kt, v_avg_kt
        """
        # Calculate average ground speed from path length and flight time
        valid_episodes = episodes_df[
            (episodes_df['flight_time_s'] > 0) & 
            (episodes_df['total_path_length_nm'] > 0)
        ]
        
        if len(valid_episodes) > 0:
            speeds_kt = valid_episodes['total_path_length_nm'] / (valid_episodes['flight_time_s'] / 3600.0)
            v_avg_kt = speeds_kt.mean()
        else:
            v_avg_kt = DEFAULT_SPEED_KT
        
        # Get scenario geometry defaults
        scenario_geom = SCENARIO_GEOMETRY.get(scenario_name, SCENARIO_GEOMETRY['canonical_crossing'])
        
        # Flow rates (aircraft per hour)
        flow_same = scenario_geom['flow_same']
        flow_opp = scenario_geom['flow_opp']
        
        # Speed parameters (kt = NM/h)
        delta_v_kt = v_avg_kt * scenario_geom['delta_v_factor']
        if delta_v_kt < SPEED_EPSILON:
            delta_v_kt = SPEED_EPSILON
        
        ydot_kt = v_avg_kt * scenario_geom['ydot_factor']
        zdot_kt = 0.0  # Same flight level (2D scenarios)
        
        return {
            'flow_same': flow_same,
            'flow_opp': flow_opp,
            'delta_v_kt': delta_v_kt,
            'ydot_kt': ydot_kt,
            'zdot_kt': zdot_kt,
            'v_avg_kt': v_avg_kt,
            'scenario_angle': scenario_geom['angle_deg'],
            'scenario_desc': scenario_geom['description'],
            'Sx_nm': Sx_nm
        }
    
    def calculate_n_ay(self, p_y: float, speed_params: Dict[str, float]) -> float:
        """
        Calculate lateral collision risk rate using ICAO/Eurocontrol CRM formula.
        
        ICAO Reich Lateral CRM (Equation 1):
        N_ay = P_y(S_y) * P_z(0) * (λ_x / S_x) * [
            E_same * (|Δv|/(2λ_x) + |ẏ|/(2λ_y) + |ż|/(2λ_z)) +
            E_opp  * (2|v|/(2λ_x) + |ẏ|/(2λ_y) + |ż|/(2λ_z))
        ]
        
        Where:
        - E_same, E_opp are OCCUPANCIES (avg aircraft in ±S_x window): E = q * 2*S_x / v
        - Speeds are in NM/h (knots)
        - S_x is longitudinal window
        - Result is per flight hour
        
        Args:
            p_y: Probability of lateral overlap within S_y
            speed_params: Dictionary with flow_same, flow_opp, speeds (kt), Sx_nm
            
        Returns:
            N_ay collision risk rate per flight hour
        """
        # Extract parameters
        v_avg_kt = speed_params['v_avg_kt']
        Sx_nm = speed_params['Sx_nm']
        
        # Calculate OCCUPANCIES (average number of aircraft in ±Sx longitudinal window)
        # E = q * (2 * S_x) / v  where q is flow rate (ac/h), v is speed (NM/h)
        E_same = speed_params['flow_same'] * (2.0 * Sx_nm) / max(SPEED_EPSILON, v_avg_kt)
        E_opp = speed_params['flow_opp'] * (2.0 * Sx_nm) / max(SPEED_EPSILON, v_avg_kt)
        
        # Speed parameters (already in kt = NM/h)
        delta_v_kt = speed_params['delta_v_kt']  # |Δv| for same-direction pairs
        ydot_kt = speed_params['ydot_kt']        # |ẏ| lateral closure rate
        zdot_kt = speed_params['zdot_kt']        # |ż| vertical rate (≈0 at same FL)
        
        # CRM formula brackets
        # Same direction: |Δv|/(2λ_x) + |ẏ|/(2λ_y) + |ż|/(2λ_z)
        bracket_same = (
            delta_v_kt / (2.0 * LAMBDA_X) +
            ydot_kt / (2.0 * LAMBDA_Y) +
            zdot_kt / (2.0 * LAMBDA_Z)
        )
        
        # Opposite direction: 2|v|/(2λ_x) + |ẏ|/(2λ_y) + |ż|/(2λ_z)
        # Note: Use 2*v_avg for head-on closure rate
        bracket_opp = (
            (2.0 * v_avg_kt) / (2.0 * LAMBDA_X) +
            ydot_kt / (2.0 * LAMBDA_Y) +
            zdot_kt / (2.0 * LAMBDA_Z)
        )
        
        # ICAO formula: P_y * P_z * (λ_x / S_x) * [E_same * bracket_same + E_opp * bracket_opp]
        n_ay = p_y * PZ_SAME_FL * (LAMBDA_X / Sx_nm) * (
            E_same * bracket_same + E_opp * bracket_opp
        )
        
        # Result is per flight hour (speeds in NM/h, no conversion needed)
        return n_ay
    
    def compute_crm_for_group(self, group_df: pd.DataFrame, 
                              model_alias: str, baseline_scenario: str,
                              test_scenario: str, model_type: str,
                              source: str) -> Dict[str, any]:
        """
        Compute CRM metrics for a group of episodes.
        
        Args:
            group_df: DataFrame with episodes for this group
            model_alias: Model identifier
            baseline_scenario: Scenario model was trained on
            test_scenario: Scenario being tested
            model_type: 'baseline', 'frozen', or 'generic'
            source: Data source ('intershift' or 'intrashift')
            
        Returns:
            Dictionary with CRM results
        """
        # Step 1: Calculate P_y (lateral overlap probability)
        p_y_tier_a, p_y_ci_lower_a, p_y_ci_upper_a = self.calculate_p_y_tier_a(group_df)
        
        p_y = p_y_tier_a
        p_y_ci_lower = p_y_ci_lower_a
        p_y_ci_upper = p_y_ci_upper_a
        p_y_method = "Tier A (episode min_sep)"
        
        # Try Tier B if enabled
        if self.use_tier_b:
            for data_dir in self.data_dirs.values():
                tier_b_result = self.calculate_p_y_tier_b(group_df, Path(data_dir))
                if tier_b_result is not None:
                    p_y, p_y_ci_lower, p_y_ci_upper = tier_b_result
                    p_y_method = "Tier B (trajectory lateral)"
                    break
        
        # Step 2: Estimate speed parameters
        speed_params = self.estimate_speed_parameters(group_df, test_scenario, self.Sx_nm)
        
        # Apply flow rate overrides if specified
        if self.flow_same_override is not None:
            speed_params['flow_same'] = self.flow_same_override
        if self.flow_opp_override is not None:
            speed_params['flow_opp'] = self.flow_opp_override
        
        # Step 3: Calculate N_ay
        n_ay_mean = self.calculate_n_ay(p_y, speed_params)
        n_ay_ci_lower = self.calculate_n_ay(p_y_ci_lower, speed_params)
        n_ay_ci_upper = self.calculate_n_ay(p_y_ci_upper, speed_params)
        
        # Step 4: TLS comparison
        tls_flag = "PASS" if n_ay_mean <= TLS_MAX else "FAIL"
        tls_margin = (TLS_MAX - n_ay_mean) / TLS_MAX * 100 if n_ay_mean > 0 else 0
        
        # Step 5: Baseline comparison
        baseline_key = (model_alias, baseline_scenario)
        n_ay_delta_vs_baseline_pct = None
        
        if baseline_key in self.baseline_stats and model_type != 'baseline':
            baseline_p_y = self.baseline_stats[baseline_key]['p_y_5nm']
            baseline_speed_params = self.estimate_speed_parameters(
                self.df_detailed[
                    (self.df_detailed['model_alias'] == model_alias) &
                    (self.df_detailed['test_scenario'] == baseline_scenario)
                ],
                baseline_scenario
            )
            baseline_n_ay = self.calculate_n_ay(baseline_p_y, baseline_speed_params)
            
            if baseline_n_ay > 0:
                n_ay_delta_vs_baseline_pct = (n_ay_mean - baseline_n_ay) / baseline_n_ay * 100
        
        # Aggregate other metrics
        result = {
            'model_alias': model_alias,
            'baseline_scenario': baseline_scenario,
            'test_scenario': test_scenario,
            'model_type': model_type,
            'source': source,
            'n_episodes': len(group_df),
            
            # P_y metrics
            'p_y_5nm': p_y,
            'p_y_ci_lower': p_y_ci_lower,
            'p_y_ci_upper': p_y_ci_upper,
            'p_y_method': p_y_method,
            
            # Speed and flow parameters
            'v_avg_kt': speed_params['v_avg_kt'],
            'delta_v_kt': speed_params['delta_v_kt'],
            'ydot_kt': speed_params['ydot_kt'],
            'flow_same_per_h': speed_params['flow_same'],
            'flow_opp_per_h': speed_params['flow_opp'],
            'Sx_nm': speed_params['Sx_nm'],
            'scenario_angle_deg': speed_params['scenario_angle'],
            
            # CRM results
            'n_ay_per_fh': n_ay_mean,
            'n_ay_ci_lower': n_ay_ci_lower,
            'n_ay_ci_upper': n_ay_ci_upper,
            
            # Comparisons
            'tls_flag': tls_flag,
            'tls_margin_pct': tls_margin,
            'n_ay_delta_vs_baseline_pct': n_ay_delta_vs_baseline_pct,
            
            # Supporting metrics
            'mean_min_separation_nm': group_df['min_separation_nm'].mean(),
            'mean_flight_time_s': group_df['flight_time_s'].mean(),
            'los_event_rate_per_ep': group_df['num_los_events'].mean(),
        }
        
        return result
    
    def compute_all_crm_results(self):
        """Compute CRM results for all model/scenario combinations."""
        print("\n🧮 Computing CRM collision risk metrics...")
        
        # Group by model, scenario, type
        groups = self.df_detailed.groupby([
            'model_alias', 'baseline_scenario', 'test_scenario', 'model_type', 'source'
        ])
        
        self.crm_results = []
        
        for (model_alias, baseline_scenario, test_scenario, model_type, source), group in groups:
            print(f"   Processing: {model_alias} / {test_scenario} ({model_type})")
            
            result = self.compute_crm_for_group(
                group, model_alias, baseline_scenario, 
                test_scenario, model_type, source
            )
            
            self.crm_results.append(result)
        
        print(f"\n✅ Computed CRM for {len(self.crm_results)} configurations")
    
    def export_results(self, output_dir: str):
        """Export CRM results to CSV and summary markdown."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Exporting CRM results to {output_dir}...")
        
        # Convert to DataFrame
        df_crm = pd.DataFrame(self.crm_results)
        
        # Sort by model, scenario
        df_crm = df_crm.sort_values(['model_alias', 'test_scenario', 'model_type'])
        
        # Export CSV
        csv_path = output_path / "crm_results.csv"
        df_crm.to_csv(csv_path, index=False, float_format='%.10e')
        print(f"   ✅ Saved: {csv_path.name}")
        
        # Generate summary markdown
        self._generate_summary_markdown(df_crm, output_path)
        
        # Generate visualizations
        self._generate_visualizations(df_crm, output_path)
        
        print(f"\n✅ CRM analysis complete!")
    
    def _generate_summary_markdown(self, df_crm: pd.DataFrame, output_path: Path):
        """Generate comprehensive summary markdown report."""
        md_path = output_path / "crm_results_summary.md"
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# CRM Collision Risk Analysis Summary\n\n")
            f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Target Level of Safety (TLS)\n\n")
            f.write(f"- **TLS Range:** {TLS_MIN:.2e} to {TLS_MAX:.2e} fatal accidents per flight hour\n")
            f.write(f"- **Separation Standard:** {S_Y_NM} NM lateral\n")
            f.write(f"- **Aircraft Geometry:** A320-like (λ_x={LAMBDA_X:.4f}, λ_y={LAMBDA_Y:.4f}, λ_z={LAMBDA_Z:.4f} NM)\n\n")
            
            # Overall statistics
            f.write("## Overall Statistics\n\n")
            f.write(f"- **Total Configurations:** {len(df_crm)}\n")
            f.write(f"- **Total Episodes:** {df_crm['n_episodes'].sum()}\n")
            f.write(f"- **TLS Pass Rate:** {(df_crm['tls_flag'] == 'PASS').sum() / len(df_crm) * 100:.1f}%\n\n")
            
            # Per-scenario summary
            f.write("## Per-Scenario Results\n\n")
            
            for scenario in sorted(df_crm['test_scenario'].unique()):
                scenario_df = df_crm[df_crm['test_scenario'] == scenario]
                
                f.write(f"### {scenario.replace('_', ' ').title()}\n\n")
                
                # Baseline results
                baseline_df = scenario_df[scenario_df['model_type'] == 'baseline']
                if len(baseline_df) > 0:
                    f.write("**Baseline Performance:**\n\n")
                    for _, row in baseline_df.iterrows():
                        f.write(f"- Model: `{row['model_alias']}`\n")
                        f.write(f"  - N_ay: {row['n_ay_per_fh']:.4e} per FH\n")
                        f.write(f"  - TLS: **{row['tls_flag']}** (margin: {row['tls_margin_pct']:.1f}%)\n")
                        f.write(f"  - P_y(5): {row['p_y_5nm']:.4f} ({row['p_y_method']})\n")
                        f.write(f"  - Avg Speed: {row['v_avg_kt']:.1f} kt\n\n")
                
                # Shift results
                shift_df = scenario_df[scenario_df['model_type'].isin(['frozen', 'generic'])]
                if len(shift_df) > 0:
                    f.write("**Distribution Shift Performance:**\n\n")
                    
                    # Group by model
                    for model in shift_df['model_alias'].unique():
                        model_df = shift_df[shift_df['model_alias'] == model]
                        f.write(f"- Model: `{model}`\n")
                        
                        for _, row in model_df.iterrows():
                            f.write(f"  - Type: {row['model_type']}\n")
                            f.write(f"    - N_ay: {row['n_ay_per_fh']:.4e} per FH\n")
                            f.write(f"    - TLS: **{row['tls_flag']}**\n")
                            
                            if pd.notna(row['n_ay_delta_vs_baseline_pct']):
                                direction = "↑" if row['n_ay_delta_vs_baseline_pct'] > 0 else "↓"
                                f.write(f"    - Δ vs Baseline: {direction} {abs(row['n_ay_delta_vs_baseline_pct']):.1f}%\n")
                            
                            f.write(f"    - P_y(5): {row['p_y_5nm']:.4f}\n\n")
                
                f.write("\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Worst performers
            worst = df_crm.nlargest(5, 'n_ay_per_fh')
            f.write("**Highest Risk Configurations:**\n\n")
            for i, (_, row) in enumerate(worst.iterrows(), 1):
                f.write(f"{i}. {row['model_alias']} on {row['test_scenario']} ({row['model_type']}): "
                       f"{row['n_ay_per_fh']:.4e} per FH - **{row['tls_flag']}**\n")
            f.write("\n")
            
            # Best performers
            best = df_crm.nsmallest(5, 'n_ay_per_fh')
            f.write("**Lowest Risk Configurations:**\n\n")
            for i, (_, row) in enumerate(best.iterrows(), 1):
                f.write(f"{i}. {row['model_alias']} on {row['test_scenario']} ({row['model_type']}): "
                       f"{row['n_ay_per_fh']:.4e} per FH - **{row['tls_flag']}**\n")
            f.write("\n")
            
            # Scenario geometry reference
            f.write("## Scenario Geometry Reference\n\n")
            f.write("| Scenario | Angle | Flow Same (ac/h) | Flow Opp (ac/h) | Description |\n")
            f.write("|----------|-------|------------------|-----------------|-------------|\n")
            for scenario_name, geom in SCENARIO_GEOMETRY.items():
                f.write(f"| {scenario_name} | {geom['angle_deg']}° | {geom['flow_same']} | "
                       f"{geom['flow_opp']} | {geom['description']} |\n")
            f.write("\n")
            
            # ICAO Formula reference
            f.write("## ICAO Reich Lateral CRM Formula\n\n")
            f.write("```\n")
            f.write("N_ay = P_y(S_y) * P_z(0) * (λ_x / S_x) * [\n")
            f.write("    E_same * (|Δv|/(2λ_x) + |ẏ|/(2λ_y) + |ż|/(2λ_z)) +\n")
            f.write("    E_opp  * (2|v|/(2λ_x) + |ẏ|/(2λ_y) + |ż|/(2λ_z))\n")
            f.write("]\n")
            f.write("```\n\n")
            f.write(f"- **S_x**: {S_X_NM} NM (longitudinal window)\n")
            f.write(f"- **λ_x**: {LAMBDA_X:.4f} NM (aircraft length)\n")
            f.write(f"- **λ_y**: {LAMBDA_Y:.4f} NM (wingspan)\n")
            f.write(f"- **λ_z**: {LAMBDA_Z:.4f} NM (height)\n")
            f.write("- **E**: Occupancy = q * 2*S_x / v (avg aircraft in ±S_x)\n")
            f.write("- **Speeds**: In NM/h (knots)\n")
            f.write("\n")
        
        print(f"   ✅ Saved: {md_path.name}")
    
    def _generate_visualizations(self, df_crm: pd.DataFrame, output_path: Path):
        """Generate CRM visualization plots."""
        print("\n📊 Generating visualizations...")
        
        # 1. N_ay comparison plot
        self._plot_nay_comparison(df_crm, output_path)
        
        # 2. TLS compliance heatmap
        self._plot_tls_heatmap(df_crm, output_path)
        
        # 3. P_y vs N_ay scatter
        self._plot_py_vs_nay(df_crm, output_path)
        
        # 4. Baseline vs shift comparison
        self._plot_baseline_vs_shift(df_crm, output_path)
    
    def _plot_nay_comparison(self, df_crm: pd.DataFrame, output_path: Path):
        """Plot N_ay comparison across scenarios."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        scenarios = sorted(df_crm['test_scenario'].unique())
        x_pos = np.arange(len(scenarios))
        width = 0.25
        
        # Group by model type
        baseline_data = []
        frozen_data = []
        generic_data = []
        
        for scenario in scenarios:
            scenario_df = df_crm[df_crm['test_scenario'] == scenario]
            
            baseline_val = scenario_df[scenario_df['model_type'] == 'baseline']['n_ay_per_fh'].mean()
            frozen_val = scenario_df[scenario_df['model_type'] == 'frozen']['n_ay_per_fh'].mean()
            generic_val = scenario_df[scenario_df['model_type'] == 'generic']['n_ay_per_fh'].mean()
            
            baseline_data.append(baseline_val if not np.isnan(baseline_val) else 0)
            frozen_data.append(frozen_val if not np.isnan(frozen_val) else 0)
            generic_data.append(generic_val if not np.isnan(generic_val) else 0)
        
        # Plot bars
        ax.bar(x_pos - width, baseline_data, width, label='Baseline', color='#2ecc71', alpha=0.8)
        ax.bar(x_pos, frozen_data, width, label='Frozen (Shift)', color='#e74c3c', alpha=0.8)
        ax.bar(x_pos + width, generic_data, width, label='Generic', color='#3498db', alpha=0.8)
        
        # TLS threshold lines
        ax.axhline(y=TLS_MAX, color='red', linestyle='--', linewidth=2, label=f'TLS Max ({TLS_MAX:.2e})')
        ax.axhline(y=TLS_MIN, color='orange', linestyle=':', linewidth=2, label=f'TLS Min ({TLS_MIN:.2e})')
        
        ax.set_xlabel('Test Scenario', fontweight='bold')
        ax.set_ylabel('N_ay (per flight hour)', fontweight='bold')
        ax.set_title('Lateral Collision Risk Rate (N_ay) by Scenario and Model Type', 
                    fontweight='bold', pad=20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], rotation=0)
        ax.legend(loc='upper left')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        save_path = output_path / "crm_nay_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
    
    def _plot_tls_heatmap(self, df_crm: pd.DataFrame, output_path: Path):
        """Plot TLS compliance heatmap."""
        # Create pivot table: rows=models, cols=scenarios, values=TLS flag
        pivot_data = []
        
        for model in sorted(df_crm['model_alias'].unique()):
            row_data = {'model': model}
            for scenario in sorted(df_crm['test_scenario'].unique()):
                subset = df_crm[
                    (df_crm['model_alias'] == model) & 
                    (df_crm['test_scenario'] == scenario)
                ]
                if len(subset) > 0:
                    # Use log10(N_ay / TLS_MAX) for color scale
                    n_ay = subset['n_ay_per_fh'].mean()
                    row_data[scenario] = np.log10(n_ay / TLS_MAX) if n_ay > 0 else -10
                else:
                    row_data[scenario] = np.nan
            pivot_data.append(row_data)
        
        pivot_df = pd.DataFrame(pivot_data).set_index('model')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
                   cbar_kws={'label': 'log10(N_ay / TLS_max)'}, ax=ax,
                   vmin=-2, vmax=2, linewidths=0.5)
        
        ax.set_title('TLS Compliance Heatmap\n(Negative = Safe, Positive = Exceeds TLS)',
                    fontweight='bold', pad=20)
        ax.set_xlabel('Test Scenario', fontweight='bold')
        ax.set_ylabel('Model', fontweight='bold')
        
        plt.tight_layout()
        save_path = output_path / "crm_tls_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
    
    def _plot_py_vs_nay(self, df_crm: pd.DataFrame, output_path: Path):
        """Scatter plot of P_y vs N_ay."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color by model type
        colors = {'baseline': '#2ecc71', 'frozen': '#e74c3c', 'generic': '#3498db'}
        
        for model_type, color in colors.items():
            subset = df_crm[df_crm['model_type'] == model_type]
            if len(subset) > 0:
                ax.scatter(subset['p_y_5nm'], subset['n_ay_per_fh'], 
                          c=color, label=model_type.title(), alpha=0.6, s=100)
        
        ax.axhline(y=TLS_MAX, color='red', linestyle='--', linewidth=2, label=f'TLS Max')
        
        ax.set_xlabel('P_y(5 NM) - Lateral Overlap Probability', fontweight='bold')
        ax.set_ylabel('N_ay (per flight hour)', fontweight='bold')
        ax.set_title('Lateral Overlap Probability vs Collision Risk Rate', 
                    fontweight='bold', pad=20)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = output_path / "crm_py_vs_nay.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")
    
    def _plot_baseline_vs_shift(self, df_crm: pd.DataFrame, output_path: Path):
        """Plot baseline vs shift N_ay delta."""
        # Filter for entries with baseline comparison
        # Note: model_type might be 'shift' not 'frozen' depending on data source
        shift_df = df_crm[
            (df_crm['model_type'].isin(['frozen', 'shift', 'generic'])) &
            (df_crm['n_ay_delta_vs_baseline_pct'].notna())
        ].copy()
        
        if len(shift_df) == 0:
            print("   ⚠️  No baseline comparisons available")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by model and scenario
        shift_df = shift_df.sort_values('n_ay_delta_vs_baseline_pct', ascending=False)
        
        y_pos = np.arange(len(shift_df))
        colors = ['red' if x > 0 else 'green' for x in shift_df['n_ay_delta_vs_baseline_pct']]
        
        ax.barh(y_pos, shift_df['n_ay_delta_vs_baseline_pct'], color=colors, alpha=0.7)
        
        # Labels
        labels = [f"{row['model_alias'][:15]} / {row['test_scenario']}" 
                 for _, row in shift_df.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Δ N_ay vs Baseline (%)', fontweight='bold')
        ax.set_title('Collision Risk Change: Distribution Shift vs Baseline\n(Red = Increased Risk, Green = Decreased Risk)',
                    fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        save_path = output_path / "crm_baseline_vs_shift.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Saved: {save_path.name}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute lateral CRM collision risk analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze intershift data only
  python crm_collision_risk.py --data_dir "out2" --output "crm_analysis"
  
  # Analyze both intershift and intrashift
  python crm_collision_risk.py --intershift "out2" --intrashift "results" --output "crm_full"
  
  # Use Tier A only (faster, no trajectory loading)
  python crm_collision_risk.py --data_dir "out2" --no-tier-b --output "crm_tier_a"
        """
    )
    
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        help='Single data directory (for backward compatibility)'
    )
    
    parser.add_argument(
        '--intershift',
        type=str,
        help='Intershift data directory (e.g., out2)'
    )
    
    parser.add_argument(
        '--intrashift',
        type=str,
        help='Intrashift data directory (e.g., results)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='crm_analysis',
        help='Output directory for CRM analysis results. Default: crm_analysis'
    )
    
    parser.add_argument(
        '--no-tier-b',
        action='store_true',
        help='Disable Tier B trajectory-based P_y calculation (faster)'
    )
    
    parser.add_argument(
        '--Sx-nm',
        type=float,
        default=S_X_NM,
        help=f'Longitudinal window for occupancy calculation (NM). Default: {S_X_NM}'
    )
    
    parser.add_argument(
        '--flow-same',
        type=float,
        help='Override same-direction flow rate (aircraft/hour). Uses scenario defaults if not specified.'
    )
    
    parser.add_argument(
        '--flow-opp',
        type=float,
        help='Override opposite-direction flow rate (aircraft/hour). Uses scenario defaults if not specified.'
    )
    
    args = parser.parse_args()
    
    # Build data directories dictionary
    data_dirs = {}
    
    if args.data_dir:
        data_dirs['primary'] = args.data_dir
    
    if args.intershift:
        data_dirs['intershift'] = args.intershift
    
    if args.intrashift:
        data_dirs['intrashift'] = args.intrashift
    
    if not data_dirs:
        print("❌ Error: Must specify at least one data directory")
        print("   Use --data_dir, --intershift, or --intrashift")
        return 1
    
    # Validate directories exist
    for name, path in data_dirs.items():
        if not Path(path).exists():
            print(f"❌ Error: Directory not found: {path}")
            return 1
    
    try:
        # Create analyzer
        analyzer = CRMCollisionRiskAnalyzer(
            data_dirs=data_dirs,
            use_tier_b=not args.no_tier_b,
            Sx_nm=args.Sx_nm,
            flow_same_override=args.flow_same,
            flow_opp_override=args.flow_opp
        )
        
        # Load data
        analyzer.load_data()
        
        # Compute CRM metrics
        analyzer.compute_all_crm_results()
        
        # Export results
        analyzer.export_results(args.output)
        
        print("\n" + "="*70)
        print("✅ CRM Collision Risk Analysis Complete!")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during CRM analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
