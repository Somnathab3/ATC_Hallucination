#!/usr/bin/env python3
"""
Bundle-Based Shift Analysis for Model Performance Evaluation

This module analyzes model performance on shifted environments by grouping
shift types into meaningful bundles and computing key performance indicators
with statistical confidence intervals.

Bundle Categories:
- KINEMATICS: agent dynamics (speed, heading)
- GEOMETRY: start/route geometry (position_closer, position_lateral, waypoint)
- AIRFRAME: plant/model mismatch (aircraft_type)
- ENVIRONMENT: environmental conditions (wind, noise, turbulence)
- CONTROL: timing and control variations (action_frequency)

Key Performance Indicators (KPIs):
1. Episode LoS risk = 1 - S(5 NM) from survival curve
2. LoS events/hour
3. FN rate (missed conflicts)
4. FP rate (false alerts)
5. Interventions/hour OR alert duty cycle
6. Flight time (min) OR extra-path ratio

Usage:
    python shift_bundle_analysis.py --data_dir "results" --scenarios "head_on,parallel,t_formation" --output "bundle_analysis"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
import glob
import os

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

# Bundle mapping according to specifications
BUNDLE_MAP = {
    "speed": "Kinematics",
    "heading": "Kinematics",
    "position_closer": "Geometry", 
    "position_lateral": "Geometry",
    "waypoint": "Geometry",
    "aircraft_type": "Airframe",
    "wind": "Environment",
    "noise": "Environment",
    "turbulence": "Environment",
    "action_frequency": "Control"
}

# Color palette for bundles
BUNDLE_COLORS = {
    "Kinematics": "#1f77b4",    # Blue
    "Geometry": "#ff7f0e",      # Orange
    "Airframe": "#2ca02c",      # Green
    "Environment": "#d62728",   # Red
    "Control": "#9467bd"        # Purple
}

class ShiftBundleAnalyzer:
    def __init__(self, data_files: List[str]):
        """Initialize with list of CSV data files."""
        self.data_files = data_files
        self.df = None
        self.baseline_stats = {}
        self.load_and_prepare_data()
        
    def load_and_prepare_data(self):
        """Load and prepare data with bundle assignments."""
        
        print("🔄 Loading shift test data...")
        
        dfs = []
        for file_path in self.data_files:
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue
                
            # Extract scenario name from file path
            file_name = Path(file_path).parent.name
            if "targeted_shift_analysis_" in file_name:
                scenario = file_name.replace("targeted_shift_analysis_", "").split("_")[0]
            else:
                scenario = "unknown"
                
            df_temp = pd.read_csv(file_path)
            df_temp['scenario'] = scenario
            dfs.append(df_temp)
            print(f"   📂 Loaded {len(df_temp)} episodes from {scenario}")
        
        if not dfs:
            raise ValueError("No valid data files found")
            
        self.df = pd.concat(dfs, ignore_index=True)
        
        # Add bundle assignments
        self.df["bundle"] = self.df["shift_type"].map(BUNDLE_MAP)
        
        # Handle range assignment (aircraft_type gets special treatment)
        self.df["range_category"] = np.where(
            self.df["shift_type"] == "aircraft_type", 
            "airframe", 
            self.df["shift_range"]
        )
        
        # Convert shift values to magnitude for direction-collapse
        self.df["shift_magnitude"] = np.abs(self.df["shift_value"])
        
        # Calculate derived KPIs
        self._calculate_kpis()
        
        # Calculate baseline statistics for each scenario
        self._calculate_baseline_stats()
        
        print(f"✅ Prepared data: {len(self.df)} episodes across {self.df['scenario'].nunique()} scenarios")
        print(f"   Bundles: {list(self.df['bundle'].dropna().unique())}")
        
    def _calculate_kpis(self):
        """Calculate key performance indicators."""
        
        # 1. Episode LoS risk = 1 - S(5 NM)
        # For simplicity, use binary indicator: did episode have min_separation < 5 NM?
        self.df['episode_los_risk'] = (self.df['min_separation_nm'] < 5.0).astype(float)
        
        # 2. LoS events per hour
        self.df['los_events_per_hour'] = self.df['num_los_events'] / (self.df['flight_time_s'] / 3600.0 + 1e-6)
        
        # 3. FN rate (missed conflicts) - already in data as 'missed_conflict'
        # (This is fn / (tp + fn))
        
        # 4. FP rate (false alerts) - compute as fp / (fp + tn)
        self.df['fp_rate'] = self.df['fp'] / (self.df['fp'] + self.df['tn'] + 1e-6)
        
        # 5. Interventions per hour
        self.df['interventions_per_hour'] = self.df['num_interventions'] / (self.df['flight_time_s'] / 3600.0 + 1e-6)
        
        # 6. Flight time in minutes
        self.df['flight_time_min'] = self.df['flight_time_s'] / 60.0
        
        # 7. Accuracy = (tp + tn) / (tp + fp + fn + tn)
        total_predictions = self.df['tp'] + self.df['fp'] + self.df['fn'] + self.df['tn']
        self.df['accuracy'] = (self.df['tp'] + self.df['tn']) / (total_predictions + 1e-6)
        
        # 8. Precision = tp / (tp + fp) - already available in data
        # 9. Recall = tp / (tp + fn) - already available in data  
        # 10. F1-score - already available in data
        
        # Additional: Extra path ratio is already available as 'avg_extra_path_ratio'
        # waypoint_reached_ratio, reward_total, min_separation_nm, num_interventions already in data
        
    def _calculate_baseline_stats(self):
        """Calculate baseline statistics for each scenario for comparison."""
        
        baseline_df = self.df[self.df['test_id'] == 'baseline']
        
        for scenario in self.df['scenario'].unique():
            scenario_baseline = baseline_df[baseline_df['scenario'] == scenario]
            
            if len(scenario_baseline) > 0:
                self.baseline_stats[scenario] = {
                    'episode_los_risk': scenario_baseline['episode_los_risk'].mean(),
                    'los_events_per_hour': scenario_baseline['los_events_per_hour'].mean(),
                    'missed_conflict': scenario_baseline['missed_conflict'].mean(),
                    'fp_rate': scenario_baseline['fp_rate'].mean(),
                    'interventions_per_hour': scenario_baseline['interventions_per_hour'].mean(),
                    'alert_duty_cycle': scenario_baseline['alert_duty_cycle'].mean(),
                    'flight_time_min': scenario_baseline['flight_time_min'].mean(),
                    'avg_extra_path_ratio': scenario_baseline['avg_extra_path_ratio'].mean(),
                    'waypoint_reached_ratio': scenario_baseline['waypoint_reached_ratio'].mean(),
                    'reward_total': scenario_baseline['reward_total'].mean(),
                    'num_interventions': scenario_baseline['num_interventions'].mean(),
                    'min_separation_nm': scenario_baseline['min_separation_nm'].mean(),
                    'accuracy': scenario_baseline['accuracy'].mean(),
                    'f1_score': scenario_baseline['f1_score'].mean(),
                    'precision': scenario_baseline['precision'].mean(),
                    'recall': scenario_baseline['recall'].mean(),
                }
            else:
                print(f"⚠️ No baseline data found for scenario: {scenario}")
    
    def calculate_bundle_statistics(self, scenario: str, confidence_level: float = 0.9) -> pd.DataFrame:
        """Calculate bundle statistics with confidence intervals."""
        
        scenario_data = self.df[
            (self.df['scenario'] == scenario) & 
            (self.df['test_id'] != 'baseline') &  # Exclude baseline
            (self.df['bundle'].notna())
        ].copy()
        
        if len(scenario_data) == 0:
            print(f"⚠️ No shift data found for scenario: {scenario}")
            return pd.DataFrame()
        
        # KPIs to analyze
        kpis = [
            'episode_los_risk',
            'los_events_per_hour', 
            'missed_conflict',
            'fp_rate',
            'alert_duty_cycle',  # Using this instead of interventions_per_hour
            'flight_time_min',
            'waypoint_reached_ratio',
            'reward_total',
            'num_interventions',
            'min_separation_nm',
            'accuracy',
            'f1_score',
            'precision',
            'recall'
        ]
        
        alpha = 1 - confidence_level
        
        results = []
        
        for bundle in scenario_data['bundle'].unique():
            for range_cat in scenario_data[scenario_data['bundle'] == bundle]['range_category'].unique():
                
                subset = scenario_data[
                    (scenario_data['bundle'] == bundle) & 
                    (scenario_data['range_category'] == range_cat)
                ]
                
                if len(subset) == 0:
                    continue
                
                row = {
                    'scenario': scenario,
                    'bundle': bundle,
                    'range_category': range_cat,
                    'n_episodes': len(subset)
                }
                
                # Calculate statistics for each KPI
                for kpi in kpis:
                    values = subset[kpi].dropna()
                    
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        sem_val = stats.sem(values)
                        
                        # 90% confidence interval
                        if len(values) > 1:
                            ci_low, ci_high = stats.t.interval(
                                confidence_level, 
                                len(values) - 1, 
                                loc=mean_val, 
                                scale=sem_val
                            )
                        else:
                            ci_low = ci_high = mean_val
                        
                        # Calculate percentage change vs baseline
                        baseline_val = self.baseline_stats.get(scenario, {}).get(kpi, 0)
                        if baseline_val > 0:
                            pct_change = ((mean_val - baseline_val) / baseline_val) * 100
                        else:
                            pct_change = 0.0
                        
                        row[f'{kpi}_mean'] = mean_val
                        row[f'{kpi}_std'] = std_val
                        row[f'{kpi}_ci_low'] = ci_low
                        row[f'{kpi}_ci_high'] = ci_high
                        row[f'{kpi}_pct_change'] = pct_change
                    else:
                        # Fill with zeros if no data
                        for suffix in ['_mean', '_std', '_ci_low', '_ci_high', '_pct_change']:
                            row[f'{kpi}{suffix}'] = 0.0
                
                results.append(row)
        
        return pd.DataFrame(results)
    
    def create_bundle_performance_plots(self, scenario: str, save_dir: str):
        """Create bundle performance plots for a scenario - split into multiple files."""
        
        bundle_stats = self.calculate_bundle_statistics(scenario)
        
        if bundle_stats.empty:
            print(f"⚠️ No data to plot for scenario: {scenario}")
            return
        
        save_path = Path(save_dir) 
        save_path.mkdir(parents=True, exist_ok=True)
        
        # KPIs to plot - all available metrics
        all_kpis = [
            ('episode_los_risk', 'Episode LoS Risk'),
            ('los_events_per_hour', 'LoS Events/Hour'),
            ('missed_conflict', 'FN Rate (Missed Conflicts)'),
            ('fp_rate', 'FP Rate (False Alerts)'),
            ('alert_duty_cycle', 'Alert Duty Cycle'),
            ('flight_time_min', 'Flight Time (min)'),
            ('waypoint_reached_ratio', 'Waypoint Reached Ratio'),
            ('reward_total', 'Total Reward'),
            ('num_interventions', 'Number of Interventions'),
            ('min_separation_nm', 'Min Separation (NM)'),
            ('accuracy', 'Accuracy'),
            ('f1_score', 'F1 Score'),
            ('precision', 'Precision'),
            ('recall', 'Recall')
        ]
        
        # Split KPIs into groups of 4 for separate plots
        kpi_groups = [
            all_kpis[0:4],   # Group 1: LoS Risk, LoS Events, FN Rate, FP Rate
            all_kpis[4:8],   # Group 2: Alert Duty, Flight Time, Waypoint Ratio, Reward
            all_kpis[8:12],  # Group 3: Interventions, Min Separation, Accuracy, F1 Score
            all_kpis[12:14]  # Group 4: Precision, Recall
        ]
        
        group_names = ['safety', 'efficiency', 'performance', 'classification']
        
        for group_idx, (kpis, group_name) in enumerate(zip(kpi_groups, group_names)):
            # Determine subplot layout
            if len(kpis) == 4:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()
            else:  # 2 metrics in last group
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                if len(kpis) == 1:
                    axes = [axes]
            
            for kpi_idx, (kpi, kpi_title) in enumerate(kpis):
                ax = axes[kpi_idx]
                
                # Get baseline value for reference line
                baseline_val = self.baseline_stats.get(scenario, {}).get(kpi, 0)
                
                # Create consistent x-axis positioning - only for data that exists
                x_labels = []
                x_positions = []
                bundles = sorted(bundle_stats['bundle'].unique())
                
                x_pos = 0
                
                for bundle in bundles:
                    bundle_data = bundle_stats[bundle_stats['bundle'] == bundle]
                    color = BUNDLE_COLORS.get(bundle, '#333333')
                    
                    for range_cat in ['micro', 'macro', 'airframe']:
                        range_data = bundle_data[bundle_data['range_category'] == range_cat]
                        
                        if len(range_data) == 0:
                            # Skip this combination entirely - no gap
                            continue
                            
                        row = range_data.iloc[0]
                        mean_val = row[f'{kpi}_mean']
                        ci_low = row[f'{kpi}_ci_low'] 
                        ci_high = row[f'{kpi}_ci_high']
                        
                        # Plot with error bars
                        if range_cat == 'airframe':
                            marker = 's'  # Square for airframe
                            alpha = 0.8
                            marker_size = 10
                        elif range_cat == 'micro':
                            marker = 'o'  # Circle for micro
                            alpha = 0.9
                            marker_size = 8
                        else:  # macro
                            marker = '^'  # Triangle for macro
                            alpha = 0.7
                            marker_size = 9
                        
                        # Plot error bar
                        ax.errorbar(x_pos, mean_val, 
                                  yerr=[[mean_val - ci_low], [ci_high - mean_val]],
                                  fmt=marker, color=color, alpha=alpha, markersize=marker_size,
                                  capsize=5, capthick=2, linewidth=2)
                        
                        # Add value annotation inside/near the marker
                        if abs(mean_val) >= 10:
                            value_text = f'{mean_val:.1f}'
                        elif abs(mean_val) >= 1:
                            value_text = f'{mean_val:.2f}'
                        else:
                            value_text = f'{mean_val:.3f}'
                        
                        # Position text slightly above the marker
                        text_y = mean_val + (ci_high - mean_val) * 1.3
                        ax.text(x_pos, text_y, value_text, ha='center', va='bottom', 
                               fontsize=10, fontweight='bold', color=color,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8, edgecolor=color))
                        
                        # Add x-axis label and position for each data point
                        x_labels.append(f'{bundle}\n{range_cat.upper()}')
                        x_positions.append(x_pos)
                        x_pos += 1  # Increment position only when we have data
                
                # Add baseline reference line for ALL plots (not just when baseline_val > 0)
                if baseline_val != 0:  # Show baseline even if it's 0
                    ax.axhline(y=baseline_val, color='red', linestyle='--', alpha=0.7, 
                              linewidth=2, label=f'Baseline ({baseline_val:.3f})')
                
                # Customize each subplot
                ax.set_title(kpi_title, fontweight='bold', fontsize=14, pad=15)
                ax.set_ylabel(kpi_title, fontsize=12, fontweight='bold')
                ax.set_xticks(x_positions)
                ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Adjust y-axis limits to accommodate text annotations
                y_min, y_max = ax.get_ylim()
                ax.set_ylim(y_min, y_max * 1.15)
                
                # Add legend to ALL subplots to show baseline consistently
                if baseline_val != 0:
                    ax.legend(fontsize=9, loc='upper right')
            
            # Hide unused subplots if any
            if len(kpis) < len(axes):
                for i in range(len(kpis), len(axes)):
                    axes[i].set_visible(False)
            
            # Add a main title
            group_title = f'{group_name.title()} Metrics'
            plt.suptitle(f'Bundle Performance Analysis - {scenario.replace("_", " ").title()}: {group_title}', 
                        fontweight='bold', fontsize=16, y=0.95)
            
            # Adjust layout with proper spacing
            plt.tight_layout(rect=(0, 0.03, 1, 0.90))
            
            plot_path = save_path / f'bundle_performance_{scenario}_{group_name}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"✅ Saved bundle performance plot: {plot_path.name}")
        
        print(f"📊 Created {len(kpi_groups)} bundle performance plot files for {scenario}")
    
    def create_delta_heatmap(self, scenario: str, save_dir: str):
        """Create compact delta heatmap showing % change vs baseline."""
        
        bundle_stats = self.calculate_bundle_statistics(scenario)
        
        if bundle_stats.empty:
            print(f"⚠️ No data for heatmap: {scenario}")
            return
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # KPIs for heatmap
        kpis = [
            ('episode_los_risk', 'LoS Risk'),
            ('los_events_per_hour', 'LoS Events/hr'),
            ('missed_conflict', 'FN Rate'),
            ('fp_rate', 'FP Rate'),
            ('alert_duty_cycle', 'Alert Duty')
        ]
        
        # Create heatmap data matrix
        heatmap_data = []
        row_labels = []
        
        for kpi, kpi_label in kpis:
            row = []
            row_labels.append(kpi_label)
            
            # For each column: MICRO, MACRO, AIRFRAME
            for col_type in ['micro', 'macro', 'airframe']:
                # Average across all bundles for this column type
                col_data = bundle_stats[bundle_stats['range_category'] == col_type]
                
                if len(col_data) > 0:
                    avg_pct_change = col_data[f'{kpi}_pct_change'].mean()
                    row.append(avg_pct_change)
                else:
                    row.append(0.0)
            
            heatmap_data.append(row)
        
        heatmap_array = np.array(heatmap_data)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use diverging colormap centered at 0
        vmax = max(abs(heatmap_array.min()), abs(heatmap_array.max()))
        im = ax.imshow(heatmap_array, cmap='RdBu_r', aspect='auto', 
                      vmin=-vmax, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('% Change vs Baseline', fontweight='bold')
        
        # Set ticks and labels
        ax.set_xticks(range(3))
        ax.set_xticklabels(['MICRO', 'MACRO', 'AIRFRAME'], fontweight='bold')
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontweight='bold')
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(3):
                value = heatmap_array[i, j]
                color = 'white' if abs(value) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{value:.1f}%', ha='center', va='center', 
                       color=color, fontweight='bold', fontsize=10)
        
        ax.set_title(f'Performance Change vs Baseline - {scenario.replace("_", " ").title()}', 
                    fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        heatmap_path = save_path / f'delta_heatmap_{scenario}.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved delta heatmap: {heatmap_path.name}")
    
    def create_survival_curve_inset(self, scenario: str, save_dir: str):
        """Create survival curve plot for tail-risk analysis with individual shift lines."""
        
        scenario_data = self.df[self.df['scenario'] == scenario].copy()
        
        if len(scenario_data) == 0:
            print(f"⚠️ No data for survival curve: {scenario}")
            return
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get baseline and shifted data
        baseline_data = scenario_data[scenario_data['test_id'] == 'baseline']
        shifted_data = scenario_data[scenario_data['test_id'] != 'baseline']
        
        # Calculate empirical survival functions
        def calculate_survival(separations):
            """Calculate empirical survival function S(x) = P(X >= x)."""
            if len(separations) == 0:
                return np.array([]), np.array([])
            
            x_sorted = np.sort(separations)
            n = len(x_sorted)
            survival_probs = 1 - np.arange(1, n + 1) / n
            
            return x_sorted, survival_probs
        
        # Plot baseline survival curve
        if len(baseline_data) > 0:
            baseline_seps = baseline_data['min_separation_nm'].values
            x_base, s_base = calculate_survival(baseline_seps)
            
            if len(x_base) > 0:
                ax.plot(x_base, s_base, color='red', linewidth=4, 
                       label=f'Baseline (N={len(baseline_seps)})', alpha=0.9, zorder=10)
                
                # Calculate and report S(5) for baseline
                s_5_baseline = np.interp(5.0, x_base, s_base, left=1.0, right=0.0)
        
        # Plot individual shift type survival curves
        if len(shifted_data) > 0:
            # Get unique shift types
            shift_types = shifted_data['shift_type'].unique()
            
            # Color palette for different shift types
            shift_colors = {
                'speed': '#1f77b4',      # Blue
                'heading': '#ff7f0e',    # Orange  
                'position_closer': '#2ca02c',  # Green
                'position_lateral': '#d62728', # Red
                'waypoint': '#9467bd',   # Purple
                'aircraft_type': '#8c564b', # Brown
                'wind': '#e377c2',       # Pink
                'noise': '#7f7f7f',      # Gray
                'turbulence': '#bcbd22', # Olive
                'action_frequency': '#17becf'  # Cyan
            }
            
            shift_s5_values = {}
            
            for shift_type in sorted(shift_types):
                shift_subset = shifted_data[shifted_data['shift_type'] == shift_type]
                shift_seps = shift_subset['min_separation_nm'].values
                
                if len(shift_seps) > 0:
                    x_shift, s_shift = calculate_survival(shift_seps)
                    
                    if len(x_shift) > 0:
                        color = shift_colors.get(shift_type, '#333333')
                        
                        # Use different line styles for better distinction
                        if shift_type in ['speed', 'heading']:  # Kinematics
                            linestyle = '-'
                            alpha = 0.8
                        elif shift_type in ['position_closer', 'position_lateral', 'waypoint']:  # Geometry
                            linestyle = '--'
                            alpha = 0.7
                        elif shift_type == 'aircraft_type':  # Airframe
                            linestyle = '-.'
                            alpha = 0.8
                        else:  # Environment/Control
                            linestyle = ':'
                            alpha = 0.6
                        
                        ax.plot(x_shift, s_shift, color=color, linewidth=2, 
                               linestyle=linestyle, alpha=alpha,
                               label=f'{shift_type.replace("_", " ").title()} (N={len(shift_seps)})')
                        
                        # Calculate S(5) for this shift type
                        s_5_shift = np.interp(5.0, x_shift, s_shift, left=1.0, right=0.0)
                        shift_s5_values[shift_type] = s_5_shift
        
        # Add 5 NM safety threshold
        ax.axvline(x=5.0, color='orange', linestyle='--', linewidth=2, 
                  alpha=0.8, label='5 NM Safety Threshold', zorder=5)
        
        ax.set_xlabel('Minimum Horizontal Separation (NM)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Survival Probability S(x) = P(X ≥ x)', fontweight='bold', fontsize=12)
        ax.set_title(f'Minimum Separation Survival Curves by Shift Type - {scenario.replace("_", " ").title()}', 
                    fontweight='bold', fontsize=14)
        
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Create a more organized legend
        handles, labels = ax.get_legend_handles_labels()
        
        # Separate baseline, threshold, and shift types
        baseline_handles = [h for h, l in zip(handles, labels) if 'Baseline' in l]
        baseline_labels = [l for l in labels if 'Baseline' in l]
        
        threshold_handles = [h for h, l in zip(handles, labels) if 'Threshold' in l]
        threshold_labels = [l for l in labels if 'Threshold' in l]
        
        shift_handles = [h for h, l in zip(handles, labels) if 'Baseline' not in l and 'Threshold' not in l]
        shift_labels = [l for l in labels if 'Baseline' not in l and 'Threshold' not in l]
        
        # Create legend in two columns if there are many shift types
        if len(shift_labels) > 6:
            ncol = 2
            bbox_to_anchor = (1.05, 1)
            loc = 'upper left'
        else:
            ncol = 1
            bbox_to_anchor = (1.02, 1)
            loc = 'upper left'
        
        # Combine all handles and labels in order
        all_handles = baseline_handles + threshold_handles + shift_handles
        all_labels = baseline_labels + threshold_labels + shift_labels
        
        ax.legend(all_handles, all_labels, fontsize=9, loc=loc, 
                 bbox_to_anchor=bbox_to_anchor, ncol=ncol)
        
        # Add S(5) annotations
        try:
            if 's_5_baseline' in locals():
                ax.text(0.02, 0.95, f'Baseline S(5) = {s_5_baseline:.3f}', 
                       transform=ax.transAxes, fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.2))
                
                # Add top 3 worst shift types
                if shift_s5_values:
                    worst_shifts = sorted(shift_s5_values.items(), key=lambda x: x[1])[:3]
                    y_pos = 0.85
                    for shift_type, s5_val in worst_shifts:
                        ax.text(0.02, y_pos, f'{shift_type.replace("_", " ").title()} S(5) = {s5_val:.3f}', 
                               transform=ax.transAxes, fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.3))
                        y_pos -= 0.05
        except:
            pass
        
        plt.tight_layout()
        
        survival_path = save_path / f'survival_curve_{scenario}.png'
        plt.savefig(survival_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Saved survival curve: {survival_path.name}")
    
    def export_bundle_statistics(self, save_dir: str):
        """Export detailed bundle statistics to CSV."""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        all_stats = []
        
        for scenario in self.df['scenario'].unique():
            scenario_stats = self.calculate_bundle_statistics(scenario)
            all_stats.append(scenario_stats)
        
        if all_stats:
            combined_stats = pd.concat(all_stats, ignore_index=True)
            
            stats_path = save_path / 'bundle_statistics_detailed.csv'
            combined_stats.to_csv(stats_path, index=False)
            print(f"✅ Saved detailed statistics: {stats_path.name}")
            
            # Create summary table
            summary_stats = combined_stats.groupby(['scenario', 'bundle', 'range_category']).agg({
                'n_episodes': 'sum',
                'episode_los_risk_mean': 'mean',
                'los_events_per_hour_mean': 'mean',
                'missed_conflict_mean': 'mean',
                'fp_rate_mean': 'mean',
                'alert_duty_cycle_mean': 'mean',
                'flight_time_min_mean': 'mean',
                'waypoint_reached_ratio_mean': 'mean',
                'reward_total_mean': 'mean',
                'num_interventions_mean': 'mean',
                'min_separation_nm_mean': 'mean',
                'accuracy_mean': 'mean',
                'f1_score_mean': 'mean',
                'precision_mean': 'mean',
                'recall_mean': 'mean'
            }).round(4).reset_index()
            
            summary_path = save_path / 'bundle_statistics_summary.csv'
            summary_stats.to_csv(summary_path, index=False)
            print(f"✅ Saved summary statistics: {summary_path.name}")
            
            return combined_stats
        
        return pd.DataFrame()
    
    def generate_full_analysis(self, output_dir: str):
        """Generate complete bundle analysis for all scenarios."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\\n🎨 Generating Bundle-Based Shift Analysis")
        print("=" * 55)
        
        scenarios = sorted(self.df['scenario'].unique())
        
        for scenario in scenarios:
            print(f"\\n📊 {scenario.replace('_', ' ').title()}")
            
            # Create bundle performance plots
            self.create_bundle_performance_plots(scenario, str(output_path))
            
            # Create delta heatmap
            self.create_delta_heatmap(scenario, str(output_path))
            
            # Create survival curve
            self.create_survival_curve_inset(scenario, str(output_path))
        
        # Export statistics
        bundle_stats = self.export_bundle_statistics(str(output_path))
        
        # Create analysis report
        self._generate_analysis_report(str(output_path), scenarios)
        
        print(f"\\n✅ Bundle analysis completed!")
        print(f"📁 Output: {output_path}")
        
        return bundle_stats
    
    def _generate_analysis_report(self, output_dir: str, scenarios: List[str]):
        """Generate comprehensive analysis report."""
        
        report_content = f"""# Bundle-Based Shift Analysis Report

## Overview
This analysis groups distribution shifts into meaningful bundles and evaluates model performance degradation across different shift categories.

## Bundle Categories
- **Kinematics**: Agent dynamics (speed, heading variations)
- **Geometry**: Start/route geometry (position closer/lateral, waypoint shifts)
- **Airframe**: Plant/model mismatch (aircraft type variations)
- **Environment**: Environmental conditions (wind, noise, turbulence)
- **Control**: Timing and control variations (action frequency)

## Range Categories
- **MICRO**: Small perturbations within training envelope
- **MACRO**: Large deviations testing failure modes
- **AIRFRAME**: Aircraft type variations (categorical)

## Key Performance Indicators
1. **Episode LoS Risk**: Fraction of episodes with min separation < 5 NM
2. **LoS Events/Hour**: Frequency of loss of separation events
3. **FN Rate**: Missed conflict rate (false negatives)
4. **FP Rate**: False alert rate (false positives)
5. **Alert Duty Cycle**: Fraction of time in alert state
6. **Flight Time**: Episode duration in minutes

## Scenarios Analyzed
{chr(10).join(f"- **{scenario.replace('_', ' ').title()}**: Bundle performance analysis with 90% confidence intervals" for scenario in scenarios)}

## Files Generated
- `bundle_performance_[scenario].png`: Bundle performance plots with CI bands
- `delta_heatmap_[scenario].png`: Compact heatmap showing % change vs baseline
- `survival_curve_[scenario].png`: Tail-risk analysis with S(5 NM) calculations
- `bundle_statistics_detailed.csv`: Complete statistical analysis
- `bundle_statistics_summary.csv`: Aggregated summary statistics

## Statistical Methods
- 90% confidence intervals using Student's t-distribution
- Baseline comparison with percentage change calculations
- Survival curve analysis for tail-risk evaluation
- Direction-collapsed magnitude analysis for symmetric shifts

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        report_path = Path(output_dir) / 'README.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Saved analysis report: {report_path.name}")


def find_shift_data_files(data_dir: str, scenarios: Optional[List[str]] = None) -> List[str]:
    """Find targeted shift test summary CSV files."""
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Look for targeted shift analysis directories
    pattern = "targeted_shift_analysis_*"
    shift_dirs = list(data_path.glob(pattern))
    
    if not shift_dirs:
        raise FileNotFoundError(f"No targeted shift analysis directories found in {data_dir}")
    
    csv_files = []
    
    for shift_dir in shift_dirs:
        # Extract scenario name
        dir_name = shift_dir.name
        if "targeted_shift_analysis_" in dir_name:
            scenario = dir_name.replace("targeted_shift_analysis_", "").split("_")[0]
        else:
            continue
        
        # Check if this scenario is requested
        if scenarios and scenario not in scenarios:
            continue
        
        # Look for summary CSV
        summary_csv = shift_dir / "targeted_shift_test_summary.csv"
        if summary_csv.exists():
            csv_files.append(str(summary_csv))
            print(f"Found: {scenario} -> {summary_csv}")
        else:
            print(f"⚠️ No summary CSV found in {shift_dir}")
    
    return csv_files


def main():
    parser = argparse.ArgumentParser(
        description='Generate bundle-based shift analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all scenarios in results directory
  python shift_bundle_analysis.py --data_dir "results" --output "bundle_analysis"
  
  # Analyze specific scenarios
  python shift_bundle_analysis.py --data_dir "results" --scenarios "head_on,parallel" --output "bundle_analysis_subset"
  
  # Use custom data files
  python shift_bundle_analysis.py --files "file1.csv,file2.csv" --output "custom_analysis"
        """
    )
    
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        help='Directory containing targeted shift analysis results'
    )
    
    parser.add_argument(
        '--scenarios', '-s',
        type=str,
        help='Comma-separated list of scenarios to analyze (e.g., "head_on,parallel,t_formation")'
    )
    
    parser.add_argument(
        '--files', '-f',
        type=str,
        help='Comma-separated list of CSV files to analyze directly'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='bundle_analysis',
        help='Output directory for analysis results. Default: bundle_analysis'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.9,
        help='Confidence level for intervals (default: 0.9 for 90%% CI)'
    )
    
    args = parser.parse_args()
    
    if not args.data_dir and not args.files:
        print("Error: Must specify either --data_dir or --files")
        return 1
    
    try:
        # Find data files
        if args.files:
            csv_files = [f.strip() for f in args.files.split(',')]
            # Verify files exist
            for f in csv_files:
                if not os.path.exists(f):
                    print(f"Error: File not found: {f}")
                    return 1
        else:
            scenarios = None
            if args.scenarios:
                scenarios = [s.strip() for s in args.scenarios.split(',')]
            
            csv_files = find_shift_data_files(args.data_dir, scenarios)
            
            if not csv_files:
                print("Error: No valid data files found")
                return 1
        
        print(f"Found {len(csv_files)} data files to analyze")
        
        # Create analyzer and run analysis
        analyzer = ShiftBundleAnalyzer(csv_files)
        bundle_stats = analyzer.generate_full_analysis(args.output)
        
        print("\\n" + "="*60)
        print("✅ BUNDLE ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results directory: {args.output}")
        print(f"Scenarios analyzed: {len(analyzer.df['scenario'].unique())}")
        print(f"Total episodes: {len(analyzer.df)}")
        print(f"Bundles identified: {list(analyzer.df['bundle'].dropna().unique())}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())