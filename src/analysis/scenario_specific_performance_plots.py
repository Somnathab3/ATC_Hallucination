#!/usr/bin/env python3
"""
Scenario-Specific Performance Analysis Plots

Creates comprehensive visualizations comparing baseline vs shift model performance
on each scenario, similar to the provided example graphs.

Key Features:
- Scenario-centric analysis (not model-centric)
- Baseline vs Shift comparisons with confidence intervals  
- Safety, efficiency, and performance metrics
- Publication-ready plots with error bars

Usage:
    python scenario_specific_performance_plots.py --data path/to/baseline_vs_shift_detailed_summary.csv --output plots/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ScenarioPerformanceAnalyzer:
    def __init__(self, data_path: str):
        """Initialize analyzer with data."""
        self.df = pd.read_csv(data_path)
        self.scenarios = sorted(self.df['test_scenario'].unique())
        self.prepare_data()
        
    def prepare_data(self):
        """Calculate derived metrics and prepare data for analysis."""
        # Calculate derived metrics using actual data
        self.df['fp_rate'] = self.df['fp'] / (self.df['fp'] + self.df['tn'] + 1e-8)
        self.df['fn_rate'] = self.df['fn'] / (self.df['fn'] + self.df['tp'] + 1e-8)
        self.df['accuracy'] = (self.df['tp'] + self.df['tn']) / (self.df['tp'] + self.df['fp'] + self.df['fn'] + self.df['tn'] + 1e-8)
        
        # Convert flight time to minutes
        self.df['flight_time_min'] = self.df['flight_time_s'] / 60.0
        
        # Calculate model type for each scenario
        self.df['model_scenario_type'] = self.df.apply(
            lambda row: 'Baseline' if row['baseline_scenario'] == row['test_scenario'] else 'Shift', 
            axis=1
        )
        
        # Calculate dynamic ranges for metrics
        self.metric_ranges = {
            'fp_rate': (0, max(0.15, self.df['fp_rate'].quantile(0.95))),
            'fn_rate': (0, max(0.15, self.df['fn_rate'].quantile(0.95))),
            'accuracy': (0, 1),
            'precision': (0, 1),
            'recall': (0, 1),
            'f1_score': (0, 1),
            'ghost_conflict': (0, max(0.5, self.df['ghost_conflict'].quantile(0.95)) if 'ghost_conflict' in self.df.columns else 0.5),
            'missed_conflict': (0, max(0.5, self.df['missed_conflict'].quantile(0.95)) if 'missed_conflict' in self.df.columns else 0.5),
            'alert_duty_cycle': (0, max(0.5, self.df['alert_duty_cycle'].quantile(0.95)) if 'alert_duty_cycle' in self.df.columns else 0.5),
            'alerts_per_min': (0, None),
            'num_los_events': (0, None),
            'flight_time_min': (0, None),
            'waypoint_reached_ratio': (0, 1),
            'num_interventions': (0, None),
            'min_separation_nm': (0, None),
            'reward_total': (None, None)
        }
        
        print(f"📊 Loaded data: {len(self.df)} episodes across {len(self.scenarios)} scenarios")
        print(f"🏠 Baseline episodes: {len(self.df[self.df['model_scenario_type'] == 'Baseline'])}")
        print(f"🔄 Shift episodes: {len(self.df[self.df['model_scenario_type'] == 'Shift'])}")
        
    def calculate_scenario_stats(self, metric: str) -> pd.DataFrame:
        """Calculate mean and confidence intervals for a metric by scenario and model type."""
        stats = []
        
        for scenario in self.scenarios:
            scenario_data = self.df[self.df['test_scenario'] == scenario]
            
            for model_type in ['Baseline', 'Shift']:
                type_data = scenario_data[scenario_data['model_scenario_type'] == model_type]
                
                if len(type_data) > 0:
                    values = type_data[metric].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        n = len(values)
                        
                        # 90% confidence interval
                        ci_90 = 1.645 * std_val / np.sqrt(n) if n > 1 else 0
                        
                        stats.append({
                            'scenario': scenario,
                            'model_type': model_type,
                            'mean': mean_val,
                            'std': std_val,
                            'ci_90': ci_90,
                            'n': n,
                            'min': values.min(),
                            'max': values.max()
                        })
        
        return pd.DataFrame(stats)
    
    def create_comparison_plot(self, metric: str, title: str, ylabel: str, 
                             figsize: Tuple[int, int] = (12, 6), 
                             ylim: Optional[Tuple[float, float]] = None,
                             save_path: Optional[str] = None) -> Optional[matplotlib.figure.Figure]:
        """Create scenario comparison plot with error bars."""
        
        stats_df = self.calculate_scenario_stats(metric)
        
        if stats_df.empty:
            print(f"⚠️ No data found for metric: {metric}")
            return None
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data for plotting
        scenarios = sorted(stats_df['scenario'].unique())
        x_pos = np.arange(len(scenarios))
        width = 0.35
        
        baseline_data = stats_df[stats_df['model_type'] == 'Baseline'].set_index('scenario')
        shift_data = stats_df[stats_df['model_type'] == 'Shift'].set_index('scenario')
        
        # Plot bars with error bars
        baseline_means = [baseline_data.loc[s, 'mean'] if s in baseline_data.index else 0 for s in scenarios]
        baseline_cis = [baseline_data.loc[s, 'ci_90'] if s in baseline_data.index else 0 for s in scenarios]
        
        shift_means = [shift_data.loc[s, 'mean'] if s in shift_data.index else 0 for s in scenarios]
        shift_cis = [shift_data.loc[s, 'ci_90'] if s in shift_data.index else 0 for s in scenarios]
        
        # Create bars
        bars1 = ax.bar(x_pos - width/2, baseline_means, width, 
                      yerr=baseline_cis, capsize=5,
                      label='Baseline Models', color='#2E86AB', alpha=0.8)
        
        bars2 = ax.bar(x_pos + width/2, shift_means, width,
                      yerr=shift_cis, capsize=5, 
                      label='Shift Models', color='#A23B72', alpha=0.8)
        
        # Add mean values in the middle of bars
        for i, (baseline_mean, shift_mean) in enumerate(zip(baseline_means, shift_means)):
            # Baseline bar mean value
            if abs(baseline_mean) > 0.001:
                text_y = baseline_mean / 2
                bar_height = abs(baseline_mean)
                
                if bar_height > 0.1:
                    text_color = 'white'
                    bbox_color = 'black'
                else:
                    text_color = 'black'
                    bbox_color = 'white'
                
                if abs(baseline_mean) >= 10:
                    text_val = f'{baseline_mean:.1f}'
                elif abs(baseline_mean) >= 1:
                    text_val = f'{baseline_mean:.2f}'
                else:
                    text_val = f'{baseline_mean:.3f}'
                
                ax.text(i - width/2, text_y, text_val, 
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color=text_color,
                       bbox=dict(boxstyle='round,pad=0.15', facecolor=bbox_color, alpha=0.7, edgecolor='gray'))
            
            # Shift bar mean value
            if abs(shift_mean) > 0.001:
                text_y = shift_mean / 2
                bar_height = abs(shift_mean)
                
                if bar_height > 0.1:
                    text_color = 'white'
                    bbox_color = 'black'
                else:
                    text_color = 'black'
                    bbox_color = 'white'
                
                if abs(shift_mean) >= 10:
                    text_val = f'{shift_mean:.1f}'
                elif abs(shift_mean) >= 1:
                    text_val = f'{shift_mean:.2f}'
                else:
                    text_val = f'{shift_mean:.3f}'
                
                ax.text(i + width/2, text_y, text_val, 
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       color=text_color,
                       bbox=dict(boxstyle='round,pad=0.15', facecolor=bbox_color, alpha=0.7, edgecolor='gray'))
        
        # Customize plot
        ax.set_xlabel('Test Scenario', fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45, ha='right')
        
        if ylim:
            ax.set_ylim(ylim)
            
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add sample size annotations
        for i, scenario in enumerate(scenarios):
            if scenario in baseline_data.index:
                n_baseline = baseline_data.loc[scenario, 'n']
                ax.text(i - width/2, baseline_means[i] + baseline_cis[i] + 0.01 * max(baseline_means), 
                       f'n={n_baseline}', ha='center', va='bottom', fontsize=8, color='#2E86AB')
            
            if scenario in shift_data.index:
                n_shift = shift_data.loc[scenario, 'n']
                ax.text(i + width/2, shift_means[i] + shift_cis[i] + 0.01 * max(shift_means), 
                       f'n={n_shift}', ha='center', va='bottom', fontsize=8, color='#A23B72')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved: {save_path}")
            
        return fig
    
    def create_all_models_per_scenario_plot(self, metric: str, title: str, ylabel: str,
                                          figsize: Tuple[int, int] = (16, 8),
                                          ylim: Optional[Tuple[float, float]] = None,
                                          save_path: Optional[str] = None) -> Optional[matplotlib.figure.Figure]:
        """Create plot showing all models for each scenario with separators."""
        
        # Get all unique models (baseline scenarios) from the data
        models = sorted(self.df['baseline_scenario'].unique())
        scenarios = sorted(self.df['test_scenario'].unique())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate positions with separators between scenarios
        models_per_scenario = len(models)
        scenario_spacing = models_per_scenario + 1  # +1 for separator
        total_positions = len(scenarios) * scenario_spacing - 1  # -1 because no separator after last scenario
        
        # Colors for different models
        import matplotlib.cm as cm
        colors = cm.get_cmap('Set3')(np.linspace(0, 1, len(models)))
        
        x_positions = []
        x_labels = []
        scenario_centers = []
        
        current_pos = 0
        for scenario_idx, scenario in enumerate(scenarios):
            scenario_start = current_pos
            
            # For each model, calculate stats when tested on this scenario
            for model_idx, model in enumerate(models):
                model_data = self.df[(self.df['baseline_scenario'] == model) & 
                                   (self.df['test_scenario'] == scenario)]
                
                if len(model_data) > 0:
                    values = model_data[metric].dropna()
                    if len(values) > 0:
                        mean_val = values.mean()
                        std_val = values.std()
                        n = len(values)
                        
                        # 90% confidence interval
                        ci_90 = 1.645 * std_val / np.sqrt(n) if n > 1 else 0
                        
                        # Check if this model is baseline for this scenario
                        is_baseline = (model == scenario)
                        
                        # Plot bar
                        ax.bar(current_pos, mean_val, width=0.8, 
                              yerr=ci_90, capsize=4, 
                              color=colors[model_idx], alpha=0.8)
                        
                        # Add mean value in the middle of the bar (for both positive and negative values)
                        if abs(mean_val) > 0.001:  # Show for any non-zero value
                            # Calculate text position
                            text_y = mean_val / 2 if mean_val > 0 else mean_val / 2
                            
                            # Smart color selection based on bar height and background
                            bar_height = abs(mean_val)
                            if bar_height > 0.1:  # For taller bars, use white text
                                text_color = 'white'
                                bbox_color = 'black'
                            else:  # For shorter bars, use black text with light background
                                text_color = 'black'
                                bbox_color = 'white'
                            
                            # Format the text based on the magnitude of the value
                            if abs(mean_val) >= 10:
                                text_val = f'{mean_val:.1f}'
                            elif abs(mean_val) >= 1:
                                text_val = f'{mean_val:.2f}'
                            else:
                                text_val = f'{mean_val:.3f}'
                            
                            ax.text(current_pos, text_y, text_val, 
                                   ha='center', va='center', fontsize=8, fontweight='bold',
                                   color=text_color,
                                   bbox=dict(boxstyle='round,pad=0.15', facecolor=bbox_color, alpha=0.7, edgecolor='gray'))
                        
                        # Add sample size annotation
                        if mean_val + ci_90 > 0:
                            ax.text(current_pos, mean_val + ci_90 + 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                                   f'n={n}', ha='center', va='bottom', fontsize=8, rotation=90)
                        elif mean_val - ci_90 < 0:  # For negative values, put sample size below
                            ax.text(current_pos, mean_val - ci_90 - 0.01 * abs(ax.get_ylim()[0] - ax.get_ylim()[1]), 
                                   f'n={n}', ha='center', va='top', fontsize=8, rotation=90)
                
                x_positions.append(current_pos)
                # Bold the label if it's the baseline model for this scenario
                model_label = model.replace('_', ' ').title()
                if model == scenario:
                    model_label = f'**{model_label}**'  # This will be handled in the plotting
                x_labels.append(model_label)
                current_pos += 1
            
            # Calculate scenario center for separator
            scenario_center = (scenario_start + current_pos - 1) / 2
            scenario_centers.append(scenario_center)
            
            # Add separator (except after last scenario)
            if scenario_idx < len(scenarios) - 1:
                ax.axvline(x=current_pos - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
                current_pos += 1
        
        # Customize plot
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14)
        
        # Set x-axis with bold baseline labels
        ax.set_xticks(x_positions)
        
        # Handle bold labels for baseline models
        clean_labels = []
        for i, label in enumerate(x_labels):
            if '**' in label:
                clean_labels.append(label.replace('**', ''))
            else:
                clean_labels.append(label)
        
        ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=10)
        
        # Make baseline model labels bold
        for i, label in enumerate(x_labels):
            if '**' in label:
                ax.get_xticklabels()[i].set_fontweight('bold')
        
        # Add scenario labels at the top
        for i, (scenario, center) in enumerate(zip(scenarios, scenario_centers)):
            ax.text(center, ax.get_ylim()[1] * 1.05, scenario.replace('_', ' ').title(), 
                   ha='center', va='bottom', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        if ylim:
            ax.set_ylim(ylim)
        
        # Remove legend as requested
        ax.grid(True, alpha=0.3, axis='y')
        
        # Extend y-axis to accommodate scenario labels
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved: {save_path}")
            
        return fig
    
    def create_safety_metrics_dashboard(self, save_dir: Optional[str] = None):
        """Create comprehensive safety metrics dashboard."""
        
        metrics = [
            ('fp_rate', 'False Positive Rate by Scenario\nMean ± 90% Confidence Interval', 'False Positive Rate', (0, None)),
            ('num_los_events', 'Loss of Separation Events by Scenario\nMean ± 90% Confidence Interval', 'Number of LOS Events', (0, None)),
            ('min_separation_nm', 'Minimum Separation by Scenario\nMean ± 90% Confidence Interval', 'Minimum Separation (NM)', (0, None)),
            ('num_interventions', 'Interventions by Scenario\nMean ± 90% Confidence Interval', 'Number of Interventions', (0, None))
        ]
        
        # Use seaborn color palette for better colors
        colors = sns.color_palette("Set2", 2)  # Get 2 colors from Set2 palette
        baseline_color = colors[0]  # Teal-like color
        shift_color = colors[1]     # Orange-like color
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # Larger figure
        axes = axes.flatten()
        
        for i, (metric, title, ylabel, ylim) in enumerate(metrics):
            stats_df = self.calculate_scenario_stats(metric)
            
            if not stats_df.empty:
                ax = axes[i]
                
                scenarios = sorted(stats_df['scenario'].unique())
                x_pos = np.arange(len(scenarios))
                width = 0.35
                
                baseline_data = stats_df[stats_df['model_type'] == 'Baseline'].set_index('scenario')
                shift_data = stats_df[stats_df['model_type'] == 'Shift'].set_index('scenario')
                
                baseline_means = [baseline_data.loc[s, 'mean'] if s in baseline_data.index else 0 for s in scenarios]
                baseline_cis = [baseline_data.loc[s, 'ci_90'] if s in baseline_data.index else 0 for s in scenarios]
                
                shift_means = [shift_data.loc[s, 'mean'] if s in shift_data.index else 0 for s in scenarios]
                shift_cis = [shift_data.loc[s, 'ci_90'] if s in shift_data.index else 0 for s in scenarios]
                
                bars1 = ax.bar(x_pos - width/2, baseline_means, width, 
                              yerr=baseline_cis, capsize=6,
                              label='Baseline' if i == 0 else '', color=baseline_color, alpha=0.8)
                
                bars2 = ax.bar(x_pos + width/2, shift_means, width,
                              yerr=shift_cis, capsize=6, 
                              label='Shift' if i == 0 else '', color=shift_color, alpha=0.8)
                
                # Add mean values in the middle of bars
                for j, (baseline_mean, shift_mean) in enumerate(zip(baseline_means, shift_means)):
                    # Baseline bar mean value
                    if abs(baseline_mean) > 0.001:
                        text_y = baseline_mean / 2
                        bar_height = abs(baseline_mean)
                        
                        if bar_height > 0.1:
                            text_color = 'white'
                            bbox_color = 'black'
                        else:
                            text_color = 'black'
                            bbox_color = 'white'
                        
                        if abs(baseline_mean) >= 10:
                            text_val = f'{baseline_mean:.1f}'
                        elif abs(baseline_mean) >= 1:
                            text_val = f'{baseline_mean:.2f}'
                        else:
                            text_val = f'{baseline_mean:.3f}'
                        
                        ax.text(j - width/2, text_y, text_val, 
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               color=text_color,
                               bbox=dict(boxstyle='round,pad=0.15', facecolor=bbox_color, alpha=0.8, edgecolor='gray'))
                    
                    # Shift bar mean value
                    if abs(shift_mean) > 0.001:
                        text_y = shift_mean / 2
                        bar_height = abs(shift_mean)
                        
                        if bar_height > 0.1:
                            text_color = 'white'
                            bbox_color = 'black'
                        else:
                            text_color = 'black'
                            bbox_color = 'white'
                        
                        if abs(shift_mean) >= 10:
                            text_val = f'{shift_mean:.1f}'
                        elif abs(shift_mean) >= 1:
                            text_val = f'{shift_mean:.2f}'
                        else:
                            text_val = f'{shift_mean:.3f}'
                        
                        ax.text(j + width/2, text_y, text_val, 
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               color=text_color,
                               bbox=dict(boxstyle='round,pad=0.15', facecolor=bbox_color, alpha=0.8, edgecolor='gray'))
                
                ax.set_xlabel('Test Scenario', fontweight='bold', fontsize=14)
                ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)
                ax.set_title(title, fontweight='bold', fontsize=15)
                ax.set_xticks(x_pos)
                ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45, ha='right', fontsize=12)
                ax.tick_params(axis='y', labelsize=12)
                
                if ylim[1] is not None:
                    ax.set_ylim(ylim)
                elif ylim[0] is not None:
                    ax.set_ylim(bottom=ylim[0])
                    
                ax.grid(True, alpha=0.3, axis='y')
                
                if i == 0:
                    ax.legend(loc='upper right', fontsize=12)
        
        plt.suptitle('Safety Metrics: Scenario-Specific Performance Comparison', 
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'safety_metrics_dashboard.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved safety dashboard: {save_path}")
            
        return fig
    
    def create_efficiency_dashboard(self, save_dir: Optional[str] = None):
        """Create operational efficiency metrics dashboard."""
        
        metrics = [
            ('accuracy', 'Detection Accuracy by Scenario\nMean ± 90% Confidence Interval', 'Accuracy', (0, 1)),
            ('flight_time_min', 'Flight Time by Scenario\nMean ± 90% Confidence Interval', 'Flight Time (minutes)', (0, None)),
            ('waypoint_reached_ratio', 'Waypoint Completion by Scenario\nMean ± 90% Confidence Interval', 'Completion Ratio', (0, 1)),
            ('path_efficiency', 'Path Efficiency by Scenario\nMean ± 90% Confidence Interval', 'Path Efficiency', (0, None))
        ]
        
        # Use seaborn color palette for better colors
        colors = sns.color_palette("Set1", 2)  # Get 2 colors from Set1 palette
        baseline_color = colors[0]  # Blue-like color
        shift_color = colors[1]     # Red-like color
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # Larger figure
        axes = axes.flatten()
        
        for i, (metric, title, ylabel, ylim) in enumerate(metrics):
            stats_df = self.calculate_scenario_stats(metric)
            
            if not stats_df.empty:
                ax = axes[i]
                
                scenarios = sorted(stats_df['scenario'].unique())
                x_pos = np.arange(len(scenarios))
                width = 0.35
                
                baseline_data = stats_df[stats_df['model_type'] == 'Baseline'].set_index('scenario')
                shift_data = stats_df[stats_df['model_type'] == 'Shift'].set_index('scenario')
                
                baseline_means = [baseline_data.loc[s, 'mean'] if s in baseline_data.index else 0 for s in scenarios]
                baseline_cis = [baseline_data.loc[s, 'ci_90'] if s in baseline_data.index else 0 for s in scenarios]
                
                shift_means = [shift_data.loc[s, 'mean'] if s in shift_data.index else 0 for s in scenarios]
                shift_cis = [shift_data.loc[s, 'ci_90'] if s in shift_data.index else 0 for s in scenarios]
                
                bars1 = ax.bar(x_pos - width/2, baseline_means, width, 
                              yerr=baseline_cis, capsize=6,
                              label='Baseline' if i == 0 else '', color=baseline_color, alpha=0.8)
                
                bars2 = ax.bar(x_pos + width/2, shift_means, width,
                              yerr=shift_cis, capsize=6, 
                              label='Shift' if i == 0 else '', color=shift_color, alpha=0.8)
                
                # Add mean values in the middle of bars
                for j, (baseline_mean, shift_mean) in enumerate(zip(baseline_means, shift_means)):
                    # Baseline bar mean value
                    if abs(baseline_mean) > 0.001:
                        text_y = baseline_mean / 2
                        bar_height = abs(baseline_mean)
                        
                        if bar_height > 0.1:
                            text_color = 'white'
                            bbox_color = 'black'
                        else:
                            text_color = 'black'
                            bbox_color = 'white'
                        
                        if abs(baseline_mean) >= 10:
                            text_val = f'{baseline_mean:.1f}'
                        elif abs(baseline_mean) >= 1:
                            text_val = f'{baseline_mean:.2f}'
                        else:
                            text_val = f'{baseline_mean:.3f}'
                        
                        ax.text(j - width/2, text_y, text_val, 
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               color=text_color,
                               bbox=dict(boxstyle='round,pad=0.15', facecolor=bbox_color, alpha=0.8, edgecolor='gray'))
                    
                    # Shift bar mean value
                    if abs(shift_mean) > 0.001:
                        text_y = shift_mean / 2
                        bar_height = abs(shift_mean)
                        
                        if bar_height > 0.1:
                            text_color = 'white'
                            bbox_color = 'black'
                        else:
                            text_color = 'black'
                            bbox_color = 'white'
                        
                        if abs(shift_mean) >= 10:
                            text_val = f'{shift_mean:.1f}'
                        elif abs(shift_mean) >= 1:
                            text_val = f'{shift_mean:.2f}'
                        else:
                            text_val = f'{shift_mean:.3f}'
                        
                        ax.text(j + width/2, text_y, text_val, 
                               ha='center', va='center', fontsize=10, fontweight='bold',
                               color=text_color,
                               bbox=dict(boxstyle='round,pad=0.15', facecolor=bbox_color, alpha=0.8, edgecolor='gray'))
                
                ax.set_xlabel('Test Scenario', fontweight='bold', fontsize=14)
                ax.set_ylabel(ylabel, fontweight='bold', fontsize=14)
                ax.set_title(title, fontweight='bold', fontsize=15)
                ax.set_xticks(x_pos)
                ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45, ha='right', fontsize=12)
                ax.tick_params(axis='y', labelsize=12)
                
                if ylim[1] is not None:
                    ax.set_ylim(ylim)
                elif ylim[0] is not None:
                    ax.set_ylim(bottom=ylim[0])
                    
                ax.grid(True, alpha=0.3, axis='y')
                
                if i == 0:
                    ax.legend(loc='upper right', fontsize=12)
        
        plt.suptitle('Efficiency Metrics: Scenario-Specific Performance Comparison', 
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / 'efficiency_metrics_dashboard.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved efficiency dashboard: {save_path}")
            
        return fig
    
    def create_all_plots(self, output_dir: str, use_models_per_scenario: bool = False):
        """Generate all scenario-specific performance plots."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎨 Generating Scenario-Specific Performance Plots")
        print("=" * 60)
        
        # Individual metric plots with dynamic ranges based on actual data
        individual_metrics = [
            ('fp_rate', 'False Positive Rate by Scenario\nMean ± 90% Confidence Interval', 'False Positive Rate', self.metric_ranges.get('fp_rate', (0, 0.1))),
            ('fn_rate', 'False Negative Rate by Scenario\nMean ± 90% Confidence Interval', 'False Negative Rate', self.metric_ranges.get('fn_rate', (0, 0.1))),
            ('accuracy', 'Detection Accuracy by Scenario\nMean ± 90% Confidence Interval', 'Accuracy', self.metric_ranges.get('accuracy', (0, 1))),
            ('precision', 'Precision by Scenario\nMean ± 90% Confidence Interval', 'Precision', self.metric_ranges.get('precision', (0, 1))),
            ('recall', 'Recall by Scenario\nMean ± 90% Confidence Interval', 'Recall', self.metric_ranges.get('recall', (0, 1))),
            ('f1_score', 'F1 Score by Scenario\nMean ± 90% Confidence Interval', 'F1 Score', self.metric_ranges.get('f1_score', (0, 1))),
            ('num_los_events', 'Loss of Separation Events by Scenario\nMean ± 90% Confidence Interval', 'Number of LOS Events', self.metric_ranges.get('num_los_events', (0, None))),
            ('flight_time_min', 'Flight Time by Scenario\nMean ± 90% Confidence Interval', 'Flight Time (minutes)', self.metric_ranges.get('flight_time_min', (0, None))),
            ('waypoint_reached_ratio', 'Completion Ratio by Scenario\nMean ± 90% Confidence Interval', 'Waypoint Completion Ratio', self.metric_ranges.get('waypoint_reached_ratio', (0, 1))),
            ('num_interventions', 'Interventions by Scenario\nMean ± 90% Confidence Interval', 'Number of Interventions', self.metric_ranges.get('num_interventions', (0, None))),
            ('min_separation_nm', 'Minimum Separation by Scenario\nMean ± 90% Confidence Interval', 'Minimum Separation (NM)', self.metric_ranges.get('min_separation_nm', (0, None))),
            ('reward_total', 'Total Reward by Scenario\nMean ± 90% Confidence Interval', 'Total Reward', self.metric_ranges.get('reward_total', (None, None))),
            ('ghost_conflict', 'Ghost Conflicts by Scenario\nMean ± 90% Confidence Interval', 'Ghost Conflict Rate', self.metric_ranges.get('ghost_conflict', (0, None))),
            ('missed_conflict', 'Missed Conflicts by Scenario\nMean ± 90% Confidence Interval', 'Missed Conflict Rate', self.metric_ranges.get('missed_conflict', (0, None))),
            ('alert_duty_cycle', 'Alert Duty Cycle by Scenario\nMean ± 90% Confidence Interval', 'Alert Duty Cycle', self.metric_ranges.get('alert_duty_cycle', (0, None))),
            ('alerts_per_min', 'Alerts per Minute by Scenario\nMean ± 90% Confidence Interval', 'Alerts per Minute', self.metric_ranges.get('alerts_per_min', (0, None)))
        ]
        
        if use_models_per_scenario:
            # Use new visualization showing all models per scenario
            for metric, title, ylabel, ylim in individual_metrics:
                save_path = output_path / f'all_models_{metric}_by_scenario.png'
                title_new = title.replace('by Scenario', 'All Models per Scenario')
                fig = self.create_all_models_per_scenario_plot(metric, title_new, ylabel, ylim=ylim, save_path=str(save_path))
                if fig:
                    plt.close(fig)
        else:
            # Use original visualization
            for metric, title, ylabel, ylim in individual_metrics:
                save_path = output_path / f'scenario_{metric}_comparison.png'
                fig = self.create_comparison_plot(metric, title, ylabel, ylim=ylim, save_path=str(save_path))
                if fig:
                    plt.close(fig)
        
        # Dashboard plots (always create these)
        print(f"\n📊 Creating comprehensive dashboards...")
        
        safety_fig = self.create_safety_metrics_dashboard(str(output_path))
        if safety_fig:
            plt.close(safety_fig)
            
        efficiency_fig = self.create_efficiency_dashboard(str(output_path))
        if efficiency_fig:
            plt.close(efficiency_fig)
        
        # Summary statistics
        self.create_summary_table(str(output_path))
        
        print(f"\n✅ All plots generated in: {output_path}")
        
    def create_summary_table(self, output_dir: str):
        """Create summary statistics table."""
        
        key_metrics = ['fp_rate', 'fn_rate', 'accuracy', 'precision', 'recall', 'f1_score',
                      'num_los_events', 'min_separation_nm', 'flight_time_min', 'waypoint_reached_ratio', 
                      'num_interventions', 'reward_total', 'ghost_conflict', 'missed_conflict', 
                      'alert_duty_cycle', 'alerts_per_min']
        
        summary_data = []
        
        for scenario in self.scenarios:
            scenario_data = self.df[self.df['test_scenario'] == scenario]
            
            baseline_data = scenario_data[scenario_data['model_scenario_type'] == 'Baseline']
            shift_data = scenario_data[scenario_data['model_scenario_type'] == 'Shift']
            
            row = {'scenario': scenario}
            
            for metric in key_metrics:
                if len(baseline_data) > 0:
                    baseline_mean = baseline_data[metric].mean()
                    row[f'{metric}_baseline_mean'] = baseline_mean
                    row[f'{metric}_baseline_std'] = baseline_data[metric].std()
                    row[f'{metric}_baseline_n'] = len(baseline_data)
                else:
                    row[f'{metric}_baseline_mean'] = np.nan
                    row[f'{metric}_baseline_std'] = np.nan
                    row[f'{metric}_baseline_n'] = 0
                
                if len(shift_data) > 0:
                    shift_mean = shift_data[metric].mean()
                    row[f'{metric}_shift_mean'] = shift_mean
                    row[f'{metric}_shift_std'] = shift_data[metric].std()
                    row[f'{metric}_shift_n'] = len(shift_data)
                    
                    # Calculate performance delta (shift - baseline)
                    if not np.isnan(baseline_mean) and not np.isnan(shift_mean):
                        row[f'{metric}_delta'] = shift_mean - baseline_mean
                        row[f'{metric}_delta_pct'] = ((shift_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else np.nan
                else:
                    row[f'{metric}_shift_mean'] = np.nan
                    row[f'{metric}_shift_std'] = np.nan
                    row[f'{metric}_shift_n'] = 0
                    row[f'{metric}_delta'] = np.nan
                    row[f'{metric}_delta_pct'] = np.nan
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        save_path = Path(output_dir) / 'scenario_performance_summary.csv'
        summary_df.to_csv(save_path, index=False)
        
        print(f"✅ Saved summary table: {save_path}")
        
        return summary_df


def main():
    parser = argparse.ArgumentParser(description='Generate scenario-specific performance comparison plots')
    parser.add_argument('--data', required=True, help='Path to baseline_vs_shift_detailed_summary.csv')
    parser.add_argument('--output', default='scenario_plots', help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving')
    parser.add_argument('--models-per-scenario', action='store_true', 
                       help='Show all models per scenario with separators (new visualization)')
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}")
        return 1
    
    # Create analyzer and generate plots
    analyzer = ScenarioPerformanceAnalyzer(args.data)
    
    if args.show:
        # Generate individual plots for viewing
        metrics_to_show = ['fp_rate', 'fn_rate', 'accuracy', 'precision', 'recall', 'ghost_conflict', 'num_los_events', 'reward_total']
        
        for metric in metrics_to_show:
            if args.models_per_scenario:
                if metric == 'fp_rate':
                    fig = analyzer.create_all_models_per_scenario_plot(metric, 
                        'False Positive Rate - All Models per Scenario\nMean ± 90% Confidence Interval', 
                        'False Positive Rate', ylim=(0, 0.1))
                elif metric == 'fn_rate':
                    fig = analyzer.create_all_models_per_scenario_plot(metric, 
                        'False Negative Rate - All Models per Scenario\nMean ± 90% Confidence Interval', 
                        'False Negative Rate', ylim=analyzer.metric_ranges.get('fn_rate', (0, 0.1)))
                elif metric == 'accuracy':
                    fig = analyzer.create_all_models_per_scenario_plot(metric,
                        'Detection Accuracy - All Models per Scenario\nMean ± 90% Confidence Interval',
                        'Accuracy', ylim=(0, 1))
                elif metric == 'precision':
                    fig = analyzer.create_all_models_per_scenario_plot(metric,
                        'Precision - All Models per Scenario\nMean ± 90% Confidence Interval',
                        'Precision', ylim=(0, 1))
                elif metric == 'recall':
                    fig = analyzer.create_all_models_per_scenario_plot(metric,
                        'Recall - All Models per Scenario\nMean ± 90% Confidence Interval',
                        'Recall', ylim=(0, 1))
                elif metric == 'ghost_conflict':
                    fig = analyzer.create_all_models_per_scenario_plot(metric,
                        'Ghost Conflicts - All Models per Scenario\nMean ± 90% Confidence Interval',
                        'Ghost Conflict Rate', ylim=analyzer.metric_ranges.get('ghost_conflict', (0, None)))
                elif metric == 'num_los_events':
                    fig = analyzer.create_all_models_per_scenario_plot(metric,
                        'Loss of Separation Events - All Models per Scenario\nMean ± 90% Confidence Interval',
                        'Number of LOS Events')
                elif metric == 'reward_total':
                    fig = analyzer.create_all_models_per_scenario_plot(metric,
                        'Total Reward - All Models per Scenario\nMean ± 90% Confidence Interval',
                        'Total Reward')
            else:
                if metric == 'fp_rate':
                    fig = analyzer.create_comparison_plot(metric, 
                        'False Positive Rate by Scenario\nMean ± 90% Confidence Interval', 
                        'False Positive Rate', ylim=(0, 0.1))
                elif metric == 'fn_rate':
                    fig = analyzer.create_comparison_plot(metric, 
                        'False Negative Rate by Scenario\nMean ± 90% Confidence Interval', 
                        'False Negative Rate', ylim=(0, 0.1))
                elif metric == 'accuracy':
                    fig = analyzer.create_comparison_plot(metric,
                        'Detection Accuracy by Scenario\nMean ± 90% Confidence Interval',
                        'Accuracy', ylim=(0, 1))
                elif metric == 'num_los_events':
                    fig = analyzer.create_comparison_plot(metric,
                        'Loss of Separation Events by Scenario\nMean ± 90% Confidence Interval',
                        'Number of LOS Events')
                elif metric == 'reward_total':
                    fig = analyzer.create_comparison_plot(metric,
                        'Total Reward by Scenario\nMean ± 90% Confidence Interval',
                        'Total Reward')
            
            if fig:
                plt.show()
                plt.close(fig)
    else:
        analyzer.create_all_plots(args.output, use_models_per_scenario=args.models_per_scenario)
    
    return 0


if __name__ == "__main__":
    exit(main())