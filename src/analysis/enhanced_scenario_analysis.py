#!/usr/bin/env python3
"""
Enhanced Scenario-Specific Performance Analysis with Safety & Detection Metrics

Comprehensive analysis including:
- Safety & Resolution: LOS events, resolution success, min separation survival curves
- Detection Quality: Precision/Recall/F1, ghost conflicts, alert burden, lead times
- Efficiency: Path efficiency, flight time, interventions per hour
- Advanced Visualizations: Delta heatmaps, survival curves, alert burden scatter plots

Usage:
    python enhanced_scenario_analysis.py --data data.csv --output enhanced_plots/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
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

class EnhancedScenarioAnalyzer:
    def __init__(self, data_path: str):
        """Initialize with comprehensive metric calculations."""
        self.df = pd.read_csv(data_path)
        self.prepare_enhanced_data()
        
    def prepare_enhanced_data(self):
        """Calculate all enhanced metrics."""
        
        # Basic classification metrics
        self.df['fp_rate'] = self.df['fp'] / (self.df['fp'] + self.df['tn'] + 1e-8)
        self.df['fn_rate'] = self.df['fn'] / (self.df['fn'] + self.df['tp'] + 1e-8)
        self.df['accuracy'] = (self.df['tp'] + self.df['tn']) / (self.df['tp'] + self.df['fp'] + self.df['fn'] + self.df['tn'] + 1e-8)
        
        # Detection quality metrics
        self.df['precision'] = self.df['tp'] / (self.df['tp'] + self.df['fp'] + 1e-8)
        self.df['recall'] = self.df['tp'] / (self.df['tp'] + self.df['fn'] + 1e-8)
        self.df['f1_score'] = 2 * (self.df['precision'] * self.df['recall']) / (self.df['precision'] + self.df['recall'] + 1e-8)
        
        # Safety & resolution metrics
        self.df['ghost_conflict_rate'] = self.df['fp'] / (self.df['fp'] + self.df['tn'] + 1e-8)
        self.df['missed_conflict_rate'] = self.df['fn'] / (self.df['fn'] + self.df['tp'] + 1e-8)
        
        # Time-based rates
        self.df['flight_time_hours'] = self.df['flight_time_s'] / 3600.0
        self.df['flight_time_min'] = self.df['flight_time_s'] / 60.0
        self.df['los_events_per_hour'] = self.df['num_los_events'] / (self.df['flight_time_hours'] + 1e-8)
        self.df['interventions_per_hour'] = self.df['num_interventions'] / (self.df['flight_time_hours'] + 1e-8)
        
        # Alert burden metrics
        self.df['alerts_per_min'] = (self.df['num_interventions'] * 60.0) / (self.df['flight_time_s'] + 1e-8)
        self.df['unwanted_interventions'] = self.df['num_interventions_false']
        self.df['unwanted_intervention_rate'] = self.df['unwanted_interventions'] / (self.df['flight_time_hours'] + 1e-8)
        
        # Resolution metrics (when resolution data is available)
        if 'num_interventions_matched' in self.df.columns:
            self.df['resolution_success_rate'] = self.df['num_interventions_matched'] / (self.df['num_interventions_matched'] + self.df['num_interventions_false'] + 1e-8)
        else:
            self.df['resolution_success_rate'] = np.nan
            
        # Efficiency metrics
        if 'total_extra_path_nm' in self.df.columns and 'total_path_length_nm' in self.df.columns:
            self.df['extra_path_ratio'] = self.df['total_extra_path_nm'] / (self.df['total_path_length_nm'] + 1e-8)
        else:
            self.df['extra_path_ratio'] = np.nan
            
        # Model training scenario extraction
        self.df['model_trained_on'] = self.df['model_alias'].str.replace('PPO_', '')
        
        # Get unique values
        self.test_scenarios = sorted(self.df['test_scenario'].unique())
        self.model_aliases = sorted(self.df['model_alias'].unique())
        
        print(f"📊 Enhanced Analyzer Ready:")
        print(f"   • Episodes: {len(self.df)}")
        print(f"   • Scenarios: {self.test_scenarios}")
        print(f"   • Models: {[alias.replace('PPO_', '') for alias in self.model_aliases]}")
        print(f"   • Metrics calculated: Safety, Detection Quality, Efficiency, Alert Burden")
        
    def calculate_model_stats_with_baseline_delta(self, scenario: str, metric: str) -> pd.DataFrame:
        """Calculate stats with delta vs baseline."""
        
        scenario_data = self.df[self.df['test_scenario'] == scenario]
        baseline_model = f"PPO_{scenario}"
        
        # Get baseline value
        baseline_data = scenario_data[scenario_data['model_alias'] == baseline_model]
        baseline_mean = baseline_data[metric].mean() if len(baseline_data) > 0 else np.nan
        
        stats = []
        for model_alias in self.model_aliases:
            model_data = scenario_data[scenario_data['model_alias'] == model_alias]
            
            if len(model_data) > 0:
                values = model_data[metric].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    n = len(values)
                    
                    # 90% confidence interval
                    ci_90 = 1.645 * std_val / np.sqrt(n) if n > 1 and std_val > 0 else 0
                    
                    # Calculate delta vs baseline
                    if not np.isnan(baseline_mean) and baseline_mean != 0:
                        delta_pct = ((mean_val - baseline_mean) / baseline_mean) * 100
                    else:
                        delta_pct = 0
                    
                    # Check if baseline
                    model_trained_on = model_alias.replace('PPO_', '')
                    is_baseline = (model_trained_on == scenario)
                    
                    stats.append({
                        'model_alias': model_alias,
                        'model_short': model_trained_on,
                        'mean': mean_val,
                        'ci_90': ci_90,
                        'ci_lo': mean_val - ci_90,
                        'ci_hi': mean_val + ci_90,
                        'n': n,
                        'is_baseline': is_baseline,
                        'delta_pct': delta_pct,
                        'baseline_mean': baseline_mean
                    })
        
        return pd.DataFrame(stats)
    
    def create_enhanced_profile_plot(self, scenario: str, metric: str, 
                                   ylabel: str, title_suffix: str = "",
                                   ylim: Optional[Tuple[float, float]] = None,
                                   figsize: Tuple[int, int] = (10, 6),
                                   save_path: Optional[str] = None):
        """Create enhanced profile plot with delta annotations."""
        
        stats_df = self.calculate_model_stats_with_baseline_delta(scenario, metric)
        
        if stats_df.empty:
            print(f"⚠️ No data for {scenario} - {metric}")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot baseline reference
        baseline_row = stats_df[stats_df['is_baseline'] == True]
        if len(baseline_row) > 0:
            baseline = baseline_row.iloc[0]
            
            # Horizontal reference line and CI band
            ax.axhline(y=baseline['mean'], color='red', linestyle='--', 
                      linewidth=2, alpha=0.8, zorder=1)
            
            x_span = [-0.5, len(stats_df) - 0.5]
            ax.fill_between(x_span, baseline['ci_lo'], baseline['ci_hi'], 
                          color='red', alpha=0.15, zorder=0,
                          label='Baseline 90% CI')
        
        # Plot all models with delta annotations
        x_positions = range(len(stats_df))
        
        for i, (_, row) in enumerate(stats_df.iterrows()):
            color = 'red' if row['is_baseline'] else 'blue'
            marker = 's' if row['is_baseline'] else 'o'
            markersize = 10 if row['is_baseline'] else 8
            
            # Error bar
            ax.errorbar(i, row['mean'], yerr=row['ci_90'], 
                       fmt=marker, color=color, capsize=5, capthick=2,
                       markersize=markersize, linewidth=2, zorder=3)
            
            # Add delta annotation (small label under dot)
            if not row['is_baseline'] and not np.isnan(row['delta_pct']):
                delta_text = f"{row['delta_pct']:+.0f}%"
                ax.text(i, row['mean'] - 0.15 * (ax.get_ylim()[1] - ax.get_ylim()[0]), 
                       delta_text, ha='center', va='top', 
                       fontsize=8, color=color, weight='bold')
        
        # Customize plot
        ax.set_xlabel('Model Alias', fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        
        title = f"Test scenario: {scenario.replace('_', ' ')}"
        if title_suffix:
            title += f"\\n{title_suffix}"
        ax.set_title(title, fontsize=12, pad=20)
        
        # X-axis labels
        model_labels = [row['model_short'] for _, row in stats_df.iterrows()]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_labels, rotation=0, ha='center')
        
        if ylim:
            ax.set_ylim(ylim)
        
        ax.grid(True, alpha=0.3)
        
        # Legend
        if len(baseline_row) > 0:
            legend_elements = [
                Line2D([0], [0], color='red', linestyle='--', linewidth=2, 
                          label='Baseline Model'),
                mpatches.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.15, 
                            label='Baseline 90% CI'),
                Line2D([0], [0], marker='o', color='blue', linestyle='None',
                          markersize=8, label='Other Models'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', 
                     frameon=True, fancybox=True, shadow=True, fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ {Path(save_path).name}")
        
        return fig
    
    def create_survival_curve_plot(self, scenario: str, save_path: Optional[str] = None):
        """Create survival curve for minimum separation with 5 NM safety line."""
        
        scenario_data = self.df[self.df['test_scenario'] == scenario]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Safety threshold
        safety_threshold = 5.0  # 5 NM
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, model_alias in enumerate(self.model_aliases):
            model_data = scenario_data[scenario_data['model_alias'] == model_alias]
            
            if len(model_data) > 0:
                min_seps = model_data['min_separation_nm'].dropna()
                
                if len(min_seps) > 0:
                    # Create survival curve (1 - CDF)
                    sorted_seps = np.sort(min_seps)
                    survival_probs = 1 - np.arange(1, len(sorted_seps) + 1) / len(sorted_seps)
                    
                    model_short = model_alias.replace('PPO_', '')
                    is_baseline = (model_short == scenario)
                    
                    linestyle = '-' if is_baseline else '--'
                    linewidth = 3 if is_baseline else 2
                    color = colors[i % len(colors)]
                    
                    ax.plot(sorted_seps, survival_probs, 
                           label=f'{model_short}{"*" if is_baseline else ""}',
                           color=color, linestyle=linestyle, linewidth=linewidth)
        
        # Add safety threshold line
        ax.axvline(x=safety_threshold, color='red', linestyle=':', 
                  linewidth=2, alpha=0.8, label='5 NM Safety Rule')
        
        ax.set_xlabel('Minimum Separation (NM)', fontweight='bold')
        ax.set_ylabel('Survival Probability', fontweight='bold')
        ax.set_title(f'Minimum Separation Survival Curves\\nTest Scenario: {scenario.replace("_", " ")}', 
                    fontweight='bold', pad=20)
        
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ {Path(save_path).name}")
        
        return fig
    
    def create_alert_burden_safety_scatter(self, scenario: str, save_path: Optional[str] = None):
        """Create scatter plot: alert burden vs safety with unwanted interventions as bubble size."""
        
        scenario_data = self.df[self.df['test_scenario'] == scenario]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_vals = []
        y_vals = []
        sizes = []
        colors = []
        labels = []
        
        for model_alias in self.model_aliases:
            model_data = scenario_data[scenario_data['model_alias'] == model_alias]
            
            if len(model_data) > 0:
                # X: Alert burden (alerts per minute)
                x_val = model_data['alerts_per_min'].mean()
                
                # Y: Safety (FN rate - lower is better)
                y_val = model_data['fn_rate'].mean()
                
                # Bubble size: Unwanted interventions per hour
                unwanted_rate = model_data['unwanted_intervention_rate'].mean()
                size_val = max(50, unwanted_rate * 100)  # Scale for visibility
                
                model_short = model_alias.replace('PPO_', '')
                is_baseline = (model_short == scenario)
                
                x_vals.append(x_val)
                y_vals.append(y_val)
                sizes.append(size_val)
                colors.append('red' if is_baseline else 'blue')
                labels.append(f'{model_short}{"*" if is_baseline else ""}')
        
        # Create scatter plot
        scatter = ax.scatter(x_vals, y_vals, s=sizes, c=colors, alpha=0.7, edgecolors='black')
        
        # Add labels
        for i, label in enumerate(labels):
            ax.annotate(label, (x_vals[i], y_vals[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Alert Burden (Alerts per Minute)', fontweight='bold')
        ax.set_ylabel('Safety Risk (FN Rate)', fontweight='bold')
        ax.set_title(f'Alert Burden vs Safety Trade-off\\nTest Scenario: {scenario.replace("_", " ")}\\n(Bubble size = Unwanted Interventions/Hour)', 
                    fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3)
        
        # Legend for colors
        legend_elements = [
            plt.scatter([], [], c='red', s=100, label='Baseline Model'),
            plt.scatter([], [], c='blue', s=100, label='Other Models')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ {Path(save_path).name}")
        
        return fig
    
    def create_precision_recall_f1_panel(self, scenario: str, save_path: Optional[str] = None):
        """Create small trio panel for precision, recall, F1."""
        
        scenario_data = self.df[self.df['test_scenario'] == scenario]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ['precision', 'recall', 'f1_score']
        titles = ['Precision', 'Recall', 'F1 Score']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            stats_df = self.calculate_model_stats_with_baseline_delta(scenario, metric)
            
            if not stats_df.empty:
                # Plot baseline reference
                baseline_row = stats_df[stats_df['is_baseline'] == True]
                if len(baseline_row) > 0:
                    baseline = baseline_row.iloc[0]
                    ax.axhline(y=baseline['mean'], color='red', linestyle='--', 
                              linewidth=2, alpha=0.8)
                    
                    x_span = [-0.5, len(stats_df) - 0.5]
                    ax.fill_between(x_span, baseline['ci_lo'], baseline['ci_hi'], 
                                  color='red', alpha=0.15)
                
                # Plot all models
                x_positions = range(len(stats_df))
                
                for j, (_, row) in enumerate(stats_df.iterrows()):
                    color = 'red' if row['is_baseline'] else 'blue'
                    marker = 's' if row['is_baseline'] else 'o'
                    markersize = 8 if row['is_baseline'] else 6
                    
                    ax.errorbar(j, row['mean'], yerr=row['ci_90'], 
                               fmt=marker, color=color, capsize=4, capthick=2,
                               markersize=markersize, linewidth=2)
                
                # Customize
                ax.set_title(title, fontweight='bold')
                ax.set_ylabel(title)
                model_labels = [row['model_short'] for _, row in stats_df.iterrows()]
                ax.set_xticks(x_positions)
                ax.set_xticklabels(model_labels, rotation=45, ha='right')
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Detection Quality Metrics\\nTest Scenario: {scenario.replace("_", " ")}', 
                    fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ {Path(save_path).name}")
        
        return fig
    
    def create_delta_heatmap(self, metrics: List[str], save_path: Optional[str] = None):
        """Create comprehensive delta heatmap for key metrics."""
        
        delta_data = []
        
        for scenario in self.test_scenarios:
            scenario_data = self.df[self.df['test_scenario'] == scenario]
            baseline_model = f"PPO_{scenario}"
            
            for metric in metrics:
                baseline_data = scenario_data[scenario_data['model_alias'] == baseline_model]
                
                if len(baseline_data) > 0:
                    baseline_mean = baseline_data[metric].mean()
                    
                    for model_alias in self.model_aliases:
                        model_data = scenario_data[scenario_data['model_alias'] == model_alias]
                        
                        if len(model_data) > 0:
                            model_mean = model_data[metric].mean()
                            
                            if baseline_mean != 0:
                                pct_change = ((model_mean - baseline_mean) / baseline_mean) * 100
                            else:
                                pct_change = 0 if model_mean == 0 else np.inf
                            
                            delta_data.append({
                                'scenario': scenario,
                                'model': model_alias.replace('PPO_', ''),
                                'metric': metric.replace('_', ' ').title(),
                                'pct_change': pct_change
                            })
        
        if not delta_data:
            return None
        
        # Create heatmap data
        delta_df = pd.DataFrame(delta_data)
        
        # Create subplots for each metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            metric_data = delta_df[delta_df['metric'] == metric.replace('_', ' ').title()]
            heatmap_data = metric_data.pivot(index='scenario', columns='model', values='pct_change')
            
            # Determine colormap direction
            cmap = 'RdYlGn_r' if metric in ['fp_rate', 'fn_rate', 'num_los_events', 'flight_time_min'] else 'RdYlGn'
            
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap=cmap, center=0,
                       cbar_kws={'label': '% Change vs Baseline'}, ax=axes[i])
            
            axes[i].set_title(f'{metric.replace("_", " ").title()} - % Change vs Baseline', 
                            fontweight='bold')
            axes[i].set_xlabel('Model (Trained On)', fontweight='bold')
            axes[i].set_ylabel('Test Scenario', fontweight='bold')
        
        plt.suptitle('Performance Degradation Heatmaps', fontweight='bold', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ {Path(save_path).name}")
        
        return fig
    
    def generate_comprehensive_analysis(self, output_dir: str):
        """Generate complete enhanced analysis."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\\n🎨 Generating Comprehensive Enhanced Analysis")
        print("=" * 60)
        
        # Enhanced metrics configuration
        safety_metrics = [
            ('num_los_events', 'Number of LOS Events'),
            ('los_events_per_hour', 'LOS Events per Hour'),
            ('total_los_duration', 'Total LOS Duration (s)'),
            ('min_separation_nm', 'Minimum Separation (NM)'),
            ('resolution_success_rate', 'Resolution Success Rate')
        ]
        
        detection_metrics = [
            ('fp_rate', 'False Positive Rate'),
            ('fn_rate', 'False Negative Rate'),
            ('precision', 'Precision'),
            ('recall', 'Recall'), 
            ('f1_score', 'F1 Score'),
            ('ghost_conflict_rate', 'Ghost Conflict Rate'),
            ('missed_conflict_rate', 'Missed Conflict Rate')
        ]
        
        burden_metrics = [
            ('alert_duty_cycle', 'Alert Duty Cycle'),
            ('alerts_per_min', 'Alerts per Minute'),
            ('total_alert_time_s', 'Total Alert Time (s)'),
            ('unwanted_interventions', 'Unwanted Interventions'),
            ('avg_lead_time_s', 'Average Lead Time (s)'),
            ('oscillation_rate', 'Action Oscillation Rate')
        ]
        
        efficiency_metrics = [
            ('path_efficiency', 'Path Efficiency'),
            ('extra_path_ratio', 'Extra Path Ratio'),
            ('flight_time_min', 'Flight Time (minutes)'),
            ('waypoint_reached_ratio', 'Waypoint Completion Ratio'),
            ('interventions_per_hour', 'Interventions per Hour'),
            ('reward_total', 'Total Reward')
        ]
        
        # Generate scenario-specific analysis
        for scenario in self.test_scenarios:
            print(f"\\n📊 {scenario.replace('_', ' ').title()}")
            
            scenario_dir = output_path / f"scenario_{scenario}"
            scenario_dir.mkdir(parents=True, exist_ok=True)
            
            # Safety metrics
            for metric, ylabel in safety_metrics:
                if metric in self.df.columns:
                    save_path = scenario_dir / f"{scenario}_{metric}_enhanced.png"
                    fig = self.create_enhanced_profile_plot(
                        scenario=scenario, metric=metric, ylabel=ylabel,
                        save_path=str(save_path)
                    )
                    if fig: plt.close(fig)
            
            # Detection quality metrics
            for metric, ylabel in detection_metrics:
                if metric in self.df.columns:
                    save_path = scenario_dir / f"{scenario}_{metric}_enhanced.png"
                    fig = self.create_enhanced_profile_plot(
                        scenario=scenario, metric=metric, ylabel=ylabel,
                        save_path=str(save_path)
                    )
                    if fig: plt.close(fig)
            
            # Alert burden metrics
            for metric, ylabel in burden_metrics:
                if metric in self.df.columns:
                    save_path = scenario_dir / f"{scenario}_{metric}_enhanced.png"
                    fig = self.create_enhanced_profile_plot(
                        scenario=scenario, metric=metric, ylabel=ylabel,
                        save_path=str(save_path)
                    )
                    if fig: plt.close(fig)
            
            # Efficiency metrics
            for metric, ylabel in efficiency_metrics:
                if metric in self.df.columns:
                    save_path = scenario_dir / f"{scenario}_{metric}_enhanced.png"
                    fig = self.create_enhanced_profile_plot(
                        scenario=scenario, metric=metric, ylabel=ylabel,
                        save_path=str(save_path)
                    )
                    if fig: plt.close(fig)
            
            # Advanced visualizations
            advanced_dir = scenario_dir / "advanced"
            advanced_dir.mkdir(parents=True, exist_ok=True)
            
            # Survival curve
            save_path = advanced_dir / f"{scenario}_min_separation_survival.png"
            fig = self.create_survival_curve_plot(scenario, str(save_path))
            if fig: plt.close(fig)
            
            # Alert burden vs safety scatter
            save_path = advanced_dir / f"{scenario}_alert_burden_safety_scatter.png"
            fig = self.create_alert_burden_safety_scatter(scenario, str(save_path))
            if fig: plt.close(fig)
            
            # Precision/Recall/F1 panel
            save_path = advanced_dir / f"{scenario}_precision_recall_f1_panel.png"
            fig = self.create_precision_recall_f1_panel(scenario, str(save_path))
            if fig: plt.close(fig)
        
        # Generate comprehensive delta heatmaps
        print(f"\\n📈 Creating Comprehensive Delta Heatmaps")
        heatmap_dir = output_path / "comprehensive_heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        
        # Key metric groups for heatmaps
        key_metrics = ['fp_rate', 'num_los_events', 'min_separation_nm', 'interventions_per_hour', 'f1_score', 'reward_total']
        available_metrics = [m for m in key_metrics if m in self.df.columns]
        
        save_path = heatmap_dir / "comprehensive_delta_heatmap.png"
        fig = self.create_delta_heatmap(available_metrics, str(save_path))
        if fig: plt.close(fig)
        
        # Export enhanced summary data
        self.export_enhanced_summary(str(output_path))
        
        print(f"\\n✅ Comprehensive enhanced analysis completed!")
        print(f"📁 Output: {output_path}")
    
    def export_enhanced_summary(self, output_dir: str):
        """Export comprehensive summary with all enhanced metrics."""
        
        summary_data = []
        
        # Define numeric columns to process
        numeric_columns = [
            'tp', 'fp', 'fn', 'tn', 'fp_rate', 'fn_rate', 'accuracy', 'precision', 'recall', 'f1_score',
            'ghost_conflict_rate', 'missed_conflict_rate', 'num_los_events', 'total_los_duration',
            'min_separation_nm', 'flight_time_s', 'flight_time_min', 'flight_time_hours',
            'los_events_per_hour', 'interventions_per_hour', 'alerts_per_min',
            'num_interventions', 'num_interventions_matched', 'num_interventions_false',
            'unwanted_interventions', 'unwanted_intervention_rate', 'resolution_success_rate',
            'total_path_length_nm', 'path_efficiency', 'extra_path_ratio', 'waypoint_reached_ratio',
            'alert_duty_cycle', 'total_alert_time_s', 'avg_lead_time_s', 'oscillation_rate',
            'reward_total', 'num_conflict_steps'
        ]
        
        for scenario in self.test_scenarios:
            scenario_data = self.df[self.df['test_scenario'] == scenario]
            
            for model_alias in self.model_aliases:
                model_data = scenario_data[scenario_data['model_alias'] == model_alias]
                
                if len(model_data) > 0:
                    model_short = model_alias.replace('PPO_', '')
                    is_baseline = (model_short == scenario)
                    
                    row = {
                        'test_scenario': scenario,
                        'model_alias': model_alias,
                        'model_short': model_short,
                        'is_baseline': is_baseline,
                        'n_episodes': len(model_data)
                    }
                    
                    # Process only numeric columns that exist in the data
                    for metric in numeric_columns:
                        if metric in model_data.columns:
                            values = model_data[metric].dropna()
                            
                            # Check if values are numeric
                            if len(values) > 0 and pd.api.types.is_numeric_dtype(values):
                                try:
                                    mean_val = float(values.mean())
                                    std_val = float(values.std())
                                    ci_90 = 1.645 * std_val / np.sqrt(len(values)) if len(values) > 1 and std_val > 0 else 0
                                    
                                    row[f'{metric}_mean'] = mean_val
                                    row[f'{metric}_std'] = std_val
                                    row[f'{metric}_ci_90'] = ci_90
                                except (TypeError, ValueError) as e:
                                    # Skip non-numeric data
                                    print(f"⚠️ Skipping non-numeric column {metric}: {e}")
                                    row[f'{metric}_mean'] = np.nan
                                    row[f'{metric}_std'] = np.nan
                                    row[f'{metric}_ci_90'] = np.nan
                            else:
                                row[f'{metric}_mean'] = np.nan
                                row[f'{metric}_std'] = np.nan
                                row[f'{metric}_ci_90'] = np.nan
                    
                    summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save comprehensive summary
        save_path = Path(output_dir) / 'enhanced_comprehensive_summary.csv'
        summary_df.to_csv(save_path, index=False)
        print(f"✅ Enhanced comprehensive summary: {save_path.name}")


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive enhanced scenario analysis')
    parser.add_argument('--data', required=True, help='Path to detailed summary CSV')
    parser.add_argument('--output', default='enhanced_scenario_analysis', help='Output directory')
    parser.add_argument('--scenario', help='Single scenario to analyze')
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}")
        return 1
    
    analyzer = EnhancedScenarioAnalyzer(args.data)
    
    if args.scenario:
        # Generate analysis for single scenario
        print(f"Analyzing scenario: {args.scenario}")
        # Add single scenario analysis here if needed
    else:
        # Generate complete analysis
        analyzer.generate_comprehensive_analysis(args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())