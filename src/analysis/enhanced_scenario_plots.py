#!/usr/bin/env python3
"""
Enhanced Scenario Performance Plots - Matching Your Reference Style

Creates publication-ready plots that exactly match the style of your provided reference graphs,
with baseline references, confidence intervals, and change-from-baseline metrics.

This script creates plots similar to:
- "VerticalCR - Model Fn Rate" with baseline mean reference
- "HorizontalCR - Ph5 Proxy Change from Baseline" with delta measurements

Usage:
    python enhanced_scenario_plots.py --data path/to/detailed_summary.csv --output enhanced_plots/
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

# Set publication-ready style
plt.style.use('default')  # Use clean default style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True
})

class EnhancedScenarioAnalyzer:
    def __init__(self, data_path: str):
        """Initialize analyzer with data."""
        self.df = pd.read_csv(data_path)
        self.scenarios = sorted(self.df['test_scenario'].unique())
        self.prepare_data()
        
    def prepare_data(self):
        """Calculate derived metrics and prepare data for analysis."""
        # Calculate derived metrics
        self.df['fp_rate'] = self.df['fp'] / (self.df['fp'] + self.df['tn'] + 1e-8)
        self.df['fn_rate'] = self.df['fn'] / (self.df['fn'] + self.df['tp'] + 1e-8)
        self.df['accuracy'] = (self.df['tp'] + self.df['tn']) / (self.df['tp'] + self.df['fp'] + self.df['fn'] + self.df['tn'] + 1e-8)
        
        # Convert flight time to minutes for better readability
        self.df['flight_time_min'] = self.df['flight_time_s'] / 60.0
        
        # Calculate model type for each scenario
        self.df['model_scenario_type'] = self.df.apply(
            lambda row: 'Baseline' if row['baseline_scenario'] == row['test_scenario'] else 'Shift', 
            axis=1
        )
        
        print(f"📊 Enhanced Analyzer Loaded:")
        print(f"   • Total episodes: {len(self.df)}")
        print(f"   • Scenarios: {len(self.scenarios)} ({', '.join(self.scenarios)})")
        print(f"   • Baseline episodes: {len(self.df[self.df['model_scenario_type'] == 'Baseline'])}")
        print(f"   • Shift episodes: {len(self.df[self.df['model_scenario_type'] == 'Shift'])}")
        
    def calculate_baseline_reference(self, metric: str) -> Dict[str, float]:
        """Calculate baseline reference values for each scenario."""
        baseline_refs = {}
        
        for scenario in self.scenarios:
            baseline_data = self.df[
                (self.df['test_scenario'] == scenario) & 
                (self.df['model_scenario_type'] == 'Baseline')
            ][metric].dropna()
            
            if len(baseline_data) > 0:
                baseline_refs[scenario] = baseline_data.mean()
            else:
                baseline_refs[scenario] = np.nan
                
        return baseline_refs
    
    def create_baseline_reference_plot(self, metric: str, title: str, ylabel: str, 
                                     figsize: Tuple[int, int] = (12, 8),
                                     ylim: Optional[Tuple[float, float]] = None,
                                     save_path: Optional[str] = None) -> matplotlib.figure.Figure:
        """Create plot similar to 'VerticalCR - Model Fn Rate' with baseline reference."""
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Colors matching your reference plots
        baseline_color = '#FF6B6B'  # Red for baseline
        shift_color = '#4ECDC4'     # Teal for shifts
        
        # Prepare data
        plot_data = []
        scenarios = sorted(self.scenarios)
        
        # Calculate baseline reference
        baseline_refs = self.calculate_baseline_reference(metric)
        overall_baseline_mean = np.nanmean(list(baseline_refs.values()))
        
        # Prepare scenario-model combinations for plotting
        x_positions = []
        x_labels = []
        x_pos = 0
        
        for scenario in scenarios:
            scenario_data = self.df[self.df['test_scenario'] == scenario]
            
            # Baseline data
            baseline_data = scenario_data[scenario_data['model_scenario_type'] == 'Baseline'][metric].dropna()
            # Shift data  
            shift_data = scenario_data[scenario_data['model_scenario_type'] == 'Shift'][metric].dropna()
            
            # Add baseline point
            if len(baseline_data) > 0:
                mean_val = baseline_data.mean()
                std_val = baseline_data.std()
                n = len(baseline_data)
                ci_90 = 1.645 * std_val / np.sqrt(n) if n > 1 else 0
                
                plot_data.append({
                    'x': x_pos,
                    'mean': mean_val,
                    'ci': ci_90,
                    'type': 'Baseline',
                    'scenario': scenario,
                    'n': n
                })
                x_positions.append(x_pos)
                x_labels.append(f"{scenario.replace('_', ' ').title()}\nBaseline")
                x_pos += 1
            
            # Add shift point  
            if len(shift_data) > 0:
                mean_val = shift_data.mean()
                std_val = shift_data.std()
                n = len(shift_data)
                ci_90 = 1.645 * std_val / np.sqrt(n) if n > 1 else 0
                
                plot_data.append({
                    'x': x_pos,
                    'mean': mean_val,
                    'ci': ci_90,
                    'type': 'Shift',
                    'scenario': scenario,
                    'n': n
                })
                x_positions.append(x_pos)
                x_labels.append(f"{scenario.replace('_', ' ').title()}\nShift")
                x_pos += 1
        
        # Plot data points with error bars
        for item in plot_data:
            color = baseline_color if item['type'] == 'Baseline' else shift_color
            marker = 's' if item['type'] == 'Baseline' else 'o'
            
            ax.errorbar(item['x'], item['mean'], yerr=item['ci'], 
                       fmt=marker, color=color, capsize=5, capthick=2, 
                       markersize=8, linewidth=2, 
                       label=item['type'] if item['x'] == 0 else "")
        
        # Add baseline reference line
        if not np.isnan(overall_baseline_mean):
            ax.axhline(y=float(overall_baseline_mean), color=baseline_color, linestyle='--', 
                      linewidth=2, alpha=0.7, label=f'Baseline Mean')
        
        # Add confidence interval band for baseline
        baseline_std = np.nanstd([item['mean'] for item in plot_data if item['type'] == 'Baseline'])
        if not np.isnan(baseline_std):
            ax.fill_between([-0.5, len(x_positions)-0.5], 
                           overall_baseline_mean - baseline_std, 
                           overall_baseline_mean + baseline_std,
                           color=baseline_color, alpha=0.1, label='Baseline 90% CI')
        
        # Customize plot
        ax.set_xlabel('Scenario Configuration', fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        if ylim:
            ax.set_ylim(ylim)
            
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add sample size annotations
        for item in plot_data:
            ax.annotate(f'n={item["n"]}', 
                       xy=(item['x'], item['mean'] + item['ci']), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8,
                       color='darkgray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved baseline reference plot: {save_path}")
            
        return fig
    
    def create_change_from_baseline_plot(self, metric: str, title: str, ylabel: str,
                                       figsize: Tuple[int, int] = (14, 8),
                                       save_path: Optional[str] = None) -> matplotlib.figure.Figure:
        """Create plot similar to 'HorizontalCR - Ph5 Proxy Change from Baseline'."""
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Colors for positive/negative changes
        positive_color = '#FF6B6B'  # Red for increases (worse)
        negative_color = '#4ECDC4'  # Teal for decreases (better) 
        
        # Calculate baseline references
        baseline_refs = self.calculate_baseline_reference(metric)
        
        # Prepare data
        plot_data = []
        scenarios = sorted(self.scenarios)
        
        for i, scenario in enumerate(scenarios):
            baseline_ref = baseline_refs.get(scenario, np.nan)
            
            if not np.isnan(baseline_ref):
                # Get shift performance
                shift_data = self.df[
                    (self.df['test_scenario'] == scenario) & 
                    (self.df['model_scenario_type'] == 'Shift')
                ][metric].dropna()
                
                if len(shift_data) > 0:
                    shift_mean = shift_data.mean()
                    shift_std = shift_data.std()
                    n = len(shift_data)
                    
                    # Calculate change from baseline
                    delta = shift_mean - baseline_ref
                    delta_ci = 1.645 * shift_std / np.sqrt(n) if n > 1 else 0
                    
                    plot_data.append({
                        'scenario': scenario,
                        'delta': delta,
                        'delta_ci': delta_ci,
                        'x': i,
                        'n': n,
                        'baseline_ref': baseline_ref,
                        'shift_mean': shift_mean
                    })
        
        # Plot change from baseline
        for item in plot_data:
            color = positive_color if item['delta'] > 0 else negative_color
            
            ax.errorbar(item['x'], item['delta'], yerr=item['delta_ci'],
                       fmt='o', color=color, capsize=5, capthick=2,
                       markersize=10, linewidth=2)
        
        # Add zero reference line (no change)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
        
        # Add positive/negative regions
        y_min, y_max = ax.get_ylim()
        ax.fill_between([-0.5, len(scenarios)-0.5], 0, y_max, 
                       color=positive_color, alpha=0.1, label='Worse than Baseline')
        ax.fill_between([-0.5, len(scenarios)-0.5], y_min, 0, 
                       color=negative_color, alpha=0.1, label='Better than Baseline')
        
        # Customize plot
        ax.set_xlabel('Scenario Configuration', fontweight='bold', fontsize=12)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
        
        scenario_labels = [s.replace('_', ' ').title() for s in scenarios]
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenario_labels, rotation=45, ha='right')
        
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add sample size and values annotations
        for item in plot_data:
            # Sample size
            ax.annotate(f'n={item["n"]}', 
                       xy=(item['x'], item['delta'] + item['delta_ci']), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8, color='darkgray')
            
            # Delta value
            delta_text = f"Δ={item['delta']:.3f}"
            ax.annotate(delta_text,
                       xy=(item['x'], item['delta']),
                       xytext=(15, 0), textcoords='offset points',
                       ha='left', va='center', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved change from baseline plot: {save_path}")
            
        return fig
    
    def create_enhanced_dashboard(self, output_dir: str):
        """Create complete enhanced dashboard matching reference plot styles."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎨 Creating Enhanced Scenario Performance Dashboard")
        print("=" * 65)
        
        # Key metrics with plot configurations
        metrics_config = [
            {
                'metric': 'fn_rate',
                'title': 'ScenarioSR - Model Fn Rate\nMean ± 90% Confidence Interval',
                'ylabel': 'Model Fn Rate',
                'ylim': (0, 0.15),
                'change_title': 'ScenarioSR - Fn Rate Change from Baseline\nChange from Baseline ± 90% Confidence Interval',
                'change_ylabel': 'Δ Fn Rate from Baseline'
            },
            {
                'metric': 'fp_rate', 
                'title': 'ScenarioSR - Model Fp Rate\nMean ± 90% Confidence Interval',
                'ylabel': 'Model Fp Rate',
                'ylim': (0, 0.1),
                'change_title': 'ScenarioSR - Fp Rate Change from Baseline\nChange from Baseline ± 90% Confidence Interval',
                'change_ylabel': 'Δ Fp Rate from Baseline'
            },
            {
                'metric': 'num_los_events',
                'title': 'ScenarioSR - LOS Events\nMean ± 90% Confidence Interval', 
                'ylabel': 'Number of LOS Events',
                'ylim': (0, None),
                'change_title': 'ScenarioSR - LOS Events Change from Baseline\nChange from Baseline ± 90% Confidence Interval',
                'change_ylabel': 'Δ LOS Events from Baseline'
            },
            {
                'metric': 'min_separation_nm',
                'title': 'ScenarioSR - Minimum Separation\nMean ± 90% Confidence Interval',
                'ylabel': 'Minimum Separation (NM)', 
                'ylim': (0, None),
                'change_title': 'ScenarioSR - Min Separation Change from Baseline\nChange from Baseline ± 90% Confidence Interval',
                'change_ylabel': 'Δ Min Separation (NM) from Baseline'
            },
            {
                'metric': 'reward_total',
                'title': 'ScenarioSR - Total Reward\nMean ± 90% Confidence Interval',
                'ylabel': 'Total Reward',
                'ylim': (None, None),
                'change_title': 'ScenarioSR - Reward Change from Baseline\nChange from Baseline ± 90% Confidence Interval', 
                'change_ylabel': 'Δ Total Reward from Baseline'
            },
            {
                'metric': 'flight_time_min',
                'title': 'ScenarioSR - Flight Time\nMean ± 90% Confidence Interval',
                'ylabel': 'Flight Time (minutes)',
                'ylim': (0, None),
                'change_title': 'ScenarioSR - Flight Time Change from Baseline\nChange from Baseline ± 90% Confidence Interval',
                'change_ylabel': 'Δ Flight Time (min) from Baseline'
            }
        ]
        
        # Generate baseline reference plots
        print(f"\n📊 Generating baseline reference plots...")
        for config in metrics_config:
            save_path = output_path / f"enhanced_{config['metric']}_baseline_ref.png"
            
            fig = self.create_baseline_reference_plot(
                metric=config['metric'],
                title=config['title'],
                ylabel=config['ylabel'], 
                ylim=config['ylim'],
                save_path=str(save_path)
            )
            if fig:
                plt.close(fig)
        
        # Generate change from baseline plots
        print(f"\n📈 Generating change from baseline plots...")
        for config in metrics_config:
            save_path = output_path / f"enhanced_{config['metric']}_change_from_baseline.png"
            
            fig = self.create_change_from_baseline_plot(
                metric=config['metric'],
                title=config['change_title'],
                ylabel=config['change_ylabel'],
                save_path=str(save_path)
            )
            if fig:
                plt.close(fig)
        
        # Create summary comparison table
        self.create_enhanced_summary_table(str(output_path))
        
        print(f"\n✅ Enhanced dashboard completed!")
        print(f"📁 Generated files in: {output_path}")
        
    def create_enhanced_summary_table(self, output_dir: str):
        """Create enhanced summary table with baseline comparisons."""
        
        key_metrics = ['fn_rate', 'fp_rate', 'num_los_events', 'min_separation_nm', 
                      'flight_time_min', 'waypoint_reached_ratio', 'reward_total']
        
        summary_data = []
        
        for scenario in self.scenarios:
            scenario_data = self.df[self.df['test_scenario'] == scenario]
            
            baseline_data = scenario_data[scenario_data['model_scenario_type'] == 'Baseline']
            shift_data = scenario_data[scenario_data['model_scenario_type'] == 'Shift']
            
            row = {
                'scenario': scenario,
                'baseline_episodes': len(baseline_data),
                'shift_episodes': len(shift_data)
            }
            
            for metric in key_metrics:
                # Baseline statistics
                if len(baseline_data) > 0:
                    baseline_values = baseline_data[metric].dropna()
                    if len(baseline_values) > 0:
                        row[f'{metric}_baseline_mean'] = baseline_values.mean()
                        row[f'{metric}_baseline_std'] = baseline_values.std()
                        row[f'{metric}_baseline_ci90'] = 1.645 * baseline_values.std() / np.sqrt(len(baseline_values))
                    else:
                        row[f'{metric}_baseline_mean'] = np.nan
                        row[f'{metric}_baseline_std'] = np.nan
                        row[f'{metric}_baseline_ci90'] = np.nan
                else:
                    row[f'{metric}_baseline_mean'] = np.nan
                    row[f'{metric}_baseline_std'] = np.nan
                    row[f'{metric}_baseline_ci90'] = np.nan
                
                # Shift statistics
                if len(shift_data) > 0:
                    shift_values = shift_data[metric].dropna()
                    if len(shift_values) > 0:
                        shift_mean = shift_values.mean()
                        row[f'{metric}_shift_mean'] = shift_mean
                        row[f'{metric}_shift_std'] = shift_values.std()
                        row[f'{metric}_shift_ci90'] = 1.645 * shift_values.std() / np.sqrt(len(shift_values))
                        
                        # Calculate change from baseline
                        baseline_mean = row[f'{metric}_baseline_mean']
                        if not np.isnan(baseline_mean):
                            row[f'{metric}_delta'] = shift_mean - baseline_mean
                            row[f'{metric}_delta_pct'] = ((shift_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else np.nan
                        else:
                            row[f'{metric}_delta'] = np.nan
                            row[f'{metric}_delta_pct'] = np.nan
                    else:
                        row[f'{metric}_shift_mean'] = np.nan
                        row[f'{metric}_shift_std'] = np.nan
                        row[f'{metric}_shift_ci90'] = np.nan
                        row[f'{metric}_delta'] = np.nan
                        row[f'{metric}_delta_pct'] = np.nan
                else:
                    row[f'{metric}_shift_mean'] = np.nan
                    row[f'{metric}_shift_std'] = np.nan
                    row[f'{metric}_shift_ci90'] = np.nan
                    row[f'{metric}_delta'] = np.nan
                    row[f'{metric}_delta_pct'] = np.nan
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save detailed summary
        save_path = Path(output_dir) / 'enhanced_scenario_performance_summary.csv'
        summary_df.to_csv(save_path, index=False)
        
        # Create concise delta summary  
        delta_columns = ['scenario'] + [col for col in summary_df.columns if '_delta' in col and '_pct' not in col]
        delta_summary = summary_df[delta_columns].copy()
        
        delta_save_path = Path(output_dir) / 'scenario_deltas_from_baseline.csv'
        delta_summary.to_csv(delta_save_path, index=False)
        
        print(f"✅ Saved enhanced summary: {save_path}")
        print(f"✅ Saved delta summary: {delta_save_path}")
        
        return summary_df


def main():
    parser = argparse.ArgumentParser(description='Generate enhanced scenario performance plots matching reference style')
    parser.add_argument('--data', required=True, help='Path to baseline_vs_shift_detailed_summary.csv')
    parser.add_argument('--output', default='enhanced_scenario_plots', help='Output directory for plots')
    parser.add_argument('--metric', help='Generate single metric plot (e.g., fp_rate, fn_rate)')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving')
    
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"❌ Data file not found: {args.data}")
        return 1
    
    # Create analyzer
    analyzer = EnhancedScenarioAnalyzer(args.data)
    
    if args.metric:
        # Generate single metric plots
        if args.show:
            # Show baseline reference plot
            fig1 = analyzer.create_baseline_reference_plot(
                args.metric, 
                f'Scenario Analysis - {args.metric.title()}\nMean ± 90% Confidence Interval',
                args.metric.replace('_', ' ').title()
            )
            plt.show()
            plt.close(fig1)
            
            # Show change from baseline plot
            fig2 = analyzer.create_change_from_baseline_plot(
                args.metric,
                f'Scenario Analysis - {args.metric.title()} Change from Baseline\nChange from Baseline ± 90% Confidence Interval',
                f'Δ {args.metric.replace("_", " ").title()} from Baseline'
            )
            plt.show()
            plt.close(fig2)
        else:
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save baseline reference plot
            save_path1 = output_path / f"enhanced_{args.metric}_baseline_ref.png"
            fig1 = analyzer.create_baseline_reference_plot(
                args.metric,
                f'Scenario Analysis - {args.metric.title()}\nMean ± 90% Confidence Interval', 
                args.metric.replace('_', ' ').title(),
                save_path=str(save_path1)
            )
            plt.close(fig1)
            
            # Save change from baseline plot
            save_path2 = output_path / f"enhanced_{args.metric}_change_from_baseline.png"
            fig2 = analyzer.create_change_from_baseline_plot(
                args.metric,
                f'Scenario Analysis - {args.metric.title()} Change from Baseline\nChange from Baseline ± 90% Confidence Interval',
                f'Δ {args.metric.replace("_", " ").title()} from Baseline',
                save_path=str(save_path2)
            )
            plt.close(fig2)
    else:
        # Generate complete dashboard
        analyzer.create_enhanced_dashboard(args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())