#!/usr/bin/env python3
"""
Proper Survival Curve Analysis for Minimum Horizontal Separation

Implements the correct mathematical definition:
- Per episode: X_i = min_t min_pairs HMD_p(t) 
- Empirical survival function: S(x) = Pr(X >= x) = (1/N) * sum(1{X_i >= x})
- Plots Reliability Curves with 5 NM safety threshold and optional confidence intervals

Usage:
    python survival_curve_analysis.py --data_dir "INTER_SHIFT_1K_E_290925" --output "survival_plots"
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import glob
import os
from scipy import stats
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

class SurvivalCurveAnalyzer:
    def __init__(self, data_dir: str):
        """Initialize with path to episode data directory."""
        self.data_dir = Path(data_dir)
        self.episode_min_separations = {}  # {(model_alias, test_scenario): [X_1, X_2, ..., X_N]}
        self.load_episode_data()
        
    def load_episode_data(self):
        """Load trajectory data and calculate episode-level minimum separations."""
        
        print("🔄 Loading episode trajectory data...")
        
        # Find all model-scenario combinations
        model_scenario_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and "__on__" in d.name]
        
        for model_scenario_dir in model_scenario_dirs:
            dir_name = model_scenario_dir.name
            
            # Parse directory name: PPO_model__on__scenario__baseline or PPO_model__on__scenario
            parts = dir_name.split("__on__")
            if len(parts) != 2:
                continue
                
            model_alias = parts[0]  # e.g., "PPO_chase_2x2_20251008_015945" or "PPO_generic_20251008_225941"
            scenario_part = parts[1]  # e.g., "chase_2x2__baseline" or "chase_3p1" or "chase_2x2__generic_shift"
            
            # Extract test scenario (remove suffix tags)
            if "__baseline" in scenario_part:
                test_scenario = scenario_part.replace("__baseline", "")
            elif "__generic_shift" in scenario_part:
                test_scenario = scenario_part.replace("__generic_shift", "")
            else:
                test_scenario = scenario_part
            
            print(f"📂 Processing: {model_alias} on {test_scenario}")
            
            # Load episode minimum separations
            episode_mins = self.load_model_scenario_episodes(model_scenario_dir)
            
            if episode_mins:
                self.episode_min_separations[(model_alias, test_scenario)] = episode_mins
                print(f"   ✅ Loaded {len(episode_mins)} episodes")
            else:
                print(f"   ⚠️ No valid episodes found")
        
        print(f"\\n📊 Total loaded: {len(self.episode_min_separations)} model-scenario combinations")
    
    def load_model_scenario_episodes(self, model_scenario_dir: Path) -> List[float]:
        """Load all episodes for a specific model-scenario combination."""
        
        episode_mins = []
        
        # Find all episode directories
        episode_dirs = [d for d in model_scenario_dir.iterdir() if d.is_dir() and d.name.startswith("ep_")]
        
        for episode_dir in episode_dirs:
            # Find trajectory CSV file
            traj_files = list(episode_dir.glob("traj_ep_*.csv"))
            
            if traj_files:
                traj_file = traj_files[0]  # Take first match
                episode_min = self.calculate_episode_minimum_separation(traj_file)
                
                if episode_min is not None:
                    episode_mins.append(episode_min)
        
        return episode_mins
    
    def calculate_episode_minimum_separation(self, traj_file: Path) -> Optional[float]:
        """Calculate X_i = min_t min_pairs HMD_p(t) for one episode."""
        
        try:
            # Load trajectory data
            df = pd.read_csv(traj_file)
            
            if 'min_separation_nm' in df.columns:
                # If min_separation_nm is already computed at each timestep
                episode_min = df['min_separation_nm'].min()
                return float(episode_min)
            
            else:
                # Calculate pairwise distances manually
                timesteps = df['step_idx'].unique()
                min_separations_per_timestep = []
                
                for step in timesteps:
                    step_data = df[df['step_idx'] == step]
                    
                    if len(step_data) < 2:
                        continue  # Need at least 2 aircraft
                    
                    # Calculate pairwise horizontal distances
                    agents = step_data['agent_id'].unique()
                    min_dist_this_step = float('inf')
                    
                    for i, agent1 in enumerate(agents):
                        for agent2 in agents[i+1:]:
                            agent1_data = step_data[step_data['agent_id'] == agent1].iloc[0]
                            agent2_data = step_data[step_data['agent_id'] == agent2].iloc[0]
                            
                            # Calculate horizontal distance using haversine formula
                            lat1, lon1 = agent1_data['lat_deg'], agent1_data['lon_deg']
                            lat2, lon2 = agent2_data['lat_deg'], agent2_data['lon_deg']
                            
                            dist_nm = self.haversine_distance(lat1, lon1, lat2, lon2)
                            min_dist_this_step = min(min_dist_this_step, dist_nm)
                    
                    if min_dist_this_step != float('inf'):
                        min_separations_per_timestep.append(min_dist_this_step)
                
                if min_separations_per_timestep:
                    episode_min = min(min_separations_per_timestep)
                    return float(episode_min)
                    
        except Exception as e:
            print(f"⚠️ Error processing {traj_file}: {e}")
            return None
        
        return None
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate horizontal distance between two lat/lon points in nautical miles."""
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in nautical miles
        r_nm = 3440.065  # nautical miles
        
        return r_nm * c
    
    def calculate_empirical_survival_function(self, episode_mins: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate empirical survival function S(x) = Pr(X >= x)."""
        
        if not episode_mins:
            return np.array([]), np.array([])
        
        # Sort episode minimums ascending: x_(1) <= ... <= x_(N)
        x_sorted = np.sort(episode_mins)
        n = len(x_sorted)
        
        # Survival probabilities: S(x_(k)) = 1 - k/N
        survival_probs = 1 - np.arange(1, n + 1) / n
        
        return x_sorted, survival_probs
    
    def bootstrap_confidence_intervals(self, episode_mins: List[float], n_bootstrap: int = 1000, 
                                     confidence_level: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate bootstrap confidence intervals for survival function."""
        
        if len(episode_mins) < 10:  # Need sufficient data for bootstrap
            return None, None, None
        
        x_eval = np.linspace(min(episode_mins), max(episode_mins), 100)
        bootstrap_survivals = []
        
        for _ in range(n_bootstrap):
            # Bootstrap resample
            bootstrap_sample = np.random.choice(episode_mins, size=len(episode_mins), replace=True)
            x_boot, s_boot = self.calculate_empirical_survival_function(bootstrap_sample.tolist())
            
            # Interpolate to common x values
            s_interp = np.interp(x_eval, x_boot, s_boot, left=1.0, right=0.0)
            bootstrap_survivals.append(s_interp)
        
        bootstrap_survivals = np.array(bootstrap_survivals)
        
        # Calculate percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_survivals, lower_percentile, axis=0)
        ci_upper = np.percentile(bootstrap_survivals, upper_percentile, axis=0)
        
        return x_eval, ci_lower, ci_upper
    
    def create_scenario_survival_plot(self, test_scenario: str, save_path: Optional[str] = None,
                                    include_confidence_intervals: bool = True):
        """Create survival curve plot for one test scenario."""
        
        # Find all models tested on this scenario
        scenario_data = {}
        for (model_alias, scenario), episode_mins in self.episode_min_separations.items():
            if scenario == test_scenario:
                scenario_data[model_alias] = episode_mins
        
        if not scenario_data:
            print(f"⚠️ No data found for scenario: {test_scenario}")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Colors for different models
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Safety threshold
        safety_threshold = 5.0  # 5 NM
        
        for i, (model_alias, episode_mins) in enumerate(scenario_data.items()):
            model_short = model_alias.replace('PPO_', '')
            
            # Determine if this is the baseline model (frozen scenario-specific) or generic model
            is_generic = 'generic' in model_short.lower()
            
            # Extract training scenario from model name (e.g., "chase_2x2_20251008_015945" -> "chase_2x2")
            # Models are named: PPO_{scenario}_{timestamp} or PPO_generic_{timestamp}
            if not is_generic:
                # Remove timestamp suffix (format: _YYYYMMDD_HHMMSS)
                model_parts = model_short.rsplit('_', 2)  # Split from right: ['chase_2x2', '20251008', '015945']
                if len(model_parts) >= 3 and model_parts[-2].isdigit() and model_parts[-1].isdigit():
                    baseline_scenario = model_parts[0]  # e.g., "chase_2x2"
                else:
                    baseline_scenario = model_short  # Fallback if timestamp pattern not found
            else:
                baseline_scenario = None
                
            is_baseline = (not is_generic and baseline_scenario == test_scenario)
            
            # Calculate survival function
            x_vals, survival_vals = self.calculate_empirical_survival_function(episode_mins)
            
            if len(x_vals) == 0:
                continue
            
            # Plot styling
            color = colors[i % len(colors)]
            linestyle = '-' if is_baseline else (':' if is_generic else '--')
            linewidth = 3 if is_baseline else (2.5 if is_generic else 2)
            alpha = 1.0 if is_baseline else 0.8
            
            # Label with markers: * = baseline (trained on this scenario), G = generic model
            marker = '*' if is_baseline else ('G' if is_generic else '')
            label = f'{model_short}{marker} (N={len(episode_mins)})'
            
            # Plot survival curve
            ax.plot(x_vals, survival_vals, color=color, linestyle=linestyle, 
                   linewidth=linewidth, alpha=alpha, label=label)
            
            # Add confidence intervals if requested and sufficient data
            if include_confidence_intervals and len(episode_mins) >= 20:
                x_ci, ci_lower, ci_upper = self.bootstrap_confidence_intervals(episode_mins)
                
                if x_ci is not None:
                    ax.fill_between(x_ci, ci_lower, ci_upper, color=color, alpha=0.2)
            
            # Calculate and annotate Pr(X < 5) = episode-level LoS risk
            if len(x_vals) > 0:
                los_risk = 1 - np.interp(safety_threshold, x_vals, survival_vals, left=1.0, right=0.0)
                
                # Add text annotation
                y_pos = 0.95 - i * 0.05
                ax.text(0.02, y_pos, f'{model_short}: P(LoS) = {los_risk:.3f}', 
                       transform=ax.transAxes, fontsize=9, color=color,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add 5 NM safety rule line
        ax.axvline(x=safety_threshold, color='red', linestyle=':', 
                  linewidth=3, alpha=0.8, label='5 NM Safety Rule')
        
        # Customize plot
        ax.set_xlabel('Minimum Horizontal Separation (NM)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Reliability Probability S(x) = Pr(X ≥ x)', fontweight='bold', fontsize=12)
        ax.set_title(f'Minimum Separation Reliability Curves\nTest Scenario: {test_scenario.replace("_", " ").title()}', 
                    fontweight='bold', fontsize=14, pad=20)
        
        # Set limits
        ax.set_xlim(left=0, right=20)  # Focus on 0-20 NM range
        ax.set_ylim(0, 1)
        
        # Legend
        ax.legend(loc='center right', fontsize=9, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Add definition text
        definition_text = (
            "Definition: X_i = min over time & pairs of HMD_p(t) per episode\n"
            "Reliability Function: S(x) = Pr(X ≥ x) = (1/N) × Σ I{X_i ≥ x}"
        )
        ax.text(0.98, 0.02, definition_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved: {Path(save_path).name}")
        
        return fig
    
    def generate_all_survival_plots(self, output_dir: str, include_confidence_intervals: bool = True):
        """Generate survival plots for all test scenarios."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\\n🎨 Generating Proper Survival Curve Analysis")
        print("=" * 55)
        
        # Get all unique test scenarios
        test_scenarios = set()
        for (model_alias, scenario) in self.episode_min_separations.keys():
            test_scenarios.add(scenario)
        
        test_scenarios = sorted(test_scenarios)
        
        for scenario in test_scenarios:
            print(f"\\n📊 {scenario.replace('_', ' ').title()}")
            
            save_path = output_path / f"survival_curve_{scenario}.png"
            
            fig = self.create_scenario_survival_plot(
                test_scenario=scenario,
                save_path=str(save_path),
                include_confidence_intervals=include_confidence_intervals
            )
            
            if fig:
                plt.close(fig)
        
        # Generate summary statistics
        self.export_survival_statistics(str(output_path))
        
        print(f"\\n✅ Survival curve analysis completed!")
        print(f"📁 Output: {output_path}")
    
    def export_survival_statistics(self, output_dir: str):
        """Export survival statistics and LoS risk calculations."""
        
        stats_data = []
        
        for (model_alias, test_scenario), episode_mins in self.episode_min_separations.items():
            if not episode_mins:
                continue
            
            model_short = model_alias.replace('PPO_', '')
            is_generic = 'generic' in model_short.lower()
            
            # Extract training scenario from model name (e.g., "chase_2x2_20251008_015945" -> "chase_2x2")
            if not is_generic:
                # Remove timestamp suffix (format: _YYYYMMDD_HHMMSS)
                model_parts = model_short.rsplit('_', 2)  # Split from right
                if len(model_parts) >= 3 and model_parts[-2].isdigit() and model_parts[-1].isdigit():
                    baseline_scenario = model_parts[0]  # e.g., "chase_2x2"
                else:
                    baseline_scenario = model_short  # Fallback
            else:
                baseline_scenario = None
                
            is_baseline = (not is_generic and baseline_scenario == test_scenario)
            
            # Calculate survival function
            x_vals, survival_vals = self.calculate_empirical_survival_function(episode_mins)
            
            if len(x_vals) == 0:
                continue
            
            # Calculate key statistics
            mean_min_sep = np.mean(episode_mins)
            median_min_sep = np.median(episode_mins)
            std_min_sep = np.std(episode_mins)
            
            # LoS risk (Pr(X < 5))
            los_risk = 1 - np.interp(5.0, x_vals, survival_vals, left=1.0, right=0.0)
            
            # Percentiles
            p5 = np.percentile(episode_mins, 5)
            p10 = np.percentile(episode_mins, 10)
            p25 = np.percentile(episode_mins, 25)
            p75 = np.percentile(episode_mins, 75)
            p90 = np.percentile(episode_mins, 90)
            p95 = np.percentile(episode_mins, 95)
            
            stats_data.append({
                'model_alias': model_alias,
                'model_short': model_short,
                'test_scenario': test_scenario,
                'is_baseline': is_baseline,
                'is_generic': is_generic,
                'n_episodes': len(episode_mins),
                'mean_min_sep_nm': mean_min_sep,
                'median_min_sep_nm': median_min_sep,
                'std_min_sep_nm': std_min_sep,
                'los_risk_pr_x_lt_5': los_risk,
                'min_separation_nm': min(episode_mins),
                'max_separation_nm': max(episode_mins),
                'p5_min_sep_nm': p5,
                'p10_min_sep_nm': p10,
                'p25_min_sep_nm': p25,
                'p75_min_sep_nm': p75,
                'p90_min_sep_nm': p90,
                'p95_min_sep_nm': p95
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Save statistics
        save_path = Path(output_dir) / 'survival_curve_statistics.csv'
        stats_df.to_csv(save_path, index=False)
        
        print(f"✅ Saved statistics: {save_path.name}")
        
        return stats_df


def main():
    parser = argparse.ArgumentParser(description='Generate proper survival curve analysis')
    parser.add_argument('--data_dir', required=True, help='Path to episode data directory (e.g., INTER_SHIFT_1K_E_290925)')
    parser.add_argument('--output', default='survival_analysis', help='Output directory')
    parser.add_argument('--scenario', help='Single scenario to analyze')
    parser.add_argument('--no_ci', action='store_true', help='Skip confidence intervals (faster)')
    
    args = parser.parse_args()
    
    if not Path(args.data_dir).exists():
        print(f"❌ Data directory not found: {args.data_dir}")
        return 1
    
    # Create analyzer
    analyzer = SurvivalCurveAnalyzer(args.data_dir)
    
    if args.scenario:
        # Generate single scenario plot
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        save_path = output_path / f"survival_curve_{args.scenario}.png"
        fig = analyzer.create_scenario_survival_plot(
            test_scenario=args.scenario,
            save_path=str(save_path),
            include_confidence_intervals=not args.no_ci
        )
        if fig:
            plt.show()
            plt.close(fig)
    else:
        # Generate all survival plots
        analyzer.generate_all_survival_plots(
            args.output, 
            include_confidence_intervals=not args.no_ci
        )
    
    return 0


if __name__ == "__main__":
    exit(main())