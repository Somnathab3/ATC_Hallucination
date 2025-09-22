"""
Main script for running targeted distribution shift tests.

This script implements the suggested shift protocol:
1. Baseline: 10 episodes on canonical scenario (frozen)
2. Targeted shifts: for each agent, vary one parameter while others remain nominal
3. Micro to macro range variations to identify training model failures

Key features:
- Only one agent modified per test case
- Comprehensive conflict-inducing scenarios
- Enhanced analysis comparing to baseline performance
- Identification of most vulnerable configurations
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
from datetime import datetime

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC
from src.testing.targeted_shift_tester import run_targeted_shift_grid

logging.basicConfig(level=logging.INFO)

def run_baseline_comparison(repo_root: str, algo_class, checkpoint_path: str, baseline_episodes: int = 10):
    """
    Run baseline episodes (no shifts) for comparison with targeted shift results.
    """
    from src.testing.shift_tester import run_shift_grid
    
    print("Running baseline comparison (no shifts)...")
    
    # Run with all shift values set to 0 (baseline)
    # This gives us the canonical scenario performance
    baseline_csv = run_shift_grid(
        repo_root=repo_root,
        algo_class=algo_class, 
        checkpoint_path=checkpoint_path,
        episodes_per_shift=baseline_episodes,
        seeds=list(range(baseline_episodes))
    )
    
    return baseline_csv

def analyze_baseline_vs_targeted(baseline_csv: str, targeted_csv: str, output_dir: str):
    """
    Compare baseline performance with targeted shift results.
    """
    # Load data
    baseline_df = pd.read_csv(baseline_csv)
    targeted_df = pd.read_csv(targeted_csv)
    
    # Filter baseline to only no-shift cases
    baseline_clean = baseline_df[
        (baseline_df['shift_value'] == 0) & 
        (baseline_df['shift_type'].isin(['speed', 'position', 'heading']))
    ].copy()
    
    if baseline_clean.empty:
        print("Warning: No baseline (zero-shift) data found")
        return
    
    # Calculate baseline statistics
    baseline_stats = {
        'avg_conflicts': baseline_clean['num_los_events'].mean(),
        'avg_los_duration': baseline_clean['total_los_duration'].mean(), 
        'avg_min_separation': baseline_clean['min_separation_nm'].mean(),
        'avg_path_efficiency': baseline_clean['path_efficiency'].mean(),
        'avg_waypoint_completion': baseline_clean['waypoint_reached_ratio'].mean(),
        'avg_resolution_fail_rate': baseline_clean['resolution_fail_rate'].mean(),
        'total_episodes': len(baseline_clean)
    }
    
    # Analyze targeted shifts by categories
    analysis_results = {}
    
    # By shift range (micro vs macro)
    range_comparison = targeted_df.groupby('shift_range').agg({
        'num_los_events': ['mean', 'std', 'max'],
        'total_los_duration': ['mean', 'std', 'max'],
        'min_separation_nm': ['mean', 'std', 'min'],
        'path_efficiency': ['mean', 'std'],
        'waypoint_reached_ratio': ['mean', 'std'],
        'resolution_fail_rate': ['mean', 'std']
    }).round(4)
    
    analysis_results['range_comparison'] = range_comparison
    
    # By target agent 
    agent_comparison = targeted_df.groupby('target_agent').agg({
        'num_los_events': ['mean', 'std', 'sum'],
        'total_los_duration': ['mean', 'std', 'sum'],
        'min_separation_nm': ['mean', 'std', 'min'],
        'resolution_fail_rate': ['mean', 'std']
    }).round(4)
    
    analysis_results['agent_comparison'] = agent_comparison
    
    # By shift type
    type_comparison = targeted_df.groupby('shift_type').agg({
        'num_los_events': ['mean', 'std', 'sum'],
        'total_los_duration': ['mean', 'std', 'sum'],
        'min_separation_nm': ['mean', 'std', 'min'],
        'resolution_fail_rate': ['mean', 'std']
    }).round(4)
    
    analysis_results['type_comparison'] = type_comparison
    
    # Find most problematic configurations
    most_conflicts = targeted_df.nlargest(10, 'num_los_events')[
        ['test_id', 'target_agent', 'shift_type', 'shift_value', 'num_los_events', 'total_los_duration', 'min_separation_nm']
    ]
    
    analysis_results['most_conflicts'] = most_conflicts
    
    # Calculate improvement/degradation relative to baseline
    relative_performance = {}
    for metric in ['num_los_events', 'total_los_duration', 'min_separation_nm', 'path_efficiency', 'waypoint_reached_ratio']:
        baseline_val = baseline_stats.get(f'avg_{metric.replace("num_", "avg_").replace("total_", "avg_")}', 0)
        if baseline_val > 0:
            targeted_mean = targeted_df[metric].mean()
            relative_change = ((targeted_mean - baseline_val) / baseline_val) * 100
            relative_performance[metric] = {
                'baseline': baseline_val,
                'targeted_mean': targeted_mean,
                'percent_change': relative_change
            }
    
    analysis_results['relative_performance'] = relative_performance
    
    # Save comprehensive comparison report
    comparison_report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'baseline_stats': baseline_stats,
        'targeted_summary': {
            'total_configurations': targeted_df['test_id'].nunique(),
            'total_episodes': len(targeted_df),
            'agents_tested': targeted_df['target_agent'].nunique(),
            'shift_types_tested': targeted_df['shift_type'].nunique()
        },
        'key_findings': analysis_results,
        'recommendations': generate_recommendations(baseline_stats, targeted_df)
    }
    
    # Save to JSON
    comparison_path = os.path.join(output_dir, "baseline_vs_targeted_analysis.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2, default=str)
    
    # Save CSV summaries
    range_comparison.to_csv(os.path.join(output_dir, "range_comparison.csv"))
    agent_comparison.to_csv(os.path.join(output_dir, "agent_comparison.csv"))
    type_comparison.to_csv(os.path.join(output_dir, "type_comparison.csv"))
    most_conflicts.to_csv(os.path.join(output_dir, "most_problematic_configs.csv"), index=False)
    
    print(f"Comprehensive comparison analysis saved to: {output_dir}")
    return comparison_report

def generate_recommendations(baseline_stats: dict, targeted_df: pd.DataFrame) -> list:
    """Generate actionable recommendations based on analysis results."""
    recommendations = []
    
    # Check if macro shifts cause significantly more conflicts than baseline
    macro_conflicts = targeted_df[targeted_df['shift_range'] == 'macro']['num_los_events'].mean()
    baseline_conflicts = baseline_stats['avg_conflicts']
    
    if macro_conflicts > baseline_conflicts * 2:
        recommendations.append({
            "priority": "HIGH",
            "category": "Model Robustness",
            "finding": f"Macro shifts cause {macro_conflicts/baseline_conflicts:.1f}x more conflicts than baseline",
            "recommendation": "Augment training data with larger perturbations to improve generalization"
        })
    
    # Check which agent is most vulnerable
    agent_conflicts = targeted_df.groupby('target_agent')['num_los_events'].sum()
    most_vulnerable_agent = agent_conflicts.idxmax()
    
    recommendations.append({
        "priority": "MEDIUM", 
        "category": "Agent-Specific Vulnerability",
        "finding": f"Agent {most_vulnerable_agent} shows highest conflict rate when modified",
        "recommendation": f"Focus additional testing and training on {most_vulnerable_agent} scenarios"
    })
    
    # Check position shifts specifically
    pos_conflicts = targeted_df[targeted_df['shift_type'].str.contains('position')]['num_los_events'].mean()
    if pos_conflicts > baseline_conflicts * 1.5:
        recommendations.append({
            "priority": "HIGH",
            "category": "Position Sensitivity", 
            "finding": "Position shifts cause disproportionate conflicts",
            "recommendation": "Implement position uncertainty modeling in training scenarios"
        })
    
    # Check resolution failures
    avg_resolution_fail = targeted_df['resolution_fail_rate'].mean()
    if avg_resolution_fail > 0.2:  # More than 20% failure rate
        recommendations.append({
            "priority": "CRITICAL",
            "category": "Resolution Capability",
            "finding": f"High resolution failure rate: {avg_resolution_fail:.1%}",
            "recommendation": "Enhance conflict resolution training with multi-agent coordination"
        })
    
    return recommendations

def main():
    """Main execution function for targeted shift testing."""
    # Add project root to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    REPO = current_dir
    
    if REPO not in sys.path:
        sys.path.append(REPO)
    
    # Look for parallel scenario models first
    parallel_model_path = os.path.join(REPO, "results_20250922_181059_Parallel", "models")
    
    if os.path.exists(parallel_model_path):
        ckpt = parallel_model_path
        print(f"Using parallel scenario model: {ckpt}")
    else:
        # Fallback to models directory
        models_dir = os.path.join(REPO, "models")
        if os.path.exists(models_dir):
            ckpts = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.startswith("checkpoint_")]
            if not ckpts:
                raise SystemExit("No checkpoints found. Train the parallel scenario first.")
            ckpt = max(ckpts, key=os.path.getmtime)
        else:
            raise SystemExit("No models directory found. Train the parallel scenario first.")
    
    # Determine algorithm class
    algo_class = PPO  # Default
    try:
        meta_path = os.path.join(ckpt, ".rllib_metadata")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if "SAC" in (meta.get("checkpoint_type", "") + meta.get("algo_class", "")):
                algo_class = SAC
                print("Detected SAC algorithm")
    except Exception:
        pass
    
    print(f"Using algorithm: {algo_class.__name__}")
    
    # Configuration
    BASELINE_EPISODES = 10  # As per suggested protocol
    TARGETED_EPISODES = 3   # Reduced for faster testing, increase for production
    
    print("="*80)
    print("TARGETED DISTRIBUTION SHIFT TESTING")
    print("="*80)
    print(f"Baseline episodes: {BASELINE_EPISODES}")
    print(f"Episodes per targeted shift: {TARGETED_EPISODES}")
    print(f"Model checkpoint: {ckpt}")
    print("="*80)
    
    # Step 1: Run baseline comparison
    print("\\n[1/3] Running baseline comparison...")
    baseline_csv = run_baseline_comparison(REPO, algo_class, ckpt, BASELINE_EPISODES)
    print(f"Baseline results: {baseline_csv}")
    
    # Step 2: Run targeted shift tests  
    print("\\n[2/3] Running targeted shift tests...")
    targeted_csv = run_targeted_shift_grid(REPO, algo_class, ckpt, "parallel", TARGETED_EPISODES)
    print(f"Targeted shift results: {targeted_csv}")
    
    # Step 3: Comprehensive analysis
    print("\\n[3/3] Performing comprehensive analysis...")
    output_dir = os.path.dirname(targeted_csv)
    comparison_report = analyze_baseline_vs_targeted(baseline_csv, targeted_csv, output_dir)
    
    # Print key findings
    print("\\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    if comparison_report:
        print(f"Total targeted configurations tested: {comparison_report['targeted_summary']['total_configurations']}")
        print(f"Total episodes run: {comparison_report['targeted_summary']['total_episodes']}")
        
        # Print top recommendations
        recommendations = comparison_report.get('recommendations', [])
        if recommendations:
            print("\\nTop Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"{i}. [{rec['priority']}] {rec['finding']}")
                print(f"   → {rec['recommendation']}")
        
        # Print most problematic configurations
        most_conflicts = comparison_report['key_findings'].get('most_conflicts')
        if most_conflicts is not None and not most_conflicts.empty:
            print("\\nMost Conflict-Prone Configurations:")
            for idx, row in most_conflicts.head(3).iterrows():
                print(f"  • {row['test_id']}: {row['num_los_events']} conflicts")
    
    print(f"\\nComplete analysis saved to: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()