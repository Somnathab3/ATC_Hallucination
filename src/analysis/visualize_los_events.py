#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Loss of Separation (LOS) Events from Intra-Shift Testing

This script filters the unified intrashift CSV for episodes with num_los_events > 0
and automatically generates visualizations for each unique model+shift combination.

Usage:
    python visualize_los_events.py
    python visualize_los_events.py --csv path/to/csv --max-visualizations 10
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import subprocess
from collections import defaultdict
import json
from typing import Optional

def load_los_events(csv_path: str) -> pd.DataFrame:
    """Load CSV and filter for LOS events"""
    print(f"📂 Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"   Total rows: {len(df):,}")
    
    # Filter for LOS events
    df_los = df[df['num_los_events'] > 0].copy()
    print(f"   Rows with LOS events (num_los_events > 0): {len(df_los):,}")
    
    if len(df_los) == 0:
        print("   ⚠️  No LOS events found in dataset!")
        return df_los
    
    # Summary statistics
    print(f"\n📊 LOS Event Statistics:")
    print(f"   Total LOS events: {df_los['num_los_events'].sum():.0f}")
    print(f"   Mean LOS events per episode: {df_los['num_los_events'].mean():.2f}")
    print(f"   Max LOS events in episode: {df_los['num_los_events'].max():.0f}")
    print(f"   Min separation (NM): {df_los['min_separation_nm'].min():.2f}")
    
    # Group by scenario
    scenario_counts = df_los.groupby('base_scenario')['num_los_events'].agg(['count', 'sum'])
    print(f"\n📋 LOS Events by Scenario:")
    for scenario, row in scenario_counts.iterrows():
        print(f"   {scenario}: {row['count']} episodes, {row['sum']:.0f} total LOS events")
    
    # Group by shift type
    shift_counts = df_los.groupby('shift_type')['num_los_events'].agg(['count', 'sum'])
    print(f"\n🔀 LOS Events by Shift Type:")
    for shift_type, row in shift_counts.iterrows():
        print(f"   {shift_type}: {row['count']} episodes, {row['sum']:.0f} total LOS events")
    
    return df_los


def extract_visualization_configs(df_los: pd.DataFrame, max_configs: Optional[int] = None) -> list:
    """Extract unique visualization configurations from LOS events
    
    Returns list of dicts with keys: model_path, scenario, test_id, description, seed, los_count
    """
    configs = []
    
    # Group by model + scenario + test_id to get unique shift configurations
    grouped = df_los.groupby(['model_name', 'base_scenario', 'test_id', 'seed'])
    
    print(f"\n🎯 Found {len(grouped)} unique model+scenario+shift combinations with LOS events")
    
    for (model_name, scenario, test_id, seed), group in grouped:
        # Get the first row for metadata
        row = group.iloc[0]
        
        # Construct model path
        model_path = Path("models") / model_name
        
        # Count total LOS events for this configuration
        total_los = group['num_los_events'].sum()
        min_sep = group['min_separation_nm'].min()
        
        config = {
            'model_name': model_name,
            'model_path': str(model_path),
            'scenario': scenario,
            'test_id': test_id,
            'target_agent': row['target_agent'],
            'shift_type': row['shift_type'],
            'shift_value': row['shift_value'],
            'shift_range': row['shift_range'],
            'seed': int(seed),
            'description': row['description'],
            'total_los_events': int(total_los),
            'min_separation_nm': float(min_sep),
            'episode_count': len(group),
        }
        
        configs.append(config)
    
    # Sort by total LOS events (descending) to prioritize worst cases
    configs.sort(key=lambda x: x['total_los_events'], reverse=True)
    
    # Limit if requested
    if max_configs and len(configs) > max_configs:
        print(f"   ⚠️  Limiting to {max_configs} configurations (top by LOS count)")
        configs = configs[:max_configs]
    
    return configs


def validate_model_checkpoint(model_path: Path) -> bool:
    """Check if model checkpoint exists"""
    if not model_path.exists():
        return False
    
    # Check for RLLib checkpoint structure (supports both old and new formats)
    # New format (RLLib 2.x+): flat structure with rllib_checkpoint.json
    # Old format: checkpoint_000XXX/ subdirectories
    
    # Check for new format
    if (model_path / "rllib_checkpoint.json").exists():
        return True
    
    # Check for old format
    checkpoint_dirs = [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith('checkpoint_')]
    if checkpoint_dirs:
        return True
    
    return False


def validate_scenario(scenario: str) -> bool:
    """Check if scenario file exists"""
    scenario_path = Path("scenarios") / f"{scenario}.json"
    return scenario_path.exists()


def run_visualization(config: dict, output_dir: Path, episodes: int = 1, fps: int = 8) -> bool:
    """Run visualization for a specific configuration
    
    Returns True if successful, False otherwise
    """
    model_path = Path(config['model_path'])
    scenario = config['scenario']
    
    print(f"\n{'='*80}")
    print(f"🎬 Visualizing LOS Event Configuration")
    print(f"{'='*80}")
    print(f"Model: {config['model_name']}")
    print(f"Scenario: {scenario}")
    print(f"Test: {config['test_id']}")
    print(f"Target Agent: {config['target_agent']}")
    print(f"Shift: {config['shift_type']} {config['shift_value']} ({config['shift_range']})")
    print(f"Seed: {config['seed']}")
    print(f"Description: {config['description']}")
    print(f"Total LOS Events: {config['total_los_events']}")
    print(f"Min Separation: {config['min_separation_nm']:.2f} NM")
    print(f"Episodes with LOS: {config['episode_count']}")
    print(f"{'='*80}\n")
    
    # Validate prerequisites
    if not validate_model_checkpoint(model_path):
        print(f"❌ Model checkpoint not found or invalid: {model_path}")
        return False
    
    if not validate_scenario(scenario):
        print(f"❌ Scenario file not found: scenarios/{scenario}.json")
        return False
    
    # Create GIF name from configuration
    gif_name = f"los_{config['model_name']}_{config['test_id']}_seed{config['seed']}"
    
    # Create output subdirectory for this configuration
    config_output_dir = output_dir / gif_name
    config_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build shift configuration for environment
    # Format: {"target_agent": {"shift_type": "...", "shift_value": ..., ...}}
    shift_config_json = json.dumps({
        "target_agent": config['target_agent'],
        "shift_type": config['shift_type'],
        "shift_value": float(config['shift_value']),
        "shift_range": config['shift_range'],
        "description": config['description']
    })
    
    # Build visualization command
    cmd = [
        sys.executable,
        "visualize_trained_model.py",
        "--scenario", scenario,
        "--checkpoint", str(model_path.resolve()),
        "--episodes", str(episodes),
        "--fps", str(fps),
        "--record-gifs",
        "--gif-output-dir", str(config_output_dir),
        "--gif-name", gif_name,
        "--log-trajectories",
        "--results-dir", str(config_output_dir / "trajectories"),
        "--max-steps", "1000",  # Allow longer episodes to capture full LOS events
        "--apply-shift", shift_config_json,  # Pass shift configuration
        "--seed", str(config['seed']),  # Use same seed as intrashift test
    ]
    
    print(f"🚀 Running visualization command:")
    print(f"   {' '.join(cmd)}\n")
    
    # Save configuration metadata
    metadata_path = config_output_dir / "los_config.json"
    with open(metadata_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Saved configuration metadata: {metadata_path}")
    
    try:
        # Run visualization with real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            print(f"\n✅ Visualization completed successfully!")
            print(f"   Output: {config_output_dir}")
            return True
        else:
            print(f"\n❌ Visualization failed with return code {return_code}")
            return False
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Visualization failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Loss of Separation (LOS) Events from Intra-Shift Testing",
        epilog="Example:\n"
               "  python visualize_los_events.py\n"
               "  python visualize_los_events.py --csv results/intra_shift_091025/unified_intrashift_episodes.csv --max-visualizations 5",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--csv", type=str,
                       default="results/intra_shift_091025/unified_intrashift_episodes.csv",
                       help="Path to unified intrashift CSV file")
    parser.add_argument("--output-dir", type=str, default="los_visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--max-visualizations", type=int, default=None,
                       help="Maximum number of configurations to visualize (prioritizes by LOS count)")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes to run per configuration")
    parser.add_argument("--fps", type=int, default=8,
                       help="Frames per second for visualization")
    parser.add_argument("--list-only", action="store_true",
                       help="Only list configurations without running visualizations")
    
    args = parser.parse_args()
    
    # Validate CSV path
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir.resolve()}\n")
    
    # Load and filter LOS events
    df_los = load_los_events(str(csv_path))
    
    if len(df_los) == 0:
        print("\n❌ No LOS events found to visualize!")
        return 1
    
    # Extract visualization configurations
    configs = extract_visualization_configs(df_los, args.max_visualizations)
    
    if len(configs) == 0:
        print("\n❌ No valid configurations extracted!")
        return 1
    
    # Display configurations
    print(f"\n{'='*80}")
    print(f"📋 Visualization Configurations (Ranked by LOS Events)")
    print(f"{'='*80}")
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['model_name']} on {config['scenario']}")
        print(f"   Test: {config['test_id']} | Seed: {config['seed']}")
        print(f"   {config['description']}")
        print(f"   💥 LOS Events: {config['total_los_events']} | Min Sep: {config['min_separation_nm']:.2f} NM")
        print(f"   Episodes: {config['episode_count']}")
    
    print(f"\n{'='*80}\n")
    
    # List-only mode
    if args.list_only:
        print("ℹ️  List-only mode. No visualizations will be run.")
        return 0
    
    # Run visualizations
    print(f"🎬 Starting visualization of {len(configs)} configurations...")
    print(f"   Episodes per config: {args.episodes}")
    print(f"   FPS: {args.fps}")
    
    success_count = 0
    fail_count = 0
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"Progress: {i}/{len(configs)}")
        print(f"{'='*80}")
        
        success = run_visualization(config, output_dir, args.episodes, args.fps)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            
            # Ask user whether to continue after failure
            if i < len(configs):
                response = input("\n❓ Continue with remaining configurations? (y/n): ")
                if response.lower() != 'y':
                    print("⏹️  Stopping visualization run.")
                    break
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 Visualization Summary")
    print(f"{'='*80}")
    print(f"Total configurations: {len(configs)}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print(f"📁 Output directory: {output_dir.resolve()}")
    print(f"{'='*80}\n")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    exit(main())
