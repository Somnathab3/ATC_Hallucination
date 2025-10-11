#!/usr/bin/env python3
"""
Run Baseline Visualizations for All Trained Models

This script automates baseline visualization execution for trained models on their
corresponding training scenarios. Results are saved to docs/scenario_plots/Trained Viz/
for documentation purposes.

Usage:
    python run_baseline_visualizations.py --episodes 3 --fps 8
    python run_baseline_visualizations.py --dry-run  # Preview commands without execution
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import time

# Model and scenario mappings
MODEL_CONFIGS = [
        {
        "checkpoint": r"F:\ATC_Hallucination\models\PPO_cross_2x2_20251008_050349",
        "scenario": "cross_2x2",
        "algo": "PPO",
        "description": "Cross 2x2 scenario - two crossing pairs"
    },
    {
        "checkpoint": r"F:\ATC_Hallucination\models\PPO_cross_3p1_20251008_055141",
        "scenario": "cross_3p1",
        "algo": "PPO",
        "description": "Cross 3+1 scenario - three crossing with one parallel"
    },
    {
        "checkpoint": r"F:\ATC_Hallucination\models\PPO_cross_4all_20251008_053820",
        "scenario": "cross_4all",
        "algo": "PPO",
        "description": "Cross 4-all scenario - four-way crossing"
    },
    {
        "checkpoint": r"F:\ATC_Hallucination\models\PPO_merge_2x2_20251008_170616",
        "scenario": "merge_2x2",
        "algo": "PPO",
        "description": "Merge 2x2 scenario - two merging pairs"
    },
    {
        "checkpoint": r"F:\ATC_Hallucination\models\PPO_merge_3p1_20251008_192721",
        "scenario": "merge_3p1",
        "algo": "PPO",
        "description": "Merge 3+1 scenario - three merging with one crossing"
    },
    {
        "checkpoint": r"F:\ATC_Hallucination\models\PPO_merge_4all_20251009_024636",
        "scenario": "merge_4all",
        "algo": "PPO",
        "description": "Merge 4-all scenario - four-way merge"
    },
    {
        "checkpoint": r"F:\ATC_Hallucination\models\PPO_chase_2x2_20251008_015945",
        "scenario": "chase_2x2",
        "algo": "PPO",
        "description": "Chase 2x2 scenario - two chase pairs"
    },
    {
        "checkpoint": r"F:\ATC_Hallucination\models\PPO_chase_3p1_20251008_015936",
        "scenario": "chase_3p1",
        "algo": "PPO",
        "description": "Chase 3+1 scenario - three chase with one crossing"
    },
    {
        "checkpoint": r"F:\ATC_Hallucination\models\PPO_chase_4all_20251008_015933",
        "scenario": "chase_4all",
        "algo": "PPO",
        "description": "Chase 4-all scenario - four-way chase"
    },


]

# Output directory for visualizations
OUTPUT_DIR = r"F:\ATC_Hallucination\docs\scenario_plots\Trained_Viz3"

# Intershift output directory
INTERSHIFT_OUTPUT_DIR = r"F:\ATC_Hallucination\docs\scenario_plots\inter_shift3"

# Trajectory logging directory
TRAJ_LOG_DIR = r"F:\ATC_Hallucination\vis_results2"

# Intershift trajectory logging directory
INTERSHIFT_TRAJ_LOG_DIR = r"F:\ATC_Hallucination\vis_results_intershift2"


class BaselineVisualizationRunner:
    """Automated baseline visualization execution for trained models"""
    
    def __init__(self, episodes: int = 1, fps: int = 8, dry_run: bool = False, 
                 record_gifs: bool = True, log_trajectories: bool = True, 
                 intershift: bool = False):
        self.episodes = episodes
        self.fps = fps
        self.dry_run = dry_run
        self.record_gifs = record_gifs
        self.log_trajectories = log_trajectories
        self.intershift = intershift
        
        # Set directories based on mode
        if intershift:
            self.output_dir = Path(INTERSHIFT_OUTPUT_DIR)
            self.traj_log_dir = Path(INTERSHIFT_TRAJ_LOG_DIR)
        else:
            self.output_dir = Path(OUTPUT_DIR)
            self.traj_log_dir = Path(TRAJ_LOG_DIR)
        
        # Create output directories
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.traj_log_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
    
    def validate_checkpoint(self, checkpoint_path: str) -> Tuple[bool, str]:
        """Validate checkpoint exists and is accessible"""
        path = Path(checkpoint_path)
        
        if not path.exists():
            return False, f"Checkpoint directory does not exist: {checkpoint_path}"
        
        # Check for checkpoint files (RLlib structure - newer format uses rllib_checkpoint.json)
        has_rllib_checkpoint = (path / "rllib_checkpoint.json").exists()
        has_algorithm_state = (path / "algorithm_state.pkl").exists()
        has_policies = (path / "policies").exists()
        
        # Also check for older tune_metadata format
        has_tune_metadata = any(path.glob("*.tune_metadata"))
        
        if not (has_rllib_checkpoint or has_tune_metadata):
            return False, f"No RLlib checkpoint files found in {checkpoint_path}"
        
        if not has_algorithm_state:
            return False, f"Missing algorithm_state.pkl in {checkpoint_path}"
        
        if not has_policies:
            return False, f"Missing policies directory in {checkpoint_path}"
        
        return True, "Valid checkpoint"
    
    def validate_scenario(self, scenario_name: str) -> Tuple[bool, str]:
        """Validate scenario file exists"""
        scenario_path = Path(f"scenarios/{scenario_name}.json")
        
        if not scenario_path.exists():
            return False, f"Scenario file does not exist: {scenario_path}"
        
        return True, "Valid scenario"
    
    def generate_intershift_configs(self) -> List[Dict]:
        """Generate all intershift configurations (each model on all other scenarios)"""
        # Exclude generic model (last entry)
        trained_models = MODEL_CONFIGS[:-1]
        
        # All available scenarios
        all_scenarios = [
            "cross_2x2", "cross_3p1", "cross_4all",
            "merge_2x2", "merge_3p1", "merge_4all", 
            "chase_2x2", "chase_3p1", "chase_4all"
        ]
        
        intershift_configs = []
        
        for model_config in trained_models:
            trained_scenario = model_config["scenario"]
            
            # Test this model on all scenarios except its training scenario
            for test_scenario in all_scenarios:
                if test_scenario != trained_scenario:
                    config = {
                        "checkpoint": model_config["checkpoint"],
                        "scenario": test_scenario,  # Test scenario
                        "trained_scenario": trained_scenario,  # Training scenario
                        "algo": model_config["algo"],
                        "description": f"Model trained on {trained_scenario} tested on {test_scenario}"
                    }
                    intershift_configs.append(config)
        
        return intershift_configs
    
    def build_command(self, config: Dict) -> List[str]:
        """Build visualization command for a model configuration"""
        # Determine output GIF name based on scenario and mode
        scenario_name = config["scenario"]
        trained_scenario = config.get("trained_scenario", scenario_name)
        is_generic = "generic" in config["checkpoint"].lower()
        
        if self.intershift:
            # For intershift: model_trained_scenario__on__test_scenario
            gif_name = f"{trained_scenario}__on__{scenario_name}"
        elif is_generic:
            gif_name = "generic_baseline"
        else:
            gif_name = f"{scenario_name}_baseline"
        
        cmd = [
            "python",
            "./src/analysis/visualize_trained_model.py",
            "--scenario", scenario_name,
            "--checkpoint", config["checkpoint"],
            "--algo", config["algo"],
            "--episodes", str(self.episodes),
            "--fps", str(self.fps),
            "--gif-output-dir", str(self.output_dir),
            "--results-dir", str(self.traj_log_dir),
            "--gif-name", gif_name,  # Custom GIF naming
        ]
        
        if self.record_gifs:
            cmd.append("--record-gifs")
        else:
            cmd.append("--no-record-gifs")
        
        if self.log_trajectories:
            cmd.append("--log-trajectories")
        else:
            cmd.append("--no-log-trajectories")
        
        return cmd
    
    def run_visualization(self, config: Dict) -> Dict:
        """Run visualization for a single model configuration"""
        scenario_name = config["scenario"]
        trained_scenario = config.get("trained_scenario", scenario_name)
        
        if self.intershift:
            title = f"Model {trained_scenario} → Scenario {scenario_name}"
        else:
            title = f"Baseline: {scenario_name}"
        
        print(f"\n{'='*80}")
        print(f"🎬 Running Visualization: {title}")
        print(f"📁 Checkpoint: {config['checkpoint']}")
        print(f"📋 Description: {config['description']}")
        print(f"{'='*80}\n")        # Validate checkpoint
        valid_checkpoint, msg = self.validate_checkpoint(config["checkpoint"])
        if not valid_checkpoint:
            print(f"❌ {msg}")
            return {
                "config": config,
                "success": False,
                "error": msg,
                "duration_s": 0
            }
        
        # Validate scenario
        valid_scenario, msg = self.validate_scenario(config["scenario"])
        if not valid_scenario:
            print(f"❌ {msg}")
            return {
                "config": config,
                "success": False,
                "error": msg,
                "duration_s": 0
            }
        
        # Build command
        cmd = self.build_command(config)
        
        if self.dry_run:
            print("🔍 DRY RUN - Command that would be executed:")
            print(" ".join(cmd))
            return {
                "config": config,
                "success": True,
                "error": None,
                "duration_s": 0,
                "dry_run": True
            }
        
        # Execute visualization
        print(f"▶️  Executing: {' '.join(cmd)}\n")
        
        start_time = time.time()
        try:
            # Use UTF-8 encoding for subprocess to handle all characters
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace undecodable characters
                timeout=600  # 10 minute timeout per visualization
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Visualization completed successfully in {duration:.1f}s")
                
                # Count generated GIF files
                if self.record_gifs:
                    gif_files = list(self.output_dir.glob(f"episode_*{config['scenario']}*.gif"))
                    print(f"📹 Generated {len(gif_files)} GIF files")
                
                # Count trajectory CSV files
                if self.log_trajectories:
                    traj_files = list(self.traj_log_dir.glob(f"*{config['scenario']}*.csv"))
                    print(f"📊 Generated {len(traj_files)} trajectory CSV files")
                
                return {
                    "config": config,
                    "success": True,
                    "error": None,
                    "duration_s": duration,
                    "stdout": result.stdout[-500:] if result.stdout else "",  # Last 500 chars
                    "gif_count": len(gif_files) if self.record_gifs else 0,
                    "csv_count": len(traj_files) if self.log_trajectories else 0
                }
            else:
                print(f"❌ Visualization failed with return code {result.returncode}")
                print(f"Error output:\n{result.stderr[-1000:]}")  # Last 1000 chars
                
                return {
                    "config": config,
                    "success": False,
                    "error": f"Process exited with code {result.returncode}",
                    "duration_s": duration,
                    "stderr": result.stderr[-1000:] if result.stderr else ""
                }
                
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏱️ Visualization timed out after {duration:.1f}s")
            return {
                "config": config,
                "success": False,
                "error": "Process timeout (10 minutes exceeded)",
                "duration_s": duration
            }
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ Unexpected error: {e}")
            return {
                "config": config,
                "success": False,
                "error": str(e),
                "duration_s": duration
            }
    
    def run_all(self) -> List[Dict]:
        """Run visualizations for all model configurations"""
        if self.intershift:
            configs = self.generate_intershift_configs()
            mode_name = "Intershift Visualization"
            mode_desc = "Testing each model on all scenarios except its training scenario"
        else:
            configs = MODEL_CONFIGS
            mode_name = "Baseline Visualization"
            mode_desc = "Testing each model on its training scenario"
        
        print(f"\n{'='*80}")
        print(f"🚀 {mode_name} Batch Runner")
        print(f"{'='*80}")
        print(f"📊 Total runs: {len(configs)}")
        print(f"🎞️  Episodes per run: {self.episodes}")
        print(f"⚡ FPS: {self.fps}")
        print(f"📹 GIF recording: {'Enabled' if self.record_gifs else 'Disabled'}")
        print(f"📊 Trajectory logging: {'Enabled' if self.log_trajectories else 'Disabled'}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📁 Trajectory directory: {self.traj_log_dir}")
        print(f"📋 Mode: {mode_desc}")
        
        if self.dry_run:
            print(f"\n🔍 DRY RUN MODE - No actual execution will occur")
        
        print(f"{'='*80}\n")
        
        # Run visualizations
        for idx, config in enumerate(configs, 1):
            print(f"\n📍 Progress: {idx}/{len(configs)}")
            result = self.run_visualization(config)
            self.results.append(result)
            
            # Brief pause between runs
            if not self.dry_run and idx < len(configs):
                print("\n⏸️  Pausing 5 seconds before next visualization...")
                time.sleep(5)
        
        # Summary
        self.print_summary()
        
        # Save results
        if not self.dry_run:
            self.save_results()
        
        return self.results
    
    def print_summary(self):
        """Print execution summary"""
        print(f"\n{'='*80}")
        print(f"📊 EXECUTION SUMMARY")
        print(f"{'='*80}\n")
        
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        print(f"✅ Successful: {len(successful)}/{len(self.results)}")
        print(f"❌ Failed: {len(failed)}/{len(self.results)}")
        
        if successful:
            total_duration = sum(r["duration_s"] for r in successful)
            avg_duration = total_duration / len(successful)
            print(f"⏱️  Average duration: {avg_duration:.1f}s per visualization")
            print(f"⏱️  Total time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
        
        if failed:
            print(f"\n❌ Failed visualizations:")
            for result in failed:
                print(f"   • {result['config']['scenario']}: {result['error']}")
        
        if self.record_gifs and not self.dry_run:
            total_gifs = sum(r.get("gif_count", 0) for r in successful)
            print(f"\n📹 Total GIF files generated: {total_gifs}")
        
        if self.log_trajectories and not self.dry_run:
            total_csvs = sum(r.get("csv_count", 0) for r in successful)
            print(f"📊 Total trajectory CSV files: {total_csvs}")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self):
        """Save execution results to JSON"""
        mode_suffix = "_intershift" if self.intershift else "_baseline"
        results_file = self.output_dir / f"visualization_results{mode_suffix}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "execution_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "intershift" if self.intershift else "baseline",
                "configuration": {
                    "episodes": self.episodes,
                    "fps": self.fps,
                    "record_gifs": self.record_gifs,
                    "log_trajectories": self.log_trajectories
                },
                "results": self.results
            }, f, indent=2)
        
        print(f"💾 Execution results saved to: {results_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run baseline and intershift visualizations for trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline visualizations (default)
  python run_baseline_visualizations.py
  
  # Run intershift visualizations (9 models × 8 scenarios = 72 runs)
  python run_baseline_visualizations.py --intershift
  
  # Run with more episodes and lower FPS
  python run_baseline_visualizations.py --episodes 3 --fps 6
  
  # Preview commands without execution
  python run_baseline_visualizations.py --dry-run
  
  # Disable GIF recording (faster execution)
  python run_baseline_visualizations.py --no-record-gifs --intershift
  
  # Run intershift with trajectory logging only
  python run_baseline_visualizations.py --intershift --no-record-gifs
        """
    )
    
    parser.add_argument("--episodes", "-e", type=int, default=1,
                       help="Number of episodes per visualization (default: 1, deterministic with explore=False)")
    parser.add_argument("--fps", type=int, default=8,
                       help="Frames per second for visualization (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview commands without executing")
    parser.add_argument("--intershift", action="store_true",
                       help="Run intershift tests: each model on all scenarios except its training scenario (72 runs total)")
    parser.add_argument("--no-record-gifs", action="store_false", dest="record_gifs",
                       help="Disable GIF recording (faster execution)")
    parser.add_argument("--no-log-trajectories", action="store_false", dest="log_trajectories",
                       help="Disable trajectory CSV logging")
    
    args = parser.parse_args()
    
    # Create runner
    runner = BaselineVisualizationRunner(
        episodes=args.episodes,
        fps=args.fps,
        dry_run=args.dry_run,
        record_gifs=args.record_gifs,
        log_trajectories=args.log_trajectories,
        intershift=args.intershift
    )
    
    # Run all visualizations
    results = runner.run_all()
    
    # Exit with appropriate code
    failed = [r for r in results if not r["success"]]
    return 1 if failed and not args.dry_run else 0


if __name__ == "__main__":
    sys.exit(main())
