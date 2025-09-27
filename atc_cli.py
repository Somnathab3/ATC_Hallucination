#!/usr/bin/env python3
"""
ATC Hallucination Project - Unified CLI

A comprehensive command-line interface for the ATC Hallucination project that provides:
- Scenario generation with custom parameters
- Model training with multiple algorithms
- Distribution shift testing (unison and targeted)
- Hallucination detection and analysis  
- Visualization and reporting
- Model evaluation and checkpoint management

Usage examples:
    # Generate scenarios
    python atc_cli.py generate-scenarios --all
    python atc_cli.py generate-scenarios --scenario head_on --params "approach_nm=20"
    
    # Train models
    python atc_cli.py train --scenario head_on --algo PPO --timesteps 100000
    python atc_cli.py train --scenario all --timesteps 50000 --checkpoint-every 10000
    python atc_cli.py train --algo PPO --timesteps 2000000 --scenario canonical_crossing --log-trajectories
    
    # Run shift testing
    python atc_cli.py test-shifts --checkpoint latest --scenario parallel --episodes 5
    python atc_cli.py test-shifts --targeted --episodes 3 --viz
    
    # Analyze and visualize
    python atc_cli.py analyze --results-dir results_PPO_head_on_20250923_190203
    python atc_cli.py visualize --trajectory traj_ep_0001.csv
    
    # Full pipeline
    python atc_cli.py full-pipeline --scenario head_on --train-timesteps 50000 --test-episodes 3
"""

import os
import sys
import argparse
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ATC_CLI")

class ATCController:
    """Main controller for the ATC Hallucination project CLI."""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.scenarios_dir = self.repo_root / "scenarios"
        self.results_dir = self.repo_root / "results"
        self.models_dir = self.repo_root / "models"
        
        # Ensure directories exist
        for dir_path in [self.scenarios_dir, self.results_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def generate_scenarios(self, scenario_types: Optional[List[str]] = None, scenario_params: Optional[Dict[str, Any]] = None) -> List[str]:
        """Generate scenario files using the scenario generator."""
        try:
            from src.scenarios.scenario_generator import (
                make_head_on, make_t_formation, make_parallel, make_converging, make_canonical_crossing
            )
            
            scenario_funcs = {
                "head_on": make_head_on,
                "t_formation": make_t_formation,
                "parallel": make_parallel,
                "converging": make_converging,
                "canonical_crossing": make_canonical_crossing
            }
            
            if scenario_types is None or "all" in scenario_types:
                scenario_types = list(scenario_funcs.keys())
            
            generated_paths = []
            
            for scenario_name in scenario_types:
                if scenario_name not in scenario_funcs:
                    logger.warning(f"Unknown scenario type: {scenario_name}")
                    continue
                
                logger.info(f"Generating scenario: {scenario_name}")
                
                # Apply custom parameters if provided
                params = scenario_params.get(scenario_name, {}) if scenario_params else {}
                
                try:
                    path = scenario_funcs[scenario_name](**params)
                    generated_paths.append(path)
                    logger.info(f"Generated: {path}")
                except Exception as e:
                    logger.error(f"Failed to generate {scenario_name}: {e}")
            
            return generated_paths
            
        except ImportError as e:
            logger.error(f"Failed to import scenario generator: {e}")
            return []
    
    def train_model(self, scenario_names: Union[str, List[str]], algo: str = "PPO", timesteps: int = 100000, 
                   checkpoint_every: int = 10000, log_trajectories: bool = False, **kwargs) -> Optional[str]:
        """Train a model on the specified scenario(s) and algorithm(s)."""
        try:
            from src.training.train_frozen_scenario import train_frozen
            
            # Handle scenario names input
            if isinstance(scenario_names, str):
                scenario_names = [scenario_names]
            
            # Handle 'all' scenarios
            if any(s.lower() == "all" for s in scenario_names):
                scenarios = self.list_scenarios()
                if not scenarios:
                    logger.warning("No scenarios found, generating all...")
                    self.generate_scenarios(["all"])
                    scenarios = self.list_scenarios()
            else:
                scenarios = scenario_names
            
            # Handle 'all' algorithms
            if algo.lower() == "all":
                algorithms = ["PPO", "SAC", "IMPALA", "CQL", "APPO"]
            else:
                algorithms = [algo]
            
            final_checkpoints = []
            
            # Train each combination
            for current_algo in algorithms:
                for current_scenario in scenarios:
                    logger.info(f"Training {current_algo} on scenario '{current_scenario}' for {timesteps:,} timesteps")
                    
                    # Ensure scenario exists
                    scenario_path = self.scenarios_dir / f"{current_scenario}.json"
                    if not scenario_path.exists():
                        logger.warning(f"Scenario {current_scenario} not found, generating...")
                        self.generate_scenarios([current_scenario])
                    
                    checkpoint_path = train_frozen(
                        repo_root=str(self.repo_root),
                        algo=current_algo,
                        scenario_name=current_scenario,
                        timesteps_total=timesteps,
                        checkpoint_every=checkpoint_every,
                        log_trajectories=log_trajectories,
                        **kwargs
                    )
                    
                    if checkpoint_path:
                        final_checkpoints.append(checkpoint_path)
                        logger.info(f"Training completed for {current_algo} on {current_scenario}. Checkpoint: {checkpoint_path}")
                    else:
                        logger.error(f"Training failed for {current_algo} on {current_scenario}")
            
            if final_checkpoints:
                logger.info(f"All training completed. {len(final_checkpoints)} models trained.")
                return final_checkpoints[-1]  # Return last checkpoint for compatibility
            else:
                logger.error("All training attempts failed.")
                return None
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return None
    
    def run_shift_testing(self, checkpoint_path: Optional[str] = None, scenario_name: str = "parallel", 
                         episodes: int = 3, targeted: bool = True, generate_viz: bool = False,
                         algo: str = "PPO") -> Optional[str]:
        """Run distribution shift testing (unison or targeted)."""
        try:
            # Clean up any existing Ray processes first
            try:
                import ray
                if ray.is_initialized():
                    logger.info("Shutting down existing Ray session...")
                    ray.shutdown()
            except Exception as e:
                logger.warning(f"Ray cleanup warning: {e}")
                
            if checkpoint_path is None:
                checkpoint_path = self._auto_detect_checkpoint()
                if checkpoint_path is None:
                    logger.error("No checkpoint found. Please train a model first.")
                    return None
            
            if targeted:
                from src.testing.targeted_shift_tester import run_targeted_shift_grid
                from ray.rllib.algorithms.ppo import PPO
                from ray.rllib.algorithms.sac import SAC
                
                algo_class = SAC if algo.upper() == "SAC" else PPO
                
                logger.info(f"Running targeted shift testing with {episodes} episodes per shift")
                
                result_path = run_targeted_shift_grid(
                    repo_root=str(self.repo_root),
                    algo_class=algo_class,
                    checkpoint_path=checkpoint_path,
                    scenario_name=scenario_name,
                    episodes_per_shift=episodes,
                    generate_viz=generate_viz
                )
                
                logger.info(f"Targeted shift testing completed: {result_path}")
                return result_path
            else:
                # Implement basic unison shift testing
                logger.info(f"Running unison shift testing with {episodes} episodes per shift")
                return self._run_unison_shift_testing(checkpoint_path, scenario_name, episodes, generate_viz, algo)
                
        except Exception as e:
            logger.error(f"Shift testing failed: {e}")
            return None
    
    def _run_unison_shift_testing(self, checkpoint_path: str, scenario_name: str, 
                                 episodes: int, generate_viz: bool, algo: str) -> Optional[str]:
        """Run unison shift testing (simplified implementation)."""
        try:
            # For now, fall back to targeted shift testing with a warning
            logger.warning("Unison shift testing not fully implemented. Running targeted shifts instead.")
            
            from src.testing.targeted_shift_tester import run_targeted_shift_grid
            from ray.rllib.algorithms.ppo import PPO
            from ray.rllib.algorithms.sac import SAC
            
            algo_class = SAC if algo.upper() == "SAC" else PPO
            
            result_path = run_targeted_shift_grid(
                repo_root=str(self.repo_root),
                algo_class=algo_class,
                checkpoint_path=checkpoint_path,
                scenario_name=scenario_name,
                episodes_per_shift=episodes,
                generate_viz=generate_viz
            )
            
            return result_path
            
        except Exception as e:
            logger.error(f"Unison shift testing failed: {e}")
            return None
    
    def analyze_results(self, results_dir: Optional[str] = None, hallucination_analysis: bool = True) -> Dict[str, Any]:
        """Analyze training or testing results."""
        try:
            results_path: Path
            if results_dir is None:
                # Find latest results directory
                results_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]
                if not results_dirs:
                    logger.error("No results directories found")
                    return {}
                results_path = max(results_dirs, key=lambda x: x.stat().st_mtime)
            else:
                results_path = Path(results_dir)
            
            logger.info(f"Analyzing results in: {results_path}")
            
            analysis_results = {
                "results_dir": str(results_path),
                "timestamp": datetime.now().isoformat()
            }
            
            # Basic trajectory analysis
            traj_files = list(results_path.glob("traj_ep_*.csv")) + list(results_path.glob("**/traj_ep_*.csv")) + list(results_path.glob("traj_*.csv")) + list(results_path.glob("**/traj_*.csv"))
            if traj_files:
                logger.info(f"Found {len(traj_files)} trajectory files")
                analysis_results["trajectory_files"] = len(traj_files)
                
                # Analyze latest trajectory
                latest_traj = max(traj_files, key=lambda x: x.stat().st_mtime)
                analysis_results.update(self._analyze_trajectory(latest_traj))
            
            # Hallucination analysis if enabled
            if hallucination_analysis and traj_files:
                analysis_results["hallucination"] = self._analyze_hallucinations(traj_files[0])
            
            # Training progress analysis
            progress_file = results_path / "training_progress.csv"
            if progress_file.exists():
                analysis_results["training"] = self._analyze_training_progress(progress_file)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {}
    
    def visualize_data(self, trajectory_path: Optional[str] = None, results_dir: Optional[str] = None, 
                      output_dir: Optional[str] = None) -> List[str]:
        """Generate visualizations for trajectories and results."""
        try:
            output_path: Path
            if output_dir is None:
                output_path = self.repo_root / "vis_temp"
            else:
                output_path = Path(output_dir)
            
            output_path.mkdir(exist_ok=True)
            
            generated_files = []
            
            if trajectory_path:
                logger.info(f"Visualizing trajectory: {trajectory_path}")
                # TODO: Implement trajectory visualization
                # from src.analysis.viz_hooks import make_episode_visuals
                # generated_files.extend(make_episode_visuals(...))
            
            if results_dir:
                logger.info(f"Visualizing results: {results_dir}")
                # TODO: Implement results visualization
                # from src.analysis.viz_hooks import make_run_visuals
                # generated_files.extend(make_run_visuals(...))
            
            logger.info(f"Generated {len(generated_files)} visualization files")
            return generated_files
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return []
    
    def full_pipeline(self, scenario_name: str, algo: str = "PPO", 
                     train_timesteps: int = 50000, test_episodes: int = 3,
                     targeted_shifts: bool = True, generate_viz: bool = True,
                     log_trajectories: bool = False) -> Dict[str, Any]:
        """Run the complete pipeline: scenario generation → training → testing → analysis."""
        logger.info("🚀 Starting full pipeline execution")
        
        pipeline_results = {
            "start_time": datetime.now().isoformat(),
            "scenario": scenario_name,
            "algorithm": algo,
            "steps": {}
        }
        
        try:
            # Step 1: Generate scenario
            logger.info("📝 Step 1: Generating scenarios")
            scenarios = self.generate_scenarios([scenario_name])
            pipeline_results["steps"]["scenario_generation"] = {
                "success": len(scenarios) > 0,
                "generated": scenarios
            }
            
            if not scenarios:
                raise Exception("Scenario generation failed")
            
            # Step 2: Train model
            logger.info("🎯 Step 2: Training model")
            checkpoint = self.train_model(
                scenario_names=[scenario_name],
                algo=algo,
                timesteps=train_timesteps,
                checkpoint_every=max(1000, train_timesteps // 10),
                log_trajectories=log_trajectories
            )
            pipeline_results["steps"]["training"] = {
                "success": checkpoint is not None,
                "checkpoint": checkpoint
            }
            
            if not checkpoint:
                raise Exception("Training failed")
            
            # Step 3: Shift testing
            logger.info("🧪 Step 3: Running shift testing")
            test_results = self.run_shift_testing(
                checkpoint_path=checkpoint,
                scenario_name=scenario_name,
                episodes=test_episodes,
                targeted=targeted_shifts,
                generate_viz=generate_viz,
                algo=algo
            )
            pipeline_results["steps"]["shift_testing"] = {
                "success": test_results is not None,
                "results": test_results
            }
            
            # Step 4: Analysis
            logger.info("📊 Step 4: Analyzing results")
            analysis = self.analyze_results()
            pipeline_results["steps"]["analysis"] = {
                "success": bool(analysis),
                "results": analysis
            }
            
            # Step 5: Visualization (optional)
            if generate_viz:
                logger.info("📈 Step 5: Generating visualizations")
                viz_files = self.visualize_data()
                pipeline_results["steps"]["visualization"] = {
                    "success": len(viz_files) > 0,
                    "files": viz_files
                }
            
            pipeline_results["success"] = True
            pipeline_results["end_time"] = datetime.now().isoformat()
            
            logger.info("✅ Full pipeline completed successfully")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            pipeline_results["success"] = False
            pipeline_results["error"] = str(e)
            pipeline_results["end_time"] = datetime.now().isoformat()
            return pipeline_results
    
    def list_scenarios(self) -> List[str]:
        """List available scenario files."""
        scenario_files = list(self.scenarios_dir.glob("*.json"))
        return [f.stem for f in scenario_files]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available model checkpoints."""
        checkpoints = []
        
        # Check models directory
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                checkpoints.append({
                    "path": str(model_dir),
                    "name": model_dir.name,
                    "modified": datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat()
                })
        
        # Check results directories for model subdirectories
        for results_dir in self.results_dir.iterdir():
            if results_dir.is_dir():
                models_subdir = results_dir / "models"
                if models_subdir.exists():
                    for model_dir in models_subdir.iterdir():
                        if model_dir.is_dir():
                            checkpoints.append({
                                "path": str(model_dir),
                                "name": f"{results_dir.name}/{model_dir.name}",
                                "modified": datetime.fromtimestamp(model_dir.stat().st_mtime).isoformat()
                            })
        
        return sorted(checkpoints, key=lambda x: x["modified"], reverse=True)
    
    def _auto_detect_checkpoint(self) -> Optional[str]:
        """Auto-detect the most recent checkpoint."""
        checkpoints = self.list_checkpoints()
        return checkpoints[0]["path"] if checkpoints else None
    
    def _analyze_trajectory(self, traj_path: Path) -> Dict[str, Any]:
        """Analyze a single trajectory file."""
        try:
            import pandas as pd
            
            df = pd.read_csv(traj_path)
            
            if df.empty:
                return {"error": "Empty trajectory file"}
            
            analysis = {
                "file": str(traj_path),
                "episodes": df["episode_id"].nunique() if "episode_id" in df.columns else 1,
                "steps": len(df),
                "agents": df["agent_id"].nunique() if "agent_id" in df.columns else 0,
            }
            
            # Safety metrics
            if "conflict_flag" in df.columns:
                analysis["conflicts"] = int(df["conflict_flag"].sum())
                analysis["conflict_rate"] = float(df["conflict_flag"].mean())
            
            if "min_separation_nm" in df.columns:
                analysis["min_separation_nm"] = float(df["min_separation_nm"].min())
                analysis["avg_separation_nm"] = float(df["min_separation_nm"].mean())
            
            if "waypoint_reached" in df.columns:
                analysis["waypoint_completion_rate"] = float(df["waypoint_reached"].mean())
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_hallucinations(self, traj_path: Path) -> Dict[str, Any]:
        """Run hallucination analysis on trajectory data."""
        try:
            from src.analysis.hallucination_detector_enhanced import HallucinationDetector
            import pandas as pd
            import numpy as np
            
            df = pd.read_csv(traj_path)
            
            # Convert to trajectory format expected by detector
            positions = []
            actions = []
            agents_data = {}
            
            # Get unique agents
            agent_ids = df["agent_id"].unique()
            for aid in agent_ids:
                agents_data[aid] = {"headings": [], "speeds": []}
            
            # Process by step
            for step in sorted(df["step_idx"].unique()):
                step_data = df[df["step_idx"] == step]
                
                pos_dict = {}
                action_dict = {}
                
                for _, row in step_data.iterrows():
                    aid = row["agent_id"]
                    pos_dict[aid] = (row["lat_deg"], row["lon_deg"])
                    action_dict[aid] = np.array([row["action_hdg_delta_deg"], row["action_spd_delta_kt"]])
                    
                    agents_data[aid]["headings"].append(row["hdg_deg"])
                    agents_data[aid]["speeds"].append(row["tas_kt"])
                
                positions.append(pos_dict)
                actions.append(action_dict)
            
            trajectory = {
                "positions": positions,
                "actions": actions,
                "agents": agents_data,
                "timestamps": list(range(len(positions)))
            }
            
            # Run hallucination detection
            detector = HallucinationDetector()
            results = detector.compute(trajectory, sep_nm=5.0, return_series=True)
            
            # Calculate derived metrics from confusion matrix
            tp = results["tp"]
            fp = results["fp"]
            fn = results["fn"]
            tn = results["tn"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            return {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score),
                "accuracy": float(accuracy),
                "resolution_efficiency": float(results.get("resolution_efficiency", 0.0)),
                "unwanted_interventions": int(results.get("unwanted_interventions", 0))
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_training_progress(self, progress_path: Path) -> Dict[str, Any]:
        """Analyze training progress from CSV."""
        try:
            import pandas as pd
            
            df = pd.read_csv(progress_path)
            
            if df.empty:
                return {"error": "Empty progress file"}
            
            return {
                "total_iterations": len(df),
                "final_reward": float(df["reward_mean"].iloc[-1]) if "reward_mean" in df.columns else None,
                "best_reward": float(df["reward_mean"].max()) if "reward_mean" in df.columns else None,
                "total_steps": int(df["steps_sampled"].iloc[-1]) if "steps_sampled" in df.columns else None,
                "zero_conflict_streak": int(df["zero_conflict_streak"].iloc[-1]) if "zero_conflict_streak" in df.columns else None
            }
            
        except Exception as e:
            return {"error": str(e)}


def parse_scenario_params(params_str: str) -> Dict[str, Any]:
    """Parse scenario parameters from string format 'param1=value1,param2=value2'."""
    if not params_str:
        return {}
    
    params = {}
    for param in params_str.split(','):
        if '=' not in param:
            continue
        key, value = param.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Try to convert to appropriate type
        if value.lower() in ('true', 'false'):
            params[key] = value.lower() == 'true'
        elif value.replace('.', '').replace('-', '').isdigit():
            params[key] = float(value) if '.' in value else int(value)
        else:
            params[key] = value
    
    return params


def main():
    parser = argparse.ArgumentParser(
        description="ATC Hallucination Project - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Global options
    parser.add_argument("--repo-root", type=str, default=str(project_root),
                       help="Project repository root directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Scenario generation
    gen_parser = subparsers.add_parser("generate-scenarios", help="Generate scenario files")
    gen_parser.add_argument("--scenario", "-s", action="append",
                           help="Scenario type to generate (can be used multiple times)")
    gen_parser.add_argument("--all", action="store_true",
                           help="Generate all scenario types")
    gen_parser.add_argument("--params", type=str,
                           help="Scenario parameters in format 'param1=value1,param2=value2'")
    
    # Training
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--scenario", "-s", action="append",
                             help="Scenario to train on (can be used multiple times, or use 'all' for all scenarios)")
    train_parser.add_argument("--algo", "-a", type=str, default="PPO",
                             help="Algorithm to use (PPO, SAC, IMPALA, CQL, APPO, or 'all' for all algorithms)")
    train_parser.add_argument("--timesteps", "-t", type=int, default=100000,
                             help="Number of training timesteps")
    train_parser.add_argument("--checkpoint-every", "-c", type=int,
                             help="Checkpoint frequency (default: timesteps/10)")
    train_parser.add_argument("--gpu", action="store_true",
                             help="Enable GPU training (auto-detect if available)")
    train_parser.add_argument("--no-gpu", action="store_true",
                             help="Force CPU-only training")
    train_parser.add_argument("--log-trajectories", action="store_true",
                             help="Enable detailed trajectory logging (default: False for faster training)")
    
    # Shift testing
    test_parser = subparsers.add_parser("test-shifts", help="Run distribution shift testing")
    test_parser.add_argument("--checkpoint", type=str,
                            help="Checkpoint path (default: auto-detect latest)")
    test_parser.add_argument("--scenario", "-s", type=str, default="parallel",
                            help="Scenario for testing")
    test_parser.add_argument("--episodes", "-e", type=int, default=3,
                            help="Episodes per shift configuration")
    test_parser.add_argument("--targeted", action="store_true", default=True,
                            help="Use targeted shifts (vs unison). Default: True")
    test_parser.add_argument("--algo", "-a", choices=["PPO", "SAC"], default="PPO",
                            help="Algorithm type")
    test_parser.add_argument("--viz", action="store_true",
                            help="Generate visualizations")
    
    # Analysis
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("--results-dir", type=str,
                               help="Results directory (default: latest)")
    analyze_parser.add_argument("--no-hallucination", action="store_true",
                               help="Skip hallucination analysis")
    
    # Visualization
    viz_parser = subparsers.add_parser("visualize", help="Generate visualizations")
    viz_parser.add_argument("--trajectory", type=str,
                           help="Trajectory file to visualize")
    viz_parser.add_argument("--results-dir", type=str,
                           help="Results directory to visualize")
    viz_parser.add_argument("--output-dir", type=str,
                           help="Output directory for visualizations")
    
    # Full pipeline
    pipeline_parser = subparsers.add_parser("full-pipeline", help="Run complete pipeline")
    pipeline_parser.add_argument("--scenario", "-s", type=str, default="head_on",
                                 help="Scenario for pipeline")
    pipeline_parser.add_argument("--algo", "-a", choices=["PPO", "SAC"], default="PPO",
                                 help="Algorithm to use")
    pipeline_parser.add_argument("--train-timesteps", "-t", type=int, default=50000,
                                 help="Training timesteps")
    pipeline_parser.add_argument("--test-episodes", "-e", type=int, default=3,
                                 help="Testing episodes")
    pipeline_parser.add_argument("--no-targeted", action="store_true",
                                 help="Use unison shifts instead of targeted")
    pipeline_parser.add_argument("--no-viz", action="store_true",
                                 help="Skip visualization generation")
    pipeline_parser.add_argument("--log-trajectories", action="store_true",
                                 help="Enable detailed trajectory logging during training")
    
    # List commands
    list_parser = subparsers.add_parser("list", help="List available resources")
    list_parser.add_argument("resource", choices=["scenarios", "checkpoints", "results"],
                            help="Resource type to list")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize controller
    controller = ATCController(args.repo_root)
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "generate-scenarios":
            scenario_types = args.scenario or []
            if args.all:
                scenario_types = ["all"]
            
            scenario_params = {}
            if args.params:
                parsed_params = parse_scenario_params(args.params)
                # Apply same params to all scenarios for now
                for scenario in ["head_on", "t_formation", "parallel", "converging"]:
                    scenario_params[scenario] = parsed_params
            
            paths = controller.generate_scenarios(scenario_types, scenario_params)
            print(f"Generated {len(paths)} scenarios:")
            for path in paths:
                print(f"  {path}")
        
        elif args.command == "train":
            checkpoint_every = args.checkpoint_every or max(1000, args.timesteps // 10)
            
            # Determine GPU usage
            use_gpu = None  # Auto-detect
            if args.gpu:
                use_gpu = True
            elif args.no_gpu:
                use_gpu = False
            
            # Handle scenarios - default to head_on if none specified
            scenarios = args.scenario or ["head_on"]
            
            checkpoint = controller.train_model(
                scenario_names=scenarios,
                algo=args.algo,
                timesteps=args.timesteps,
                checkpoint_every=checkpoint_every,
                use_gpu=use_gpu,
                log_trajectories=args.log_trajectories
            )
            if checkpoint:
                print(f"Training completed. Checkpoint: {checkpoint}")
            else:
                print("Training failed.")
                sys.exit(1)
        
        elif args.command == "test-shifts":
            result = controller.run_shift_testing(
                checkpoint_path=args.checkpoint,
                scenario_name=args.scenario,
                episodes=args.episodes,
                targeted=args.targeted,
                generate_viz=args.viz,
                algo=args.algo
            )
            if result:
                print(f"Shift testing completed: {result}")
            else:
                print("Shift testing failed.")
                sys.exit(1)
        
        elif args.command == "analyze":
            results = controller.analyze_results(
                results_dir=args.results_dir,
                hallucination_analysis=not args.no_hallucination
            )
            if results:
                print("Analysis Results:")
                print(json.dumps(results, indent=2))
            else:
                print("Analysis failed.")
                sys.exit(1)
        
        elif args.command == "visualize":
            files = controller.visualize_data(
                trajectory_path=args.trajectory,
                results_dir=args.results_dir,
                output_dir=args.output_dir
            )
            print(f"Generated {len(files)} visualization files:")
            for file in files:
                print(f"  {file}")
        
        elif args.command == "full-pipeline":
            results = controller.full_pipeline(
                scenario_name=args.scenario,
                algo=args.algo,
                train_timesteps=args.train_timesteps,
                test_episodes=args.test_episodes,
                targeted_shifts=not args.no_targeted,
                generate_viz=not args.no_viz,
                log_trajectories=args.log_trajectories
            )
            
            print("Pipeline Results:")
            print(json.dumps(results, indent=2))
            
            if not results.get("success", False):
                sys.exit(1)
        
        elif args.command == "list":
            if args.resource == "scenarios":
                scenarios = controller.list_scenarios()
                print("Available scenarios:")
                for scenario in scenarios:
                    print(f"  {scenario}")
            
            elif args.resource == "checkpoints":
                checkpoints = controller.list_checkpoints()
                print("Available checkpoints:")
                for ckpt in checkpoints:
                    print(f"  {ckpt['name']} ({ckpt['modified']})")
            
            elif args.resource == "results":
                results_dirs = [d for d in controller.results_dir.iterdir() if d.is_dir()]
                print("Available results directories:")
                for results_dir in sorted(results_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
                    modified = datetime.fromtimestamp(results_dir.stat().st_mtime)
                    print(f"  {results_dir.name} ({modified.strftime('%Y-%m-%d %H:%M:%S')})")
    
    except KeyboardInterrupt:
        print("\\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()