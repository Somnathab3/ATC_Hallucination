#!/usr/bin/env python3
"""
CLI Training Test Script

Test training with different algorithms, scenarios, and timesteps.

Usage examples:
    python cli_train_test.py --algo PPO --scenario head_on --timesteps 5000
    python cli_train_test.py --algo SAC --scenario parallel --timesteps 10000
    python cli_train_test.py --scenario all --timesteps 2000  # Test all scenarios
    python cli_train_test.py --help
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def list_available_scenarios():
    """List all available scenario files."""
    scenarios_dir = os.path.join(project_root, "scenarios")
    if not os.path.exists(scenarios_dir):
        return []
    
    scenarios = []
    for f in os.listdir(scenarios_dir):
        if f.endswith('.json'):
            scenarios.append(f[:-5])  # Remove .json extension
    
    return sorted(scenarios)

def validate_parameters(algo, scenario, timesteps):
    """Validate input parameters."""
    errors = []
    
    # Validate algorithm
    if algo not in ["PPO", "SAC"]:
        errors.append(f"Invalid algorithm: {algo}. Must be PPO or SAC")
    
    # Validate scenario
    available_scenarios = list_available_scenarios()
    if scenario != "all" and scenario not in available_scenarios:
        errors.append(f"Invalid scenario: {scenario}. Available: {available_scenarios}")
    
    # Validate timesteps
    if timesteps < 1000:
        errors.append(f"Timesteps too low: {timesteps}. Minimum: 1000")
    elif timesteps > 10_000_000:
        errors.append(f"Timesteps too high: {timesteps}. Maximum: 10,000,000")
    
    return errors

def run_training_test(algo, scenario, timesteps, checkpoint_every=None):
    """Run training test with specified parameters."""
    print(f"🚀 Starting training test...")
    print(f"   Algorithm: {algo}")
    print(f"   Scenario: {scenario}")
    print(f"   Timesteps: {timesteps:,}")
    
    if checkpoint_every is None:
        checkpoint_every = max(1000, timesteps // 10)  # Checkpoint every 10% or minimum 1000
    
    print(f"   Checkpoint every: {checkpoint_every:,} steps")
    
    try:
        from src.training.train_frozen_scenario import train_frozen
        
        start_time = time.time()
        
        if scenario == "all":
            # Test all available scenarios
            available_scenarios = list_available_scenarios()
            results = {}
            
            for scen in available_scenarios:
                print(f"\n📝 Training scenario: {scen}")
                try:
                    ckpt = train_frozen(
                        repo_root=project_root,
                        algo=algo,
                        scenario_name=scen,
                        timesteps_total=timesteps,
                        checkpoint_every=checkpoint_every
                    )
                    results[scen] = {"success": True, "checkpoint": ckpt}
                    print(f"   ✅ {scen} completed successfully")
                except Exception as e:
                    results[scen] = {"success": False, "error": str(e)}
                    print(f"   ❌ {scen} failed: {e}")
            
            # Summary
            successful = [s for s, r in results.items() if r["success"]]
            failed = [s for s, r in results.items() if not r["success"]]
            
            print(f"\n📊 Summary:")
            print(f"   Successful: {len(successful)} - {successful}")
            print(f"   Failed: {len(failed)} - {failed}")
            
            return len(failed) == 0
            
        else:
            # Single scenario
            ckpt = train_frozen(
                repo_root=project_root,
                algo=algo,
                scenario_name=scenario,
                timesteps_total=timesteps,
                checkpoint_every=checkpoint_every
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\n✅ Training completed successfully!")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Final checkpoint: {ckpt}")
            
            # Verify results directory was created with proper naming
            expected_prefix = f"results_{algo}_{scenario}_"
            results_dirs = [d for d in os.listdir(project_root) if d.startswith(expected_prefix)]
            
            if results_dirs:
                latest_dir = max(results_dirs)
                print(f"   Results directory: {latest_dir}")
                
                # Check if models were saved
                models_dir = os.path.join(project_root, latest_dir, "models")
                if os.path.exists(models_dir):
                    model_files = os.listdir(models_dir)
                    print(f"   Model files: {len(model_files)} files saved")
                    print(f"✅ Model naming verification passed")
                else:
                    print(f"⚠️  Models directory not found")
            else:
                print(f"⚠️  Results directory with expected naming not found")
            
            return True
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="CLI Training Test Script for ATC Hallucination Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_train_test.py --algo PPO --scenario head_on --timesteps 5000
  python cli_train_test.py --algo SAC --scenario parallel --timesteps 10000
  python cli_train_test.py --scenario all --timesteps 2000
        """
    )
    
    parser.add_argument("--algo", "-a", choices=["PPO", "SAC"], default="PPO",
                       help="Algorithm to use (default: PPO)")
    
    parser.add_argument("--scenario", "-s", type=str, default="head_on",
                       help="Scenario to train on, or 'all' for all scenarios (default: head_on)")
    
    parser.add_argument("--timesteps", "-t", type=int, default=5000,
                       help="Number of timesteps to train (default: 5000)")
    
    parser.add_argument("--checkpoint-every", "-c", type=int,
                       help="Checkpoint every N steps (default: timesteps/10)")
    
    parser.add_argument("--list-scenarios", action="store_true",
                       help="List available scenarios and exit")
    
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate parameters, don't run training")
    
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    print("🎯 CLI Training Test Script")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # List scenarios if requested
    if args.list_scenarios:
        scenarios = list_available_scenarios()
        print("Available scenarios:")
        for scenario in scenarios:
            print(f"  • {scenario}")
        return
    
    # Validate parameters
    validation_errors = validate_parameters(args.algo, args.scenario, args.timesteps)
    
    if validation_errors:
        print("❌ Parameter validation failed:")
        for error in validation_errors:
            print(f"   • {error}")
        
        if not args.validate_only:
            available_scenarios = list_available_scenarios()
            print(f"\nAvailable scenarios: {available_scenarios}")
        
        sys.exit(1)
    
    print("✅ Parameter validation passed")
    
    if args.validate_only:
        print("🏁 Validation complete (--validate-only specified)")
        return
    
    # Run training test
    success = run_training_test(
        algo=args.algo,
        scenario=args.scenario,
        timesteps=args.timesteps,
        checkpoint_every=args.checkpoint_every
    )
    
    if success:
        print(f"\n🎉 CLI Training Test PASSED")
        sys.exit(0)
    else:
        print(f"\n💥 CLI Training Test FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()